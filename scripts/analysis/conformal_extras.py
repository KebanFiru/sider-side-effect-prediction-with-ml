"""Extra conformal statistics for the rf_oof estimator (RF ECFP4+RDKit, calibration = scaffold-grouped OOF).

Methods: standard (one threshold per endpoint), mondrian (class-conditional), gated Mondrian with
tau in {0.10, 0.20} (Mondrian for endpoints whose CALIBRATION prevalence lies in [tau, 1-tau], standard
otherwise -- the decision uses only calibration labels), and margin (score = -(2y-1)(p-0.5); it is a
monotone transform of 1-p(y), so it is expected to reproduce `standard` exactly; reported as a check).
alpha in {0.10, 0.20}; two splits (scaffold-204, OGB/Hu).

Per method/alpha, POOLED over molecules x 27 endpoints on the test set:
  coverage, coverage of positives / negatives, mean set size, singleton rate, and the selective numbers
  on singleton sets: accuracy, PPV (singleton {1} that is truly positive), NPV (singleton {0} truly
  negative), share of {1} / {0} singletons.
  * 95% bootstrap CI over TEST MOLECULES (2000 resamples of rows, RandomState(0), same procedure as
    src.eval_metrics.bootstrap_ci, vectorised; calibration thresholds stay fixed).
  * mean +- SD (population) over 10 repeated scaffold-grouped CV calibration sets (shuffle seeds 0-9).
Per endpoint: prevalence (calibration / test), test AUC, singleton rate, singleton accuracy, chosen method.
Per molecule: simultaneous coverage (all 27 endpoints covered), mean number covered, and the distribution
of the number of singleton endpoints.

Outputs (outputs/conformal/): results_boot_<split>.json, results_gated_<split>.json,
results_molecule_level_<split>.json, results_selective_<split>.json, results_extras_summary.csv
Usage: PYTHONPATH=. python -m scripts.analysis.conformal_extras
"""
import csv
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import scripts.eval.classic as ec
from scripts.analysis.conformal_probs import N_FOLDS, load_split_sets
from scripts.analysis.conformal_repeated_cv import shuffled_scaffold_folds
from scripts.analysis.conformal_analysis import _quantile

OUT = "outputs/conformal"
ALPHAS = [0.10, 0.20]
METHODS = ["standard", "mondrian", "gated0.10", "gated0.20", "margin"]
N_BOOT = 2000
REPEATS = 10
MET = ["coverage", "cov_pos", "cov_neg", "mean_size", "singleton_rate", "singleton_accuracy", "ppv", "npv",
       "share_singleton_pos", "share_singleton_neg"]


def thresholds(p_cal, y_cal, alpha, method):
    """q1[E], q0[E] (inf -> always in set) and the per-endpoint choice."""
    E = p_cal.shape[1]
    q1, q0, choice = np.zeros(E), np.zeros(E), []
    for e in range(E):
        s = np.where(y_cal[:, e] == 1, 1 - p_cal[:, e], p_cal[:, e])
        prev = y_cal[:, e].mean()
        if method == "margin":
            sm = -(2 * y_cal[:, e] - 1) * (p_cal[:, e] - 0.5)   # = (1-p(y)) - 0.5 : same ranking
            qm = _quantile(sm, alpha)
            q1[e], q0[e] = qm + 0.5, qm + 0.5                    # back to the 1-p(y) scale
            choice.append("margin")
            continue
        mond = method == "mondrian"
        if method.startswith("gated"):
            tau = float(method[5:])
            mond = tau <= prev <= 1 - tau
        if mond:
            q1[e], q0[e] = _quantile(s[y_cal[:, e] == 1], alpha), _quantile(s[y_cal[:, e] == 0], alpha)
        else:
            q1[e] = q0[e] = _quantile(s, alpha)
        choice.append("mondrian" if mond else "standard")
    return q1, q0, choice


def cell_stats(p, y, q1, q0):
    """Per-molecule x endpoint indicator arrays."""
    in1, in0 = (1 - p) <= q1, p <= q0
    cov = np.where(y == 1, in1, in0)
    size = in1.astype(int) + in0.astype(int)
    single = size == 1
    pred1, pred0 = single & in1, single & in0
    return dict(cov=cov, size=size, single=single, pos=(y == 1), neg=(y == 0),
                cov_pos=cov & (y == 1), cov_neg=cov & (y == 0), correct=single & cov,
                pred1=pred1, tp=pred1 & (y == 1), pred0=pred0, tn=pred0 & (y == 0)), in1, in0


def reduce(st, W=None, axis=None):
    """Pooled metrics; W = weights over molecules (None -> unweighted). Returns dict of arrays/floats."""
    def S(a):
        a = a.astype(float)
        a = a.sum(1) if a.ndim == 2 else a
        return a.sum() if W is None else W @ a
    div = lambda a, b: a / np.maximum(b, 1e-12)
    n = S(np.ones_like(st["cov"]))
    return dict(coverage=div(S(st["cov"]), n), cov_pos=div(S(st["cov_pos"]), S(st["pos"])),
                cov_neg=div(S(st["cov_neg"]), S(st["neg"])), mean_size=div(S(st["size"]), n),
                singleton_rate=div(S(st["single"]), n), singleton_accuracy=div(S(st["correct"]), S(st["single"])),
                ppv=div(S(st["tp"]), S(st["pred1"])), npv=div(S(st["tn"]), S(st["pred0"])),
                share_singleton_pos=div(S(st["pred1"]), n), share_singleton_neg=div(S(st["pred0"]), n))


def run_split(split, W, X_tr, tr_ids, y_tr, rf):
    p_te, y_te = rf["test_p"], rf["test_y"]
    oof, y_oof = rf["oof_p"], rf["oof_y"]
    E = y_te.shape[1]
    auc = [roc_auc_score(y_te[:, e], p_te[:, e]) for e in range(E)]
    boot, gated, molv, sel = {"split": split, "n_boot": N_BOOT, "n_test": int(len(y_te)),
                              "procedure": "rows resampled with RandomState(0), 2000 resamples, 95% percentile CI; thresholds fixed"}, \
        {"split": split, "repeats": REPEATS, "std": "population SD over repeated scaffold-grouped CV calibration sets"}, \
        {"split": split}, {"split": split, "endpoints": {}}
    rep_oof = []
    for r in range(REPEATS):
        folds = shuffled_scaffold_folds(tr_ids, r)
        o = np.zeros(y_tr.shape)
        for k in range(N_FOLDS):
            te = folds == k
            o[te] = ec.fit_predict("rf", X_tr[~te], y_tr[~te], [X_tr[te]])[0]
        rep_oof.append(o)
    for a in ALPHAS:
        for m in METHODS:
            key = f"alpha{a}_{m}"
            q1, q0, choice = thresholds(oof, y_oof, a, m)
            st, in1, in0 = cell_stats(p_te, y_te, q1, q0)
            point = reduce(st)
            bw = reduce(st, W)
            boot[key] = {k: dict(point=float(point[k]), ci_lo=float(np.nanpercentile(bw[k], 2.5)),
                                 ci_hi=float(np.nanpercentile(bw[k], 97.5))) for k in MET}
            runs = []
            for o in rep_oof:
                q1r, q0r, _ = thresholds(o, y_oof, a, m)
                runs.append(reduce(cell_stats(p_te, y_te, q1r, q0r)[0]))
            gated[key] = {k: dict(mean=float(np.nanmean([x[k] for x in runs])), std=float(np.nanstd([x[k] for x in runs])))
                          for k in MET}
            gated[key]["endpoint_choice"] = choice
            single_e = st["single"].mean(0)
            acc_e = st["correct"].sum(0) / np.maximum(st["single"].sum(0), 1)
            sel["endpoints"][key] = [dict(endpoint_idx=e, prevalence_cal=float(y_oof[:, e].mean()),
                                          prevalence_test=float(y_te[:, e].mean()), test_auc=float(auc[e]),
                                          singleton_rate=float(single_e[e]), singleton_accuracy=float(acc_e[e]),
                                          n_singletons=int(st["single"][:, e].sum()), method=choice[e]) for e in range(E)]
            allc = st["cov"].all(1)
            nsing = st["single"].sum(1)
            rng = np.random.RandomState(0)
            idxs = [rng.randint(0, len(y_te), size=len(y_te)) for _ in range(N_BOOT)]
            bs = np.array([allc[i].mean() for i in idxs])
            molv[key] = dict(simultaneous_coverage=float(allc.mean()),
                             simultaneous_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             mean_endpoints_covered=float(st["cov"].sum(1).mean()),
                             n_endpoints=int(E),
                             singleton_endpoints=dict(mean=float(nsing.mean()), median=float(np.median(nsing)),
                                                      min=int(nsing.min()), max=int(nsing.max())),
                             independence_reference=float((1 - a) ** E))
    for name, obj in (("boot", boot), ("gated", gated), ("molecule_level", molv), ("selective", sel)):
        json.dump(obj, open(f"{OUT}/results_{name}_{split}.json", "w"), indent=1)
    return boot, gated, molv


def main():
    ec.N_JOBS = 12
    rows = []
    for split in ("scaffold", "ogb_scaffold"):
        rf = np.load(f"{OUT}/rf_{split}.npz", allow_pickle=True)
        (tr_ids, y_tr), _, _ = load_split_sets(split)
        n = len(rf["test_y"])
        rng = np.random.RandomState(0)
        W = np.stack([np.bincount(rng.randint(0, n, size=n), minlength=n) for _ in range(N_BOOT)]).astype(float)
        X_tr = ec.featurize(tr_ids, True)
        boot, gated, molv = run_split(split, W, X_tr, tr_ids, y_tr, rf)
        for a in ALPHAS:
            for m in METHODS:
                key = f"alpha{a}_{m}"
                for k in MET:
                    b, g = boot[key][k], gated[key][k]
                    rows.append([split, a, m, k, b["point"], b["ci_lo"], b["ci_hi"], g["mean"], g["std"]])
                rows.append([split, a, m, "simultaneous_coverage", molv[key]["simultaneous_coverage"],
                             *molv[key]["simultaneous_ci"], "", ""])
        print(f"[{split}] done", flush=True)
    with open(f"{OUT}/results_extras_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "alpha", "method", "metric", "test_point", "test_boot_ci_lo", "test_boot_ci_hi",
                    "repeated_cv_mean", "repeated_cv_sd"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
