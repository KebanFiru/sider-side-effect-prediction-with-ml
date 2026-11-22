"""Conformal coverage, calibration and applicability-domain analysis (raw numbers).

Input : outputs/conformal/{rf,gin,yrand}_<split>.npz  (scripts.analysis.conformal_probs)
Output: outputs/conformal/results_<split>.json  and  outputs/conformal/results_flat.csv

Estimators (calibration set -> test set, 27 endpoints each):
  rf_oof     RF ECFP4+RDKit, calibrated on 5-fold scaffold-grouped out-of-fold
             probabilities of the training molecules (primary estimator)
  rf_stop    the same RF, calibrated on the early-stopping slice
  gin_stop   GIN A3+A5 five-seed ensemble, calibrated on the early-stopping slice
             (that slice drove epoch selection -> biased; secondary)
  blend_stop 0.5*GIN + 0.5*RF, calibrated on the early-stopping slice (secondary)

Conformal method: split conformal per endpoint, nonconformity score 1 - p(true
label); quantile ceil((n+1)(1-alpha))/n. Variants: "standard" (one threshold per
endpoint) and "mondrian" (class-conditional thresholds). alpha in {0.10, 0.20}.
Metrics: marginal coverage, coverage of positives and of negatives, mean set
size, singleton rate, empty-set rate.
Calibration: ECE with equal-mass bins (10), reliability curves, and Platt
rescaling fitted on a random half of the calibration set (conformal then uses
the other half, so the rescaler never sees its own calibration scores).
Applicability domain: nearest-neighbour Tanimoto similarity (ECFP4, 2048 bits) of
each test molecule to the training set; metrics per similarity bin (tertiles and
quintiles of the test set) and a similarity-threshold scan.
Y-randomization: RF fitted on permuted training labels, same conformal procedure
(calibration = early-stopping slice with true labels).

Usage: PYTHONPATH=. python -m scripts.analysis.conformal_analysis [--splits scaffold ogb_scaffold]
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

OUT = "outputs/conformal"
ALPHAS = [0.10, 0.20]
N_BINS = 10
EPS = 1e-6


# ---------------------------------------------------------------- conformal
def _quantile(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or k > n:
        return np.inf
    return np.sort(scores)[k - 1]


def conformal_sets(p_cal, y_cal, p_test, alpha, mondrian):
    """Boolean arrays (n_test, E) 'class 1 in set' and 'class 0 in set'."""
    n_test, E = p_test.shape
    in1 = np.zeros((n_test, E), bool)
    in0 = np.zeros((n_test, E), bool)
    for e in range(E):
        s1 = 1 - p_cal[:, e]            # score if the true label were 1
        s0 = p_cal[:, e]                # score if the true label were 0
        sc = np.where(y_cal[:, e] == 1, s1, s0)
        if mondrian:
            q1 = _quantile(sc[y_cal[:, e] == 1], alpha)
            q0 = _quantile(sc[y_cal[:, e] == 0], alpha)
        else:
            q1 = q0 = _quantile(sc, alpha)
        in1[:, e] = (1 - p_test[:, e]) <= q1
        in0[:, e] = p_test[:, e] <= q0
    return in1, in0


def set_metrics(in1, in0, y):
    """Per-endpoint and pooled metrics for the given rows."""
    cov = np.where(y == 1, in1, in0)
    size = in1.astype(int) + in0.astype(int)
    per = {}
    def agg(mask_axis):
        return mask_axis
    E = y.shape[1]
    def one(c, yy, s):
        pos, neg = yy == 1, yy == 0
        return dict(coverage=float(c.mean()) if c.size else np.nan,
                    cov_pos=float(c[pos].mean()) if pos.any() else np.nan,
                    cov_neg=float(c[neg].mean()) if neg.any() else np.nan,
                    mean_size=float(s.mean()) if s.size else np.nan,
                    singleton_rate=float((s == 1).mean()) if s.size else np.nan,
                    empty_rate=float((s == 0).mean()) if s.size else np.nan,
                    n=int(c.size), n_pos=int(pos.sum()))
    per["endpoints"] = [one(cov[:, e], y[:, e], size[:, e]) for e in range(E)]
    per["pooled"] = one(cov.ravel(), y.ravel(), size.ravel())
    keys = ["coverage", "cov_pos", "cov_neg", "mean_size", "singleton_rate", "empty_rate"]
    per["macro"] = {k: float(np.nanmean([d[k] for d in per["endpoints"]])) for k in keys}
    return per


# -------------------------------------------------------------- calibration
def ece_equal_mass(p, y, n_bins=N_BINS):
    n = len(p)
    if n == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, min(n_bins, n))
    return float(sum(len(b) / n * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def reliability(p, y, n_bins=N_BINS):
    order = np.argsort(p, kind="mergesort")
    return [dict(mean_pred=float(p[b].mean()), frac_pos=float(y[b].mean()), n=int(len(b)))
            for b in np.array_split(order, n_bins) if len(b)]


def platt_fit(p, y):
    """Per-endpoint Platt scaling; returns a function, identity if impossible."""
    if y.sum() < 2 or (1 - y).sum() < 2:
        return lambda q: q
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))[:, None]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return lambda q: lr.predict_proba(np.log(np.clip(q, EPS, 1 - EPS) / (1 - np.clip(q, EPS, 1 - EPS)))[:, None])[:, 1]


def ece_report(p, y):
    E = y.shape[1]
    per = [ece_equal_mass(p[:, e], y[:, e]) for e in range(E)]
    return dict(ece_per_endpoint=per, ece_macro=float(np.nanmean(per)),
                ece_pooled=ece_equal_mass(p.ravel(), y.ravel()),
                reliability_pooled=reliability(p.ravel(), y.ravel()))


def rescale_experiment(p_cal, y_cal, p_te, y_te, seed=0):
    """Platt fitted on one half of the calibration set, conformal calibrated on the other."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_cal))
    fit_i, cal_i = idx[: len(idx) // 2], idx[len(idx) // 2:]
    p_cal2, p_te2 = np.zeros((len(cal_i), p_cal.shape[1])), np.zeros_like(p_te)
    for e in range(p_cal.shape[1]):
        f = platt_fit(p_cal[fit_i, e], y_cal[fit_i, e])
        p_cal2[:, e], p_te2[:, e] = f(p_cal[cal_i, e]), f(p_te[:, e])
    out = dict(n_fit=int(len(fit_i)), n_cal=int(len(cal_i)), calibration=ece_report(p_te2, y_te), conformal={})
    # reference without rescaling on the same conformal half
    for a in ALPHAS:
        for name, pc, pt in (("raw_half", p_cal[cal_i], p_te), ("platt", p_cal2, p_te2)):
            i1, i0 = conformal_sets(pc, y_cal[cal_i], pt, a, False)
            m = set_metrics(i1, i0, y_te)
            out["conformal"][f"{name}_alpha{a}"] = dict(macro=m["macro"], pooled=m["pooled"])
    return out


# -------------------------------------------------------------------- data
def estimators(split):
    rf = np.load(f"{OUT}/rf_{split}.npz", allow_pickle=True)
    gin = np.load(f"{OUT}/gin_{split}.npz", allow_pickle=True)
    assert np.array_equal(rf["stop_ids"], gin["stop_ids"]) and np.array_equal(rf["test_ids"], gin["test_ids"]), \
        "RF and GIN rows are not aligned"
    assert np.array_equal(rf["test_y"], gin["test_y"]) and np.array_equal(rf["stop_y"], gin["stop_y"])
    ests = {
        "rf_oof": (rf["oof_p"], rf["oof_y"], rf["test_p"], rf["test_y"]),
        "rf_stop": (rf["stop_p"], rf["stop_y"], rf["test_p"], rf["test_y"]),
        "gin_stop": (gin["stop_p"], gin["stop_y"], gin["test_p"], gin["test_y"]),
        "blend_stop": (0.5 * gin["stop_p"] + 0.5 * rf["stop_p"], gin["stop_y"],
                       0.5 * gin["test_p"] + 0.5 * rf["test_p"], gin["test_y"]),
    }
    return ests, rf


def nn_similarity(tr_ids, te_ids):
    from rdkit import Chem, DataStructs
    from scripts.eval.classic import _MORGAN
    fps = [_MORGAN.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr_ids]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(_MORGAN.GetFingerprint(Chem.MolFromSmiles(s)), fps))
                     for s in te_ids])


def ad_analysis(sim, p_cal, y_cal, p_te, y_te, alpha, mondrian=False):
    in1, in0 = conformal_sets(p_cal, y_cal, p_te, alpha, mondrian)
    cov = np.where(y_te == 1, in1, in0)
    res = {"tertiles": [], "quintiles": [], "threshold_scan": []}
    for name, qs in (("tertiles", [1 / 3, 2 / 3]), ("quintiles", [.2, .4, .6, .8])):
        bins = np.digitize(sim, np.quantile(sim, qs))
        for b in range(len(qs) + 1):
            m = bins == b
            met = set_metrics(in1[m], in0[m], y_te[m])["pooled"]
            met.update(bin=b, n_molecules=int(m.sum()), sim_mean=float(sim[m].mean()),
                       sim_min=float(sim[m].min()), sim_max=float(sim[m].max()),
                       ece_pooled=ece_equal_mass(p_te[m].ravel(), y_te[m].ravel()))
            res[name].append(met)
    for t in np.unique(np.round(np.quantile(sim, np.linspace(0, .9, 19)), 4)):
        m = sim >= t
        met = set_metrics(in1[m], in0[m], y_te[m])["pooled"]
        res["threshold_scan"].append(dict(threshold=float(t), retained_fraction=float(m.mean()),
                                          coverage=met["coverage"], cov_pos=met["cov_pos"], cov_neg=met["cov_neg"],
                                          mean_size=met["mean_size"]))
    rec = [r for r in res["threshold_scan"] if r["coverage"] >= 1 - alpha]
    rec_pos = [r for r in res["threshold_scan"] if r["cov_pos"] >= 1 - alpha]
    res["first_threshold_coverage_ok"] = rec[0] if rec else None
    res["first_threshold_cov_pos_ok"] = rec_pos[0] if rec_pos else None
    per_mol = cov.mean(1)
    rho = spearmanr(sim, per_mol)
    res["spearman_sim_vs_molecule_coverage"] = dict(rho=float(rho.statistic), p=float(rho.pvalue))
    return res


# --------------------------------------------------------------------- run
def run_split(split):
    from scripts.analysis.conformal_probs import load_split_sets
    ests, rf = estimators(split)
    (tr_ids, _), _, _ = load_split_sets(split)
    sim = nn_similarity(tr_ids, list(rf["test_ids"]))
    res = dict(split=split, n_train=len(tr_ids), n_test=int(len(rf["test_ids"])),
               calibration_set_sizes={"rf_oof": int(len(rf["oof_ids"])), "stopping_slice": int(len(rf["stop_ids"]))},
               notes={"gin_stop": "stopping slice was used for early stopping / epoch selection: biased",
                      "blend_stop": "contains the biased GIN probabilities",
                      "rf_oof": "5-fold scaffold-grouped CV, one repeat (src.train.assign_folds)"},
               test_nn_similarity=dict(mean=float(sim.mean()), median=float(np.median(sim)),
                                       min=float(sim.min()), max=float(sim.max())),
               estimators={})
    for name, (p_cal, y_cal, p_te, y_te) in ests.items():
        r = dict(n_cal=int(len(y_cal)), calibration_test=ece_report(p_te, y_te), conformal={}, ad={}, rescale={})
        for a in ALPHAS:
            for mond in (False, True):
                i1, i0 = conformal_sets(p_cal, y_cal, p_te, a, mond)
                r["conformal"][f"alpha{a}_{'mondrian' if mond else 'standard'}"] = set_metrics(i1, i0, y_te)
            r["ad"][f"alpha{a}"] = ad_analysis(sim, p_cal, y_cal, p_te, y_te, a)
        r["rescale"] = rescale_experiment(p_cal, y_cal, p_te, y_te)
        res["estimators"][name] = r
        print(f"[{split}] {name}: marginal coverage (standard) a=0.1: "
              f"{r['conformal']['alpha0.1_standard']['pooled']['coverage']:.3f}", flush=True)
    # Y-randomization
    yr = f"{OUT}/yrand_{split}.npz"
    if os.path.exists(yr):
        d = np.load(yr, allow_pickle=True)
        res["yrand"] = {}
        for a in ALPHAS:
            runs = []
            for k in range(d["stop_p"].shape[0]):
                i1, i0 = conformal_sets(d["stop_p"][k], d["stop_y"], d["test_p"][k], a, False)
                m = set_metrics(i1, i0, d["test_y"])["pooled"]
                runs.append({kk: m[kk] for kk in ("coverage", "cov_pos", "cov_neg", "mean_size", "singleton_rate", "empty_rate")})
            res["yrand"][f"alpha{a}"] = dict(
                per_permutation=runs,
                mean={k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]},
                std={k: float(np.nanstd([r[k] for r in runs])) for k in runs[0]},
                calibration="early-stopping slice with true labels; null model trained on permuted labels")
    json.dump(res, open(f"{OUT}/results_{split}.json", "w"), indent=1)
    return res


def flatten(results):
    import gzip
    with gzip.open("data/raw/sider.csv.gz", "rt") as fh:
        names = next(csv.reader(fh))[1:]
    rows = []
    for res in results:
        for est, r in res["estimators"].items():
            for key, m in r["conformal"].items():
                a, v = key.replace("alpha", "").split("_")
                for e, d in enumerate(m["endpoints"]):
                    rows.append([res["split"], est, a, v, "all", f"{e}:{names[e]}", *[d[k] for k in
                                 ("coverage", "cov_pos", "cov_neg", "mean_size", "singleton_rate", "empty_rate", "n", "n_pos")]])
                d = m["pooled"]
                rows.append([res["split"], est, a, v, "all", "pooled", *[d[k] for k in
                             ("coverage", "cov_pos", "cov_neg", "mean_size", "singleton_rate", "empty_rate", "n", "n_pos")]])
            for a, ad in r["ad"].items():
                for name in ("tertiles", "quintiles"):
                    for d in ad[name]:
                        rows.append([res["split"], est, a.replace("alpha", ""), "standard", f"ad_{name}_bin{d['bin']}", "pooled",
                                     *[d[k] for k in ("coverage", "cov_pos", "cov_neg", "mean_size", "singleton_rate",
                                                      "empty_rate", "n", "n_pos")]])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["scaffold", "ogb_scaffold"])
    a = ap.parse_args()
    results = [run_split(s) for s in a.splits]
    with open(f"{OUT}/results_flat.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "estimator", "alpha", "variant", "subset", "endpoint", "coverage", "cov_pos", "cov_neg",
                    "mean_size", "singleton_rate", "empty_rate", "n_cells", "n_pos"])
        w.writerows(flatten(results))
    print(f"wrote {OUT}/results_<split>.json and {OUT}/results_flat.csv")
