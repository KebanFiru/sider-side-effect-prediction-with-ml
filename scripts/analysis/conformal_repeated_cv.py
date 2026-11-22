"""Repeated scaffold-grouped CV for the RF calibration set.

The single out-of-fold set of conformal_probs.py depends on one deterministic scaffold-to-fold
assignment. Here the assignment is repeated R times with shuffled scaffold-group order
(seeds 0..R-1; each group goes to the currently smallest fold), the RF out-of-fold probabilities are
recomputed, and the conformal analysis of conformal_analysis.py is run on every repeat. The test set
and the test-time RF (fitted on the full training set) stay fixed, so the spread across repeats is the
calibration-set variance.

Writes outputs/conformal/repeated_cv_<split>.json (per repeat + mean/std) and repeated_cv_summary.csv.
Usage: PYTHONPATH=. python -m scripts.analysis.conformal_repeated_cv [--repeats 10]
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from scripts.analysis import conformal_analysis as ca
from scripts.analysis.conformal_probs import N_FOLDS, load_split_sets

OUT = ca.OUT
KEYS = ["coverage", "cov_pos", "cov_neg", "mean_size", "singleton_rate"]


def shuffled_scaffold_folds(ids, seed, k=N_FOLDS):
    from src.train import _scaffold
    groups = defaultdict(list)
    for i, s in enumerate(ids):
        groups[_scaffold(s) or f"__invalid_{i}"].append(i)
    keys = list(groups)
    order = np.random.default_rng(seed).permutation(len(keys))
    sizes, fold = [0] * k, np.full(len(ids), -1)
    for j in order:
        f = int(np.argmin(sizes))
        fold[groups[keys[j]]] = f
        sizes[f] += len(groups[keys[j]])
    return fold


def run_split(split, repeats):
    import scripts.eval.classic as ec
    ec.N_JOBS = 12
    rf = np.load(f"{OUT}/rf_{split}.npz", allow_pickle=True)
    (tr_ids, y_tr), _, _ = load_split_sets(split)
    assert list(rf["oof_ids"]) == tr_ids
    X = ec.featurize(tr_ids, True)
    sim = ca.nn_similarity(tr_ids, list(rf["test_ids"]))
    p_te, y_te = rf["test_p"], rf["test_y"]
    runs = []
    for r in range(repeats):
        folds = shuffled_scaffold_folds(tr_ids, r)
        oof = np.zeros(y_tr.shape)
        for k in range(N_FOLDS):
            te = folds == k
            oof[te] = ec.fit_predict("rf", X[~te], y_tr[~te], [X[te]])[0]
        run = {"repeat": r, "fold_sizes": [int((folds == k).sum()) for k in range(N_FOLDS)]}
        for a in ca.ALPHAS:
            for mond in (False, True):
                i1, i0 = ca.conformal_sets(oof, y_tr, p_te, a, mond)
                m = ca.set_metrics(i1, i0, y_te)["pooled"]
                run[f"alpha{a}_{'mondrian' if mond else 'standard'}"] = {k: m[k] for k in KEYS}
            ad = ca.ad_analysis(sim, oof, y_tr, p_te, y_te, a)
            run[f"alpha{a}_ad_tertiles"] = [{k: d[k] for k in ("coverage", "cov_pos", "cov_neg", "mean_size")} for d in ad["tertiles"]]
            run[f"alpha{a}_spearman_rho"] = ad["spearman_sim_vs_molecule_coverage"]["rho"]
        runs.append(run)
        print(f"[{split}] repeat {r}: coverage a=0.1 {run['alpha0.1_standard']['coverage']:.3f}", flush=True)
    summ = {}
    for key in runs[0]:
        if key in ("repeat", "fold_sizes"):
            continue
        v = [r[key] for r in runs]
        if isinstance(v[0], dict):
            summ[key] = {k: [float(np.mean([x[k] for x in v])), float(np.std([x[k] for x in v]))] for k in v[0]}
        elif isinstance(v[0], list):
            summ[key] = [{k: [float(np.mean([x[b][k] for x in v])), float(np.std([x[b][k] for x in v]))] for k in v[0][b]}
                         for b in range(len(v[0]))]
        else:
            summ[key] = [float(np.mean(v)), float(np.std(v))]
    res = dict(split=split, repeats=repeats, std_is="population SD (ddof=0) across repeats", runs=runs, summary=summ)
    json.dump(res, open(f"{OUT}/repeated_cv_{split}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--splits", nargs="*", default=["scaffold", "ogb_scaffold"])
    a = ap.parse_args()
    results = [run_split(s, a.repeats) for s in a.splits]
    with open(f"{OUT}/repeated_cv_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "alpha", "variant", "metric", "mean", "std", "repeats"])
        for r in results:
            for key, v in r["summary"].items():
                alpha = key.split("_")[0].replace("alpha", "")
                rest = key.split("_", 1)[1]
                if isinstance(v, dict):
                    for k, (m, s) in v.items():
                        w.writerow([r["split"], alpha, rest, k, m, s, r["repeats"]])
                elif isinstance(v, list) and isinstance(v[0], dict):
                    for b, d in enumerate(v):
                        for k, (m, s) in d.items():
                            w.writerow([r["split"], alpha, f"{rest}_bin{b}", k, m, s, r["repeats"]])
                else:
                    w.writerow([r["split"], alpha, rest, "value", v[0], v[1], r["repeats"]])
    print("wrote repeated_cv_summary.csv")
