"""Scaffold-aware uncertainty for the conformal results (rf_oof, both splits, alpha 0.10/0.20,
methods standard / mondrian / gated0.10 / gated0.20) and the exact majority baseline.

1. Murcko scaffolds of the test molecules (src.train._scaffold; acyclic molecules share one label) and the
   number of test molecules per scaffold.
2. Scaffold-balanced coverage: coverage within each scaffold, then the equal-weight mean over scaffolds
   (pooled, positive cells, negative cells), next to the molecule-weighted values.
3. Cluster bootstrap: scaffolds (not molecules) are resampled, 2000 times, RandomState(0), 95% percentile
   interval, calibration thresholds fixed; coverage, positive/negative coverage, singleton rate. Interval
   widths are compared with the molecule bootstrap of results_boot_<split>.json.
4. Exact majority baseline: in exactly the (molecule, endpoint) cells that receive a singleton set under the
   method, accuracy of predicting the calibration-majority label of the endpoint, next to the accuracy of the
   conformal singleton itself.

Outputs: outputs/conformal/results_cluster_<split>.json, results_majority_baseline_<split>.json,
results_cluster_summary.csv. Usage: PYTHONPATH=. python -m scripts.analysis.conformal_cluster
"""
import csv
import json
from collections import Counter

import numpy as np

from scripts.analysis.conformal_extras import ALPHAS, METHODS, N_BOOT, OUT, cell_stats, reduce, thresholds
from src.train import _scaffold

USE = [m for m in METHODS if m != "margin"]
KEYS = ["coverage", "cov_pos", "cov_neg", "singleton_rate"]


def main():
    rows = []
    for split in ("scaffold", "ogb_scaffold"):
        rf = np.load(f"{OUT}/rf_{split}.npz", allow_pickle=True)
        p_te, y_te, oof, y_oof = rf["test_p"], rf["test_y"], rf["oof_p"], rf["oof_y"]
        ids = [str(s) for s in rf["test_ids"]]
        scaf = [(_scaffold(s) or "__acyclic__") or "__acyclic__" for s in ids]
        labels, inv = np.unique(scaf, return_inverse=True)
        n_sc = len(labels)
        sizes = np.bincount(inv)
        sc_info = dict(n_test_molecules=len(ids), n_scaffolds=int(n_sc),
                       n_singleton_scaffolds=int((sizes == 1).sum()), max_molecules_per_scaffold=int(sizes.max()),
                       mean_molecules_per_scaffold=float(sizes.mean()), size_distribution=dict(Counter(sizes.tolist())),
                       acyclic_molecules=int((np.array(scaf) == "__acyclic__").sum()),
                       largest_scaffolds=[[labels[i], int(sizes[i])] for i in np.argsort(-sizes)[:5]])
        boot_mol = json.load(open(f"{OUT}/results_boot_{split}.json"))
        rng = np.random.RandomState(0)
        Wc = np.zeros((N_BOOT, len(ids)))
        for b in range(N_BOOT):
            pick = np.bincount(rng.randint(0, n_sc, size=n_sc), minlength=n_sc)
            Wc[b] = pick[inv]
        res, maj = {"split": split, "scaffolds": sc_info}, {"split": split,
            "definition": "cells = (molecule, endpoint) pairs that receive a singleton set under the method; baseline = predict the calibration-majority label of the endpoint"}, 
        for a in ALPHAS:
            for m in USE:
                key = f"alpha{a}_{m}"
                q1, q0, choice = thresholds(oof, y_oof, a, m)
                st, in1, in0 = cell_stats(p_te, y_te, q1, q0)
                mol = reduce(st)
                bal = {}
                for k, num, den in (("coverage", "cov", None), ("cov_pos", "cov_pos", "pos"), ("cov_neg", "cov_neg", "neg")):
                    vals = []
                    for s in range(n_sc):
                        idx = inv == s
                        d = idx.sum() * y_te.shape[1] if den is None else st[den][idx].sum()
                        if d > 0:
                            vals.append(st[num][idx].sum() / d)
                    bal[k] = dict(scaffold_balanced=float(np.mean(vals)), molecule_weighted=float(mol[k]), n_scaffolds_used=len(vals))
                cb = reduce(st, Wc)
                clus = {}
                for k in KEYS:
                    lo, hi = np.nanpercentile(cb[k], [2.5, 97.5])
                    mb = boot_mol[key][k]
                    clus[k] = dict(point=float(mol[k]), cluster_ci=[float(lo), float(hi)], cluster_width=float(hi - lo),
                                   molecule_ci=[mb["ci_lo"], mb["ci_hi"]], molecule_width=float(mb["ci_hi"] - mb["ci_lo"]),
                                   width_ratio=float((hi - lo) / max(mb["ci_hi"] - mb["ci_lo"], 1e-12)))
                res[key] = dict(balanced=bal, cluster_bootstrap=clus)
                # exact majority baseline on the singleton cells
                maj_label = (y_oof.mean(0) >= 0.5).astype(int)[None, :]
                cells = st["single"]
                n_cells = int(cells.sum())
                base_acc = float(((y_te == maj_label) & cells).sum() / max(n_cells, 1))
                conf_acc = float(st["correct"].sum() / max(n_cells, 1))
                maj[key] = dict(n_singleton_cells=n_cells, conformal_singleton_accuracy=conf_acc,
                                majority_baseline_accuracy=base_acc, difference=conf_acc - base_acc)
                rows.append([split, a, m, n_cells, conf_acc, base_acc, conf_acc - base_acc,
                             bal["coverage"]["scaffold_balanced"], bal["coverage"]["molecule_weighted"],
                             clus["coverage"]["cluster_width"], clus["coverage"]["molecule_width"], clus["coverage"]["width_ratio"],
                             clus["cov_neg"]["width_ratio"], clus["singleton_rate"]["width_ratio"]])
        json.dump(res, open(f"{OUT}/results_cluster_{split}.json", "w"), indent=1)
        json.dump(maj, open(f"{OUT}/results_majority_baseline_{split}.json", "w"), indent=1)
        print(f"[{split}] {n_sc} scaffolds for {len(ids)} test molecules", flush=True)
    with open(f"{OUT}/results_cluster_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "alpha", "method", "n_singleton_cells", "conformal_singleton_accuracy", "majority_baseline_accuracy",
                    "accuracy_minus_baseline", "coverage_scaffold_balanced", "coverage_molecule_weighted",
                    "coverage_ci_width_cluster", "coverage_ci_width_molecule", "coverage_width_ratio", "cov_neg_width_ratio",
                    "singleton_rate_width_ratio"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
