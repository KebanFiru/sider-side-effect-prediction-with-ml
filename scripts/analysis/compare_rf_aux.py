"""Paired bootstrap comparison (macro-AUC) of RF + auxiliary-data stacking
(ecfp_rdkit_aux / _auxsmall) against plain RF ECFP+RDKit on the same split.
Splits: scaffold-204, OGB/Hu scaffold, random 0-2 (mean +/- std).
Output: outputs/eval/rf_aux_comparison.json
Usage: PYTHONPATH=. python -m scripts.analysis.compare_rf_aux
"""
import json

import numpy as np

from src.eval_metrics import macro_auc, paired_bootstrap_pvalue

D = "outputs/eval/ginv2_heldout_ensemble_classic_rf_"
SPLITS = {"scaffold204": "", "ogb": "_ogb",
          **{f"random{s}": f"_random_seed{s}" for s in range(3)}}


def main():
    out = {}
    for variant in ("ecfp_rdkit_aux", "ecfp_rdkit_auxsmall"):
        out[variant] = {}
        for split, suf in SPLITS.items():
            a = np.load(f"{D}{variant}{suf}_probs.npz")
            b = np.load(f"{D}ecfp_rdkit{suf}_probs.npz")
            assert np.array_equal(a["targets"], b["targets"])
            r = paired_bootstrap_pvalue(a["targets"], a["probs"], b["probs"], macro_auc, n_boot=2000)
            out[variant][split] = {"auc_aux": macro_auc(a["targets"], a["probs"]),
                                   "auc_plain": macro_auc(b["targets"], b["probs"]),
                                   "diff": r["observed_diff"], "p": r["p_value"]}
            o = out[variant][split]
            print(f"{variant:22s} {split:12s} RF={o['auc_plain']:.3f}  RF+aux={o['auc_aux']:.3f}  "
                  f"diff={o['diff']:+.3f}  p={o['p']:.3f}")
        rnd = [out[variant][f"random{s}"] for s in range(3)]
        for k in ("auc_plain", "auc_aux", "diff"):
            v = [r[k] for r in rnd]
            out[variant][f"random_mean_{k}"] = [float(np.mean(v)), float(np.std(v))]
        print(f"{variant:22s} random mean RF={np.mean([r['auc_plain'] for r in rnd]):.3f}  "
              f"RF+aux={np.mean([r['auc_aux'] for r in rnd]):.3f} +/- {np.std([r['auc_aux'] for r in rnd]):.3f}  "
              f"diff={np.mean([r['diff'] for r in rnd]):+.3f}")
    json.dump(out, open("outputs/eval/rf_aux_comparison.json", "w"), indent=2)


if __name__ == "__main__":
    main()
