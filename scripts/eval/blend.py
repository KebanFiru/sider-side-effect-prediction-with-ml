"""GNN + RF blend: the mean of the two models' clean-test probabilities.
No training; the saved predictions of both models
(outputs/eval/ginv2_heldout_ensemble_{tag}_probs.npz) are combined.

The blending rule is fixed in advance: an equal-weight per-task mean of the
probabilities (0.5 GNN + 0.5 RF). The weight is NOT chosen on the test set
(that would make the score optimistic). Because the two models' probability
scales can differ, the macro-AUC of the per-task rank mean is also written,
for information only; the decision uses the probability mean.

Paired bootstrap p: blend vs GNN, blend vs RF, and on the scaffold-204 split
also vs the baseline. Both models must cover the same molecules in the same
order (asserted).

Usage:
  PYTHONPATH=. python -m scripts.eval.blend <gnn_tag> <rf_tag> <output_tag>
  Scaffold : ... eval_gnn_rf_blend 2_3 classic_rf_ecfp_rdkit blend_2_3_rf
  Random   : ... eval_gnn_rf_blend 2_3_random classic_rf_ecfp_rdkit_random_seed0 blend_2_3_rf_random
"""
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

from src.eval_metrics import (
    bootstrap_ci,
    macro_auc,
    macro_aupr,
    macro_balanced_accuracy,
    macro_f1,
    macro_report,
    paired_bootstrap_pvalue,
)

N_BOOT = 2000
W_GNN = 0.5
OUT_DIR = os.environ.get("EVAL_DIR", "outputs/eval")
BASELINE_TAG = "baseline"


def load(tag):
    d = np.load(f"{OUT_DIR}/ginv2_heldout_ensemble_{tag}_probs.npz")
    meta = json.load(open(f"{OUT_DIR}/ginv2_heldout_ensemble_{tag}.json"))
    return d["probs"], d["targets"], meta


def rank_mean(a, b):
    # Per-task rank scaled to [0, 1]; independent of the probability scale.
    n = a.shape[0]
    ra = np.apply_along_axis(rankdata, 0, a) / n
    rb = np.apply_along_axis(rankdata, 0, b) / n
    return W_GNN * ra + (1 - W_GNN) * rb


def main():
    gnn_tag, rf_tag, tag = sys.argv[1:4]
    p_gnn, y, m_gnn = load(gnn_tag)
    p_rf, y_rf, m_rf = load(rf_tag)
    assert np.array_equal(y, y_rf), "GNN and RF cover different molecules / different order"
    split_gnn = m_gnn.get("heldout_split", "scaffold")
    split_rf = m_rf.get("heldout_split", "scaffold")
    assert split_gnn == split_rf, f"split types differ: {split_gnn} / {split_rf}"

    probs = W_GNN * p_gnn + (1 - W_GNN) * p_rf
    rep = macro_report(y, probs, threshold=0.5)
    print(f"=== [{tag}] GNN ({gnn_tag}) + RF ({rf_tag}), weight {W_GNN}/{1 - W_GNN}, "
          f"{split_gnn} split, n={y.shape[0]} molecules ===")
    print(f"macro_AUC={rep.macro_auc:.3f}  macro_AUPR={rep.macro_aupr:.3f}  "
          f"balanced_acc={rep.balanced_accuracy:.3f}  F1_macro={rep.f1_macro:.3f}")
    cis = {}
    for name, fn in [("macro_auc", macro_auc), ("macro_aupr", macro_aupr),
                     ("balanced_accuracy", macro_balanced_accuracy), ("f1_macro", macro_f1)]:
        cis[name] = bootstrap_ci(y, probs, fn, n_boot=N_BOOT)
        print(f"  {name}: {cis[name]['point']:.3f}  95% CI=[{cis[name]['ci_lo']:.3f}, {cis[name]['ci_hi']:.3f}]")
    rank_auc = macro_auc(y, rank_mean(p_gnn, p_rf))
    print(f"  (info) rank-mean macro_AUC={rank_auc:.3f}")

    out = {"tag": tag, "gnn_tag": gnn_tag, "rf_tag": rf_tag, "w_gnn": W_GNN,
           "n_molecules": int(y.shape[0]), "n_tasks_total": rep.n_tasks_total,
           "n_tasks_excluded": rep.n_tasks_excluded,
           "macro_auc": rep.macro_auc, "macro_aupr": rep.macro_aupr,
           "balanced_accuracy": rep.balanced_accuracy, "f1_macro": rep.f1_macro,
           "ci": cis, "rank_mean_macro_auc": rank_auc, "heldout_split": split_gnn}
    if "random_split_seed" in m_rf:
        out["random_split_seed"] = m_rf["random_split_seed"]

    refs = [("gnn", gnn_tag, p_gnn), ("rf", rf_tag, p_rf)]
    base_path = f"{OUT_DIR}/ginv2_heldout_ensemble_{BASELINE_TAG}_probs.npz"
    if split_gnn == "scaffold" and os.path.exists(base_path):
        base = np.load(base_path)
        assert np.array_equal(base["targets"], y), "targets / molecule order differ from the baseline"
        refs.append(("baseline", BASELINE_TAG, base["probs"]))
    out["vs"] = {}
    for key, ref_tag, p_ref in refs:
        out["vs"][key] = {"reference_tag": ref_tag, "reference_macro_auc": macro_auc(y, p_ref)}
        for name, fn in [("macro_auc", macro_auc), ("macro_aupr", macro_aupr)]:
            paired = paired_bootstrap_pvalue(y, probs, p_ref, fn, n_boot=N_BOOT)
            out["vs"][key][name] = paired
            print(f"  diff vs {ref_tag} ({out['vs'][key]['reference_macro_auc']:.3f}) [{name}]: "
                  f"{paired['observed_diff']:+.3f}  paired bootstrap p={paired['p_value']:.3f}")

    np.savez_compressed(f"{OUT_DIR}/ginv2_heldout_ensemble_{tag}_probs.npz", probs=probs, targets=y)
    json.dump(out, open(f"{OUT_DIR}/ginv2_heldout_ensemble_{tag}.json", "w"), indent=2)
    print(f"Saved: {OUT_DIR}/ginv2_heldout_ensemble_{tag}.json")


if __name__ == "__main__":
    main()
