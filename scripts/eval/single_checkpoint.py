"""GIN-v2 ablation arm: evaluates a SINGLE checkpoint on the held-out scaffold
test set that was NEVER seen in training or early stopping (src/splits.py,
data/raw/gin_v2_heldout_test_ids.json, ~204 molecules).

Not to be mixed with the old "70-molecule" or "stopping slice" numbers of
group2a/3/PairGNN/EGNN -- this set never entered those pipelines (see the
src/splits.py docstring).

Usage: PYTHONPATH=. python -m scripts.eval.single_checkpoint <ckpt_path> [seed_tag]
  e.g. python -m scripts.eval.single_checkpoint outputs/ensemble_hetero/ginv2_baseline_0.pt seed0
"""
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.featurize import MoleculeGraphDataset
from src.eval_metrics import bootstrap_ci, macro_auc, macro_aupr, macro_balanced_accuracy, macro_f1, macro_report
from src.splits import load_heldout_test_set
from src.train import build_model, load_config

N_BOOT = 2000
OUT_DIR = "outputs/eval"


def main():
    ckpt_path = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else ckpt_path

    config = load_config(chembl_feature_version="v2")
    heldout_dc = load_heldout_test_set(config)
    ds = MoleculeGraphDataset(heldout_dc, cache_dir=config["dataset"]["processed_multitask_root"])
    loader = DataLoader(ds, batch_size=config["train"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_sider_tasks = 27
    # The checkpoint was trained with total_tasks = 60 (including auxiliary tasks).
    state_dict = torch.load(ckpt_path, map_location=device)
    model = build_model(config, 60, device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.global_features))
            out = out[:, :num_sider_tasks]
            all_preds.append(out.cpu())
            all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    rep = macro_report(targets, preds, threshold=0.5)
    print(f"\n=== GIN-v2 baseline [{tag}] -- truly unseen held-out test "
          f"(n={targets.shape[0]} molecules) ===")
    print(f"Total tasks: {rep.n_tasks_total}, EXCLUDED (single-class) tasks: {rep.n_tasks_excluded}")
    print(f"macro_AUC={rep.macro_auc:.3f}  macro_AUPR={rep.macro_aupr:.3f}  "
          f"balanced_acc={rep.balanced_accuracy:.3f}  F1_macro={rep.f1_macro:.3f}")

    cis = {}
    for metric_name, fn in [
        ("macro_auc", macro_auc), ("macro_aupr", macro_aupr),
        ("balanced_accuracy", macro_balanced_accuracy), ("f1_macro", macro_f1),
    ]:
        ci = bootstrap_ci(targets, preds, fn, n_boot=N_BOOT)
        cis[metric_name] = ci
        lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
        print(f"  {metric_name}: {ci['point']:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    out = {
        "tag": tag, "ckpt_path": ckpt_path, "n_molecules": int(targets.shape[0]),
        "n_tasks_total": rep.n_tasks_total, "n_tasks_excluded": rep.n_tasks_excluded,
        "macro_auc": rep.macro_auc, "macro_aupr": rep.macro_aupr,
        "balanced_accuracy": rep.balanced_accuracy, "f1_macro": rep.f1_macro,
        "ci": cis,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/ginv2_heldout_{tag}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
