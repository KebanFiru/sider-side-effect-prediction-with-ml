"""GIN-v2 ablation arm: evaluates the 5-seed ENSEMBLE (mean of the checkpoints'
sigmoid outputs) on the untouched held-out test set. Used for every row of
the paper; each component (2.1-2.6) runs with the same template by
changing the checkpoint prefix.

Usage: PYTHONPATH=. python -m scripts.eval.ensemble \
    outputs/ensemble_hetero/ginv2_baseline_ baseline
  For components that change the architecture, pass the training script's
  config overrides UNCHANGED (otherwise load_state_dict fails on shapes), e.g. 2.1:
    ... outputs/ensemble_hetero/ginv2_2_1_layers_ 2_1 \
        ablation.gin_layers=5 model.hidden_dim=300
  2.3 (RDKit descriptors; the normalizer must be the same file as in training):
    ... outputs/ensemble_hetero/ginv2_2_3_rdkit_ 2_3 ablation.use_rdkit_descriptors=true \
        'dataset.rdkit_descriptor_normalizer_path="data/raw/ginv2_rdkit_descriptor_normalizer.json"'

Ensemble probabilities and targets are also saved as
outputs/eval/..._{tag}_probs.npz. If tag != the reference and the reference
.npz exists, a paired bootstrap p-value on the SAME molecules (
item 4) is computed and written to the JSON as "vs_baseline".
"""
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.featurize import MoleculeGraphDataset, configure_rdkit_descriptors
from src.eval_metrics import (
    bootstrap_ci,
    macro_auc,
    macro_aupr,
    macro_balanced_accuracy,
    macro_f1,
    macro_report,
    paired_bootstrap_pvalue,
)
from src.splits import load_heldout_test_set
from src.train import build_model, load_config

N_BOOT = 2000
OUT_DIR = os.environ.get("EVAL_OUT_DIR", "outputs/eval")  # EVAL_OUT_DIR: write elsewhere (reproducibility test)
N_SEEDS = int(os.environ.get("N_SEEDS", 5))  # number of seeds in the ensemble
# Literature protocol (each seed scored separately): SEED_INDICES="3" loads only
# that checkpoint (index 3 = seed 1003); default 0..N_SEEDS-1.
SEED_INDICES = [int(x) for x in os.environ["SEED_INDICES"].split(",")] if os.environ.get("SEED_INDICES") else None
# Reference for the paired comparison. On other splits (e.g. dataset.heldout_split=
# "random") the scaffold baseline covers DIFFERENT molecules, so a reference on the
# same split must be given via REF_TAG (e.g. REF_TAG=classic_rf_ecfp_rdkit_random_seed0).
BASELINE_TAG = os.environ.get("REF_TAG", "baseline")


def _apply_overrides(config, overrides):
    """`section.key=value` (e.g. ablation.gin_layers=5) -> config[section][key]."""
    for item in overrides:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        config[section][name] = json.loads(value)


def main():
    ckpt_prefix = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "ensemble"
    overrides = sys.argv[3:]

    config = load_config(chembl_feature_version="v2")
    _apply_overrides(config, overrides)
    # load_config configured descriptors BEFORE the overrides; redo it here.
    use_rdkit = config["ablation"].get("use_rdkit_descriptors", False)
    configure_rdkit_descriptors(use_rdkit, config["dataset"].get("rdkit_descriptor_normalizer_path") if use_rdkit else None)
    heldout_dc = load_heldout_test_set(config)
    ds = MoleculeGraphDataset(heldout_dc, cache_dir=config["dataset"]["processed_multitask_root"])
    loader = DataLoader(ds, batch_size=config["train"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_sider_tasks = 27

    all_probs_per_model = []
    targets = None
    indices = SEED_INDICES if SEED_INDICES is not None else list(range(N_SEEDS))
    for i in indices:
        ckpt_path = f"{ckpt_prefix}{i}.pt"
        state_dict = torch.load(ckpt_path, map_location=device)
        model = build_model(json.loads(json.dumps(config)), 60, device)
        model.load_state_dict(state_dict)
        model.eval()

        preds, tgts = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.global_features))
                out = out[:, :num_sider_tasks]
                preds.append(out.cpu())
                tgts.append(batch.y.cpu())
        preds = torch.cat(preds).numpy()
        all_probs_per_model.append(preds)
        if targets is None:
            targets = torch.cat(tgts).numpy()
        print(f"  [{i}] {ckpt_path} loaded.")

    ensemble_probs = np.mean(all_probs_per_model, axis=0)

    rep = macro_report(targets, ensemble_probs, threshold=0.5)
    print(f"\n=== GIN-v2 [{tag}] -- {len(indices)}-seed ENSEMBLE, held-out test "
          f"(n={targets.shape[0]} molecules) ===")
    print(f"Total tasks: {rep.n_tasks_total}, EXCLUDED (single-class) tasks: {rep.n_tasks_excluded}")
    print(f"macro_AUC={rep.macro_auc:.3f}  macro_AUPR={rep.macro_aupr:.3f}  "
          f"balanced_acc={rep.balanced_accuracy:.3f}  F1_macro={rep.f1_macro:.3f}")

    cis = {}
    for metric_name, fn in [
        ("macro_auc", macro_auc), ("macro_aupr", macro_aupr),
        ("balanced_accuracy", macro_balanced_accuracy), ("f1_macro", macro_f1),
    ]:
        ci = bootstrap_ci(targets, ensemble_probs, fn, n_boot=N_BOOT)
        cis[metric_name] = ci
        lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
        print(f"  {metric_name}: {ci['point']:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    out = {
        "tag": tag, "ckpt_prefix": ckpt_prefix, "n_seeds": len(indices), "seed_indices": indices,
        "n_molecules": int(targets.shape[0]),
        "n_tasks_total": rep.n_tasks_total, "n_tasks_excluded": rep.n_tasks_excluded,
        "macro_auc": rep.macro_auc, "macro_aupr": rep.macro_aupr,
        "balanced_accuracy": rep.balanced_accuracy, "f1_macro": rep.f1_macro,
        "ci": cis,
    }
    if overrides:
        out["config_overrides"] = overrides
    out["heldout_split"] = config["dataset"].get("heldout_split", "scaffold")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(f"{OUT_DIR}/ginv2_heldout_ensemble_{tag}_probs.npz",
                        probs=ensemble_probs, targets=targets)

    baseline_probs_path = f"{OUT_DIR}/ginv2_heldout_ensemble_{BASELINE_TAG}_probs.npz"
    if tag != BASELINE_TAG and os.path.exists(baseline_probs_path):
        base = np.load(baseline_probs_path)
        # Pairing is only valid for the SAME molecules in the SAME order.
        assert np.array_equal(base["targets"], targets), "targets / molecule order differ from the reference"
        out["vs_baseline"] = {"reference_tag": BASELINE_TAG}
        for metric_name, fn in [("macro_auc", macro_auc), ("macro_aupr", macro_aupr)]:
            paired = paired_bootstrap_pvalue(targets, ensemble_probs, base["probs"], fn, n_boot=N_BOOT)
            out["vs_baseline"][metric_name] = paired
            print(f"  diff vs reference [{metric_name}]: {paired['observed_diff']:+.3f}  "
                  f"paired bootstrap p={paired['p_value']:.3f}")

    out_path = f"{OUT_DIR}/ginv2_heldout_ensemble_{tag}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
