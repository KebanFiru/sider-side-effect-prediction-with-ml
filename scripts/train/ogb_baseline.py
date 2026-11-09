"""GIN-v2 ablation baseline on the literature split: the OGB ogbg-molsider
scaffold split (= the deterministic scaffold split of Hu et al. 2020; training
1141 / stopping = OGB valid 143 / test = OGB test 143).

Purpose: test whether the 2.3 gain (RDKit descriptors) holds on a second
split besides scaffold-204. Settings are identical to
scripts/train/scaffold204_baseline.py (GINEConv, 3 layers, embedding 128, focal loss,
ChEMBL V2, 33 auxiliary tasks, sider_batch_fraction=0.5); only the split differs.

NOT comparable with the scaffold-204 ablation table.

Usage: PYTHONPATH=. python -m scripts.train.ogb_baseline [indices]
  (no indices: 0 1 2 3 4; seed = 1000 + index)
"""
import os
import sys

from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR

CKPT_PREFIX = "ginv2_baseline_ogb_"
GROUP_LABEL = "2a_ogb"
MANIFEST_NAME = "manifest_ginv2_baseline_ogb.json"


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    config = load_config(chembl_feature_version="v2")
    config["dataset"]["heldout_split"] = "ogb_scaffold"
    # Baseline: no ablation component enabled (same as configs/default.yaml),
    # so no descriptor normalizer is needed.
    config["ablation"]["use_rdkit_descriptors"] = False
    config["ablation"]["use_swa"] = False
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("[baseline-ogb] OGB/Hu scaffold split, no ablation component")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, os.path.join(CKPT_DIR, MANIFEST_NAME),
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
