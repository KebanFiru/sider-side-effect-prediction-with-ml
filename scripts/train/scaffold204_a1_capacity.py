"""GIN-v2 ablation 2.1: gin_layers=5 (baseline 3) AND hidden_dim=300
(baseline 128), the default sizes of pretrain-GNN. Everything else as the
baseline (ChEMBL V2, sider_batch_fraction=0.5, focal loss).

Same held-out test set (src/splits.py) and same five seeds (1000-1004) as the
baseline.

Both changes are applied TOGETHER, so this run cannot tell whether a
difference comes from depth or from width.

Usage: PYTHONPATH=. python -m scripts.train.scaffold204_a1_capacity [indices]
"""
import os
import sys

from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR

CKPT_PREFIX = "ginv2_2_1_layers_"
GROUP_LABEL = "2.1_gin_layers"
MANIFEST_NAME = "manifest_ginv2_2_1_layers.json"
GIN_LAYERS = 5
HIDDEN_DIM = 300


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    config = load_config(chembl_feature_version="v2")
    config["ablation"]["gin_layers"] = GIN_LAYERS
    config["model"]["hidden_dim"] = HIDDEN_DIM
    os.makedirs(CKPT_DIR, exist_ok=True)
    manifest_path = os.path.join(CKPT_DIR, MANIFEST_NAME)

    print(f"[ginv2-2.1] gin_layers={GIN_LAYERS} (baseline=3), hidden_dim={HIDDEN_DIM} "
          f"(baseline=128), everything else unchanged, same held-out split.")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, manifest_path,
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
