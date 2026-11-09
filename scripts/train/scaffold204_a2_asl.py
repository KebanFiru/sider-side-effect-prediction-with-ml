"""GIN-v2 ablation 2.2: Asymmetric Loss instead of focal loss (ablation.use_asl).
Everything else as the baseline (GINEConv, 3 layers, embedding 128, ChEMBL V2,
sider_batch_fraction=0.5). ASL parameters from configs/default.yaml
(asl_gamma_neg=4, asl_gamma_pos=1, asl_clip=0.05).

With ASL on, the end-of-training step that folds per-task thresholds into the
bias is skipped automatically (see src/train.py); unlike the baseline,
balanced accuracy and F1 are measured at the uncorrected 0.5 threshold.

Same held-out test set (src/splits.py) and same five seeds (1000-1004) as the
baseline.

Usage: PYTHONPATH=. python -m scripts.train.scaffold204_a2_asl [indices]
"""
import os
import sys

from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR

CKPT_PREFIX = "ginv2_2_2_asl_"
GROUP_LABEL = "2.2_asl"
MANIFEST_NAME = "manifest_ginv2_2_2_asl.json"


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    config = load_config(chembl_feature_version="v2")
    config["ablation"]["use_asl"] = True
    os.makedirs(CKPT_DIR, exist_ok=True)
    manifest_path = os.path.join(CKPT_DIR, MANIFEST_NAME)

    print("[ginv2-2.2] use_asl=True (baseline=focal loss), everything else unchanged, same held-out split.")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, manifest_path,
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
