"""GIN-v2 ablation 2.5: task weighting by uncertainty weighting
(Kendall et al. 2018; ablation.task_weighting=uncertainty). Everything else as
the baseline (GINEConv, 3 layers, embedding 128, focal loss, ChEMBL V2,
sider_batch_fraction=0.5, RDKit descriptors off).

Each output task (SIDER + auxiliary tasks, 60 in total) has a learnable
log_sigma; the loss is the mean of precision * task_loss + log_sigma over the
tasks labelled in the batch. One further difference from the baseline follows
from the method itself: the baseline averages over labelled cells (tasks with
many labels weigh more), whereas here the loss is first averaged per task
(every task is equal before the sigmas apply).

Same held-out test set and same five seeds (1000-1004) as the baseline.

Usage: PYTHONPATH=. python -m scripts.train.scaffold204_a4_uncertainty [indices]
"""
import os
import sys

from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR

CKPT_PREFIX = "ginv2_2_5_uncertainty_"
GROUP_LABEL = "2.5_uncertainty"
MANIFEST_NAME = "manifest_ginv2_2_5_uncertainty.json"


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    config = load_config(chembl_feature_version="v2")
    config["ablation"]["task_weighting"] = "uncertainty"
    os.makedirs(CKPT_DIR, exist_ok=True)
    manifest_path = os.path.join(CKPT_DIR, MANIFEST_NAME)

    print("[ginv2-2.5] task_weighting=uncertainty (baseline=none), everything else unchanged, same held-out split.")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, manifest_path,
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
