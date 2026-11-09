"""GIN-v2 ablation 2.6: Stochastic Weight Averaging (ablation.use_swa).
Everything else as the baseline (GINEConv, 3 layers, embedding 128, focal loss,
ChEMBL V2, sider_batch_fraction=0.5, RDKit descriptors off).

On top of the best early-stopped model, swa_epochs (5) extra epochs are run
(SWALR, swa_lr=5e-5); the weights at the end of each epoch are averaged and
BatchNorm statistics are recomputed on the training data. The SWA model is
used even if it is worse on the stopping slice, since the component's own
effect is being measured. Settings from configs/default.yaml.

Same held-out test set and same five seeds (1000-1004) as the baseline.

Usage: PYTHONPATH=. python -m scripts.train.scaffold204_a5_swa [indices]
"""
import os
import sys

from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR

CKPT_PREFIX = "ginv2_2_6_swa_"
GROUP_LABEL = "2.6_swa"
MANIFEST_NAME = "manifest_ginv2_2_6_swa.json"


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    config = load_config(chembl_feature_version="v2")
    config["ablation"]["use_swa"] = True
    os.makedirs(CKPT_DIR, exist_ok=True)
    manifest_path = os.path.join(CKPT_DIR, MANIFEST_NAME)

    print("[ginv2-2.6] use_swa=True (baseline=False), everything else unchanged, same held-out split.")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, manifest_path,
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
