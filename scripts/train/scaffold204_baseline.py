"""GIN-v2 ablation baseline on the scaffold-204 split.

GINEConv, 3 layers, per-task GlobalAttention readout, focal loss,
sider_batch_fraction=0.5, ChEMBL target features V2. The 204-molecule scaffold
test set (~14.3%, data/raw/gin_v2_heldout_test_ids.json, built by
src/splits.py) is carved out of training and the stopping slice
(carve_heldout_test=True). These models fill the "Baseline" row of the
ablation table in the paper.

Five seeds (1000-1004). The manifest is updated after every model, so an
interrupted run resumes where it stopped (see src/train.py::train_model).

Usage: PYTHONPATH=. python -m scripts.train.scaffold204_baseline [indices]
  (no indices: 0 1 2 3 4; otherwise only the given ones, e.g. "0 1" trains the
   first two seeds, which lets seeds be spread over parallel processes)
"""
import os
import sys

from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR

CKPT_PREFIX = "ginv2_baseline_"
GROUP_LABEL = "2a_heldout"
MANIFEST_NAME = "manifest_ginv2_baseline.json"


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    # ChEMBL target features V2, set explicitly.
    config = load_config(chembl_feature_version="v2")
    os.makedirs(CKPT_DIR, exist_ok=True)
    manifest_path = os.path.join(CKPT_DIR, MANIFEST_NAME)

    print(f"[ginv2-baseline] held-out test set: data/raw/gin_v2_heldout_test_ids.json "
          f"(created now if missing; see src/splits.py)")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, manifest_path,
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
