"""GIN-v2 combined model (2.3 RDKit 2D descriptors + 2.6 SWA) on the literature
split: the OGB ogbg-molsider scaffold split (= Hu et al. 2020; training 1141 /
stopping = OGB valid 143 / test = OGB test 143).

OGB/Hu counterpart of the scaffold-204 combined model
(scripts/train/scaffold204_a3a5.py). Everything else as the baseline.

The descriptor normalizer must be the same file used by 2.3-ogb (fitted on
this split's training molecules), so the graph cache is reused without
re-featurization. If it is missing, run scripts/train/ogb_a3.py first.

NOT comparable with the scaffold-204 ablation table.

Usage: PYTHONPATH=. python -m scripts.train.ogb_a3a5 [indices]
  (no indices: 0 1 2 3 4; seed = 1000 + index)
"""
import os
import sys

from src.data.featurize import configure_rdkit_descriptors
from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR

CKPT_PREFIX = "ginv2_karma_ogb_"
GROUP_LABEL = "karma_ogb"
MANIFEST_NAME = "manifest_ginv2_karma_ogb.json"
NORMALIZER_PATH = "data/raw/ginv2_rdkit_descriptor_normalizer_ogb.json"  # same file as 2.3-ogb


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    config = load_config(chembl_feature_version="v2")
    config["dataset"]["heldout_split"] = "ogb_scaffold"
    assert os.path.exists(NORMALIZER_PATH), (
        f"{NORMALIZER_PATH} missing -- run scripts/train/ogb_a3.py first "
        "(the normalizer is fitted on this split's training molecules)."
    )
    config["ablation"]["use_rdkit_descriptors"] = True
    config["ablation"]["use_swa"] = True
    config["dataset"]["rdkit_descriptor_normalizer_path"] = NORMALIZER_PATH
    configure_rdkit_descriptors(True, NORMALIZER_PATH)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("[karma-ogb] OGB/Hu scaffold split, use_rdkit_descriptors=True + use_swa=True")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, os.path.join(CKPT_DIR, MANIFEST_NAME),
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
