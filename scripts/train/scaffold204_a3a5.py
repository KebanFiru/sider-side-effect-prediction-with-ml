"""GIN-v2 combined model on the SCAFFOLD split (the same 204-molecule clean test
as the ablation): 2.3 (RDKit 2D descriptors) + 2.6 (SWA) together. Everything
else as the baseline. Question: do the effects of the two useful components
add up? Compared with the baseline, 2.3 and 2.6 by paired bootstrap on the
same molecules.

The descriptor normalizer is the same as for 2.3 (fitted on the scaffold
training set), so the 2.3 graph cache is reused without re-featurization.

Usage: PYTHONPATH=. python -m scripts.train.scaffold204_a3a5 [indices]
"""
import os
import sys

from src.data.featurize import configure_rdkit_descriptors
from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR

CKPT_PREFIX = "ginv2_karma_scaffold_"
GROUP_LABEL = "karma_scaffold"
MANIFEST_NAME = "manifest_ginv2_karma_scaffold.json"
NORMALIZER_PATH = "data/raw/ginv2_rdkit_descriptor_normalizer.json"  # same file as 2.3


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    config = load_config(chembl_feature_version="v2")
    assert config["dataset"].get("heldout_split", "scaffold") == "scaffold"
    assert os.path.exists(NORMALIZER_PATH), NORMALIZER_PATH
    config["ablation"]["use_rdkit_descriptors"] = True
    config["ablation"]["use_swa"] = True
    config["dataset"]["rdkit_descriptor_normalizer_path"] = NORMALIZER_PATH
    configure_rdkit_descriptors(True, NORMALIZER_PATH)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("[karma-scaffold] use_rdkit_descriptors=True + use_swa=True, same held-out split as the ablation.")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, os.path.join(CKPT_DIR, MANIFEST_NAME),
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
