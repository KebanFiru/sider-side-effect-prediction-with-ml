"""GIN-v2 combined model (2.3 RDKit 2D descriptors + 2.6 SWA) on the RANDOM split.

Everything else as the baseline (GINEConv, 3 layers, embedding 128, focal loss,
ChEMBL V2, 33 auxiliary tasks, sider_batch_fraction=0.5).

Split: dataset.heldout_split=random -- 10% random test
(data/raw/gin_v2_random_test_ids_seed0.json, 143 molecules), 1/9 of the rest as
a random stopping slice (143), training 1141; 80/10/10 as in MoleculeNet. This
is for comparison with random-split results in the literature (RF 0.684,
Chemprop 0.648/0.676); it is NOT comparable with the scaffold ablation table.

The descriptor normalizer is fitted separately on this split's training
molecules (the 2.3 normalizer was fitted on the scaffold training set, which
may contain random-split test molecules).

Other random splits (to match the three RF splits): RANDOM_SPLIT_SEED=1|2.
For splits other than 0, "s{N}_" / "_s{N}" is appended to checkpoint and
manifest names, and the normalizer is fitted on that split's training molecules.

Usage: [RANDOM_SPLIT_SEED=N] PYTHONPATH=. python -m scripts.train.random_a3a5 [indices]
"""
import os
import sys

from src.data.featurize import RDKitDescriptorNormalizer, configure_rdkit_descriptors
from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR, _split_sider

SPLIT_SEED = int(os.environ.get("RANDOM_SPLIT_SEED", 0))
_SFX = "" if SPLIT_SEED == 0 else f"s{SPLIT_SEED}_"  # split 0 keeps its original names
CKPT_PREFIX = f"ginv2_karma_random_{_SFX}"
GROUP_LABEL = "karma_random" + ("" if SPLIT_SEED == 0 else f"_s{SPLIT_SEED}")
MANIFEST_NAME = "manifest_ginv2_karma_random" + ("" if SPLIT_SEED == 0 else f"_s{SPLIT_SEED}") + ".json"
NORMALIZER_PATH = f"data/raw/ginv2_rdkit_descriptor_normalizer_random_seed{SPLIT_SEED}.json"


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    config = load_config(chembl_feature_version="v2")
    config["dataset"]["heldout_split"] = "random"
    config["dataset"]["random_split_seed"] = SPLIT_SEED
    if not os.path.exists(NORMALIZER_PATH):
        _, sider_train_dc, _ = _split_sider(config, carve_heldout_test=True)
        RDKitDescriptorNormalizer().fit(list(sider_train_dc.ids)).to_json(NORMALIZER_PATH)
        print(f"[karma-random] normalizer fitted on {len(sider_train_dc.ids)} training molecules: {NORMALIZER_PATH}")
    config["ablation"]["use_rdkit_descriptors"] = True
    config["ablation"]["use_swa"] = True
    config["dataset"]["rdkit_descriptor_normalizer_path"] = NORMALIZER_PATH
    configure_rdkit_descriptors(True, NORMALIZER_PATH)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"[karma-random] random split {SPLIT_SEED}, use_rdkit_descriptors=True + use_swa=True")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, os.path.join(CKPT_DIR, MANIFEST_NAME),
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
