"""GIN-v2 2.3 (RDKit 2D descriptors, no SWA) on exactly the literature split:
the OGB ogbg-molsider scaffold split (= the deterministic scaffold split of
Hu et al. 2020; training 1141 / stopping = OGB valid 143 / test = OGB test 143).
Purpose: direct comparison with the scaffold results of Hu, GraphCL, GraphMVP,
GraphMAE, SimSGT and others. NOT comparable with the scaffold-204 ablation
table.

The descriptor normalizer is fitted on this split's training molecules.

Usage: PYTHONPATH=. python -m scripts.train.ogb_a3 [indices]
"""
import os
import sys

from src.data.featurize import RDKitDescriptorNormalizer, configure_rdkit_descriptors
from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR, _split_sider

CKPT_PREFIX = "ginv2_2_3_ogb_"
GROUP_LABEL = "2.3_ogb"
MANIFEST_NAME = "manifest_ginv2_2_3_ogb.json"
NORMALIZER_PATH = "data/raw/ginv2_rdkit_descriptor_normalizer_ogb.json"


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    config = load_config(chembl_feature_version="v2")
    config["dataset"]["heldout_split"] = "ogb_scaffold"
    if not os.path.exists(NORMALIZER_PATH):
        _, sider_train_dc, _ = _split_sider(config, carve_heldout_test=True)
        RDKitDescriptorNormalizer().fit(list(sider_train_dc.ids)).to_json(NORMALIZER_PATH)
        print(f"[2.3-ogb] normalizer fitted on {len(sider_train_dc.ids)} training molecules: {NORMALIZER_PATH}")
    config["ablation"]["use_rdkit_descriptors"] = True
    config["ablation"]["use_swa"] = False
    config["dataset"]["rdkit_descriptor_normalizer_path"] = NORMALIZER_PATH
    configure_rdkit_descriptors(True, NORMALIZER_PATH)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("[2.3-ogb] OGB/Hu scaffold split, use_rdkit_descriptors=True, use_swa=False")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, os.path.join(CKPT_DIR, MANIFEST_NAME),
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
