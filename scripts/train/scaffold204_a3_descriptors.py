"""GIN-v2 ablation 2.3: RDKit 2D descriptors are added to the global features
(ablation.use_rdkit_descriptors). Everything else as the baseline (GINEConv,
3 layers, embedding 128, focal loss, ChEMBL V2, sider_batch_fraction=0.5).

Descriptor set: RDKit's own Descriptors._descList (217 in this environment),
not Chemprop's 200-descriptor descriptastorus set; see src/data/featurize.py.

Normalization (mean/std) is fitted ONLY on SIDER training molecules; the
held-out test set (src/splits.py) and the stopping slice are excluded. The file
is written once and then always read back, so training and testing use the
same statistics.

Same held-out test set and same five seeds (1000-1004) as the baseline.

Usage: PYTHONPATH=. python -m scripts.train.scaffold204_a3_descriptors [indices]
"""
import os
import sys

from src.data.featurize import RDKitDescriptorNormalizer, configure_rdkit_descriptors
from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR, _split_sider

CKPT_PREFIX = "ginv2_2_3_rdkit_"
GROUP_LABEL = "2.3_rdkit"
MANIFEST_NAME = "manifest_ginv2_2_3_rdkit.json"
NORMALIZER_PATH = "data/raw/ginv2_rdkit_descriptor_normalizer.json"


def ensure_normalizer(config):
    if os.path.exists(NORMALIZER_PATH):
        print(f"[ginv2-2.3] normalizer exists, using: {NORMALIZER_PATH}")
        return
    _, sider_train_dc, _ = _split_sider(config, carve_heldout_test=True)
    RDKitDescriptorNormalizer().fit(list(sider_train_dc.ids)).to_json(NORMALIZER_PATH)
    print(f"[ginv2-2.3] normalizer fitted on {len(sider_train_dc.ids)} SIDER training molecules: "
          f"{NORMALIZER_PATH}")


def main():
    indices = [int(x) for x in sys.argv[1:]] or list(range(5))
    config = load_config(chembl_feature_version="v2")
    ensure_normalizer(config)
    config["ablation"]["use_rdkit_descriptors"] = True
    config["dataset"]["rdkit_descriptor_normalizer_path"] = NORMALIZER_PATH
    configure_rdkit_descriptors(True, NORMALIZER_PATH)
    os.makedirs(CKPT_DIR, exist_ok=True)
    manifest_path = os.path.join(CKPT_DIR, MANIFEST_NAME)

    print("[ginv2-2.3] use_rdkit_descriptors=True (baseline=False), everything else unchanged, same held-out split.")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        CKPT_PREFIX, GROUP_LABEL, manifest_path,
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
