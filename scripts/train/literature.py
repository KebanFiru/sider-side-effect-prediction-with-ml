"""Training under the literature protocol: baseline, 2.3 and combined models on
the literature splits.

  ogb      : OGB ogbg-molsider scaffold split (= Hu et al. 2020), 10 seeds
             (indices 0-9 -> seeds 1000-1009); the literature evaluates each
             seed separately and reports mean +/- std.
  random   : 80/10/10 random split (MoleculeNet protocol), split 0/1/2 via
             RANDOM_SPLIT_SEED.
  parbal / parseq : paradigm experiment (src/splits.py): test = scaffold-204 +
             OGB/Hu test (328); stopping slice from our balanced (parbal) or the
             literature's sequential (parseq) scaffold procedure.

Names match the existing scripts: 2.3/ogb shares the manifest of
train/ogb_a3.py (indices 5-9 are appended); karma/random and 2.3/random
use the same names as train/random_a3a5.py / train/random_a3.py.

Usage: [RANDOM_SPLIT_SEED=N] PYTHONPATH=. python -m scripts.train.literature <baseline|2_3|karma> <ogb|random|parbal|parseq> [indices]
"""
import os
import sys

from src.data.featurize import RDKitDescriptorNormalizer, configure_rdkit_descriptors
from src.train import load_config
from src.train_ensemble_2groups import run_group
from src.train_ensemble_hetero import CKPT_DIR, _split_sider

# model -> (rdkit descriptors, swa, group label stem)
MODELS = {"baseline": (False, False, "baseline"), "2_3": (True, False, "2.3"), "karma": (True, True, "karma")}


def names(model, split, split_seed):
    if split in ("parbal", "parseq"):
        return f"ginv2_{model}_{split}_", f"{MODELS[model][2]}_{split}", f"manifest_ginv2_{model}_{split}.json", \
            f"data/raw/ginv2_rdkit_descriptor_normalizer_{split}.json"
    if split == "ogb":
        return f"ginv2_{model}_ogb_", f"{MODELS[model][2]}_ogb", f"manifest_ginv2_{model}_ogb.json", \
            "data/raw/ginv2_rdkit_descriptor_normalizer_ogb.json"
    sfx = "" if split_seed == 0 else f"_s{split_seed}"
    return (f"ginv2_{model}_random_" + ("" if split_seed == 0 else f"s{split_seed}_"),
            ("karma_random" if model == "karma" else f"{MODELS[model][2]}_random") + sfx,
            f"manifest_ginv2_{model}_random{sfx}.json",
            f"data/raw/ginv2_rdkit_descriptor_normalizer_random_seed{split_seed}.json")


def main():
    model, split = sys.argv[1], sys.argv[2]
    assert model in MODELS and split in ("ogb", "random", "parbal", "parseq"), sys.argv[1:3]
    indices = [int(x) for x in sys.argv[3:]] or list(range(10 if split == "ogb" else 5))
    split_seed = int(os.environ.get("RANDOM_SPLIT_SEED", 0))
    use_rdkit, use_swa, _ = MODELS[model]
    prefix, label, manifest, norm = names(model, split, split_seed)

    config = load_config(chembl_feature_version="v2")
    config["dataset"]["heldout_split"] = {"ogb": "ogb_scaffold", "random": "random"}.get(split, "paradigm")
    if split in ("parbal", "parseq"):
        config["dataset"]["stopping_paradigm"] = "balanced" if split == "parbal" else "sequential"
    config["dataset"]["random_split_seed"] = split_seed
    if use_rdkit:
        if not os.path.exists(norm):
            _, sider_train_dc, _ = _split_sider(config, carve_heldout_test=True)
            RDKitDescriptorNormalizer().fit(list(sider_train_dc.ids)).to_json(norm)
            print(f"[lit] normalizer fitted on {len(sider_train_dc.ids)} training molecules: {norm}")
        config["dataset"]["rdkit_descriptor_normalizer_path"] = norm
        configure_rdkit_descriptors(True, norm)
    config["ablation"]["use_rdkit_descriptors"] = use_rdkit
    config["ablation"]["use_swa"] = use_swa
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"[lit] {model} / {split} (split {split_seed}), indices {indices}, prefix {prefix}, label {label}")
    run_group(
        config, config["dataset"].get("aux_datasets", []),
        config["dataset"]["processed_multitask_root"], indices,
        prefix, label, os.path.join(CKPT_DIR, manifest),
        carve_heldout_test=True, num_workers=4,
    )


if __name__ == "__main__":
    main()
