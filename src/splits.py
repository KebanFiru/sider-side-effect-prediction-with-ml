"""A genuinely untouched scaffold test split for the GIN-v2 ablation.

`src/train_ensemble_hetero.py::_split_sider` (and `src/train.py::prepare_data`)
dissolve DeepChem's scaffold train/valid/test split and make their own 90/10
training/"stopping slice" split. Because that stopping slice is used as the
early-stopping and model-selection criterion, it is not an independent test:
every component measured without a clean test set looks good on its own
selection criterion, which makes the components indistinguishable.

This module permanently carves out a scaffold-based ~14.3% (1/7) slice BEFORE
assign_folds (the 90/10 stopping split) is called, and writes it to disk. Once
written it is always read back from that file: re-running never produces a
different split and silently swaps it (consistency is cross-checked against
`all_ids`).

Scope: used only by the GIN-v2 ablation, and only when `carve_heldout_test=True`
is passed explicitly. The existing group2a/group3/PairGNN/EGNN training and
evaluation paths never call this module, so their behaviour is unchanged and
the historical weights stand as-is.
"""
import json
import os

import numpy as np

HELDOUT_TEST_PATH = "data/raw/gin_v2_heldout_test_ids.json"
HELDOUT_NUM_FOLDS = 7  # 1/7 = 14.3%, the closest integer divisor to the planned 15%
# Random split (for comparison with the literature, 80/10/10 like MoleculeNet):
# 10% test, and 1/9 of the remainder as the stopping slice. The scaffold file is
# never touched.
RANDOM_TEST_PATH = "data/raw/gin_v2_random_test_ids_seed{seed}.json"
RANDOM_TEST_FRAC = 0.1
# The literature's split, exactly: the OGB ogbg-molsider "scaffold" split
# (= the deterministic scaffold split of Hu et al. 2020, 80/10/10:
# 1141/143/143). OGB's files live under data/raw/ogb_sider_scaffold/; molecules
# are matched by canonical SMILES (1427/1427 match, with identical labels).
# test = OGB test, stopping slice = OGB valid, training = OGB train.
OGB_DIR = "data/raw/ogb_sider_scaffold"
# Paradigm experiment: test = our scaffold-204 test UNION the OGB/Hu test (~328
# molecules). From the remaining molecules the stopping slice is chosen by one
# of two paradigms (dataset.stopping_paradigm):
#   "balanced"   : ours -- scaffold groups are spread evenly across folds
#                  (assign_folds), 1/10 of them becoming the stopping slice
#   "sequential" : the literature's (Hu/OGB) -- scaffold groups are filled
#                  largest-first, and once training is full the remaining
#                  (small, rare) groups go to the stopping slice; same 1/10 share
# Same test, same training pool: the only difference is how training and
# stopping molecules are selected.


def random_split_seed(config) -> int:
    return int(config["dataset"].get("random_split_seed", 0))


def heldout_mode(config) -> str:
    mode = config["dataset"].get("heldout_split", "scaffold")
    if mode not in ("scaffold", "random", "ogb_scaffold", "paradigm"):
        raise ValueError(f"dataset.heldout_split: {mode!r} (must be scaffold|random|ogb_scaffold|paradigm)")
    return mode


def get_or_create_random_heldout_mask(all_ids: list, seed: int) -> np.ndarray:
    """A random 10% test split: generated and saved on the first call, read from
    the file afterwards (the same principle as the scaffold split)."""
    path = RANDOM_TEST_PATH.format(seed=seed)
    if os.path.exists(path):
        heldout_ids = set(json.load(open(path)))
        missing = heldout_ids - set(all_ids)
        if missing:
            raise RuntimeError(f"{len(missing)} molecules from '{path}' are not in the current dataset.")
        return np.array([smi in heldout_ids for smi in all_ids])
    unique = sorted(set(all_ids))
    rng = np.random.RandomState(seed)
    chosen = set(rng.choice(len(unique), size=int(round(RANDOM_TEST_FRAC * len(unique))), replace=False).tolist())
    heldout_ids = sorted(unique[i] for i in chosen)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(heldout_ids, open(path, "w"), indent=2)
    print(f"[splits] NEW random test split generated and saved: {path} ({len(heldout_ids)} molecules)")
    hs = set(heldout_ids)
    return np.array([smi in hs for smi in all_ids])


def ogb_split_masks(all_ids: list):
    """Map the OGB scaffold split onto the order of `all_ids`:
    (train_mask, valid_mask, test_mask)."""
    import pandas as pd
    from rdkit import Chem

    def canon(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"could not parse SMILES: {smi}")
        return Chem.MolToSmiles(mol)

    ogb_smiles = pd.read_csv(os.path.join(OGB_DIR, "mol.csv.gz"))["smiles"].tolist()
    part = {}
    for name in ("train", "valid", "test"):
        idx = pd.read_csv(os.path.join(OGB_DIR, f"{name}.csv.gz"), header=None)[0].tolist()
        for i in idx:
            part[canon(ogb_smiles[i])] = name
    ours = [part.get(canon(smi)) for smi in all_ids]
    if any(p is None for p in ours):
        raise RuntimeError(f"{sum(p is None for p in ours)} molecules were not found in the OGB split")
    ours = np.array(ours)
    return ours == "train", ours == "valid", ours == "test"


def sequential_scaffold_stopping_mask(ids: list, frac: float) -> np.ndarray:
    """The logic of Hu et al. (pretrain-gnns splitters.scaffold_split) without a
    test set: scaffold groups (by decreasing size, ties broken by first index)
    are added to training in order, and the group that would push training past
    the (1-frac) limit goes to the stopping slice instead."""
    from collections import defaultdict
    from src.train import _scaffold

    groups = defaultdict(list)
    for i, smi in enumerate(ids):
        groups[_scaffold(smi) or f"__invalid_{i}"].append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), -g[0]), reverse=True)
    cutoff = (1.0 - frac) * len(ids)
    mask = np.zeros(len(ids), dtype=bool)
    n_train = 0
    for g in ordered:
        if n_train + len(g) > cutoff:
            mask[g] = True
        else:
            n_train += len(g)
    return mask


def get_or_create_heldout_mask(all_ids: list, path: str = HELDOUT_TEST_PATH,
                                num_folds: int = HELDOUT_NUM_FOLDS) -> np.ndarray:
    """Verify and use a previously saved split over `all_ids` if one exists;
    otherwise generate a scaffold-balanced split and save it permanently. If the
    molecule set has changed (a SMILES in the file is no longer in `all_ids`)
    this raises instead of silently using a different split.
    """
    from src.train import assign_folds  # imported here to avoid a circular import

    if os.path.exists(path):
        heldout_ids = set(json.load(open(path)))
        missing = heldout_ids - set(all_ids)
        if missing:
            raise RuntimeError(
                f"{len(missing)} molecules from '{path}' are not in the current dataset "
                f"(for example: {list(missing)[:3]}). Data loading may have changed -- "
                f"inspect it by hand. This file is never regenerated silently, because "
                f"the ablation has to use the SAME test split throughout.")
        return np.array([smi in heldout_ids for smi in all_ids])

    fold = assign_folds(all_ids, num_folds)
    mask = fold == 0
    heldout_ids = sorted({all_ids[i] for i in range(len(all_ids)) if mask[i]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(heldout_ids, open(path, "w"), indent=2)
    print(f"[splits] NEW held-out test split generated and saved: {path} "
          f"({len(heldout_ids)}/{len(all_ids)} molecules, {100 * len(heldout_ids) / len(all_ids):.1f}%)")
    return mask


def carve_out_heldout(all_ids: list, all_y: np.ndarray, all_w: np.ndarray, path: str = HELDOUT_TEST_PATH,
                      mode: str = "scaffold", seed: int = 0):
    """Return (remaining_ids, remaining_y, remaining_w, heldout_ids, heldout_y,
    heldout_w). The `remaining_*` arrays are what must then be fed into the
    existing 90/10 training/stopping-slice split (assign_folds), so that the
    held-out set is entirely separate from that split and removed first.
    """
    if mode == "random":
        mask = get_or_create_random_heldout_mask(all_ids, seed)
    elif mode == "ogb_scaffold":
        mask = ogb_split_masks(all_ids)[2]
    elif mode == "paradigm":
        mask = get_or_create_heldout_mask(all_ids, path) | ogb_split_masks(all_ids)[2]
    else:
        mask = get_or_create_heldout_mask(all_ids, path)
    remaining_ids = [all_ids[i] for i in range(len(all_ids)) if not mask[i]]
    heldout_ids = [all_ids[i] for i in range(len(all_ids)) if mask[i]]
    return remaining_ids, all_y[~mask], all_w[~mask], heldout_ids, all_y[mask], all_w[mask]


def load_heldout_test_set(config):
    """For the final comparison in the ablation table: return the held-out
    molecules and their labels. This set was never used in training or early
    stopping -- it was excluded the same way in every training run that calls
    `_split_sider(config, carve_heldout_test=True)`.
    """
    from src.data.datasets import load_sider
    from src.train import CombinedDataset

    _, train_dc, valid_dc, test_dc = load_sider(root=config["dataset"]["root"])
    all_ids = list(train_dc.ids) + list(valid_dc.ids) + list(test_dc.ids)
    all_y = np.concatenate([train_dc.y, valid_dc.y, test_dc.y], axis=0)
    all_w = np.concatenate([train_dc.w, valid_dc.w, test_dc.w], axis=0)
    _, _, _, heldout_ids, heldout_y, heldout_w = carve_out_heldout(
        all_ids, all_y, all_w, mode=heldout_mode(config), seed=random_split_seed(config))
    return CombinedDataset(heldout_ids, heldout_y, heldout_w)
