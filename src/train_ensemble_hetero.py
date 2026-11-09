"""Shared data helpers for the training scripts: split loading (`_split_sider`),
per-run data building (`_build_group_data`) and the checkpoint directory."""
import json
import os

import numpy as np
import torch

from src.train import (
    CombinedDataset,
    STOPPING_SLICE_FOLDS,
    assign_folds,
    build_combined_train_set,
    build_model,
    compute_pos_rate,
    find_degenerate_tasks,
    load_config,
    train_model,
)
from src.data.datasets import load_sider
from src.data.featurize import MoleculeGraphDataset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

CKPT_DIR = "outputs/ensemble_hetero"


def _split_sider(config, carve_heldout_test=False):
    """`carve_heldout_test=True` (the GIN-v2 ablation only, see src/splits.py)
    permanently carves out a scaffold-based ~14.3% slice before assign_folds
    makes its 90/10 split, and keeps that slice out of both training and the
    stopping slice. It is used only for the final comparison in the ablation
    table. The default, False, reproduces the older behaviour exactly, so
    group2a/3/PairGNN/EGNN are unaffected.
    """
    tasks, train_dc, valid_dc, test_dc = load_sider(root=config["dataset"]["root"])
    all_ids = list(train_dc.ids) + list(valid_dc.ids) + list(test_dc.ids)
    all_y = np.concatenate([train_dc.y, valid_dc.y, test_dc.y], axis=0)
    all_w = np.concatenate([train_dc.w, valid_dc.w, test_dc.w], axis=0)

    random_mode = ogb_mode = paradigm_mode = False
    if carve_heldout_test:
        from src.splits import carve_out_heldout, heldout_mode, random_split_seed
        random_mode = heldout_mode(config) == "random"
        ogb_mode = heldout_mode(config) == "ogb_scaffold"
        paradigm_mode = heldout_mode(config) == "paradigm"
        all_ids, all_y, all_w, _, _, _ = carve_out_heldout(
            all_ids, all_y, all_w, mode=heldout_mode(config), seed=random_split_seed(config))

    if paradigm_mode and config["dataset"].get("stopping_paradigm") == "sequential":
        from src.splits import sequential_scaffold_stopping_mask
        stopping_mask = sequential_scaffold_stopping_mask(all_ids, 1.0 / STOPPING_SLICE_FOLDS)
    elif paradigm_mode:
        if config["dataset"].get("stopping_paradigm") != "balanced":
            raise ValueError("dataset.stopping_paradigm: must be balanced|sequential")
        stopping_mask = assign_folds(all_ids, STOPPING_SLICE_FOLDS) == 0
    elif ogb_mode:
        # OGB / Hu et al. split: the stopping slice is OGB valid (143 molecules).
        from src.splits import ogb_split_masks
        stopping_mask = ogb_split_masks(all_ids)[1]
    elif random_mode:
        # On a random split the stopping slice is random too (1/9 of the
        # remainder, giving ~80/10/10 overall); the scaffold-grouped
        # assign_folds is not used here.
        rng = np.random.RandomState(1000 + random_split_seed(config))
        stopping_mask = np.zeros(len(all_ids), dtype=bool)
        stopping_mask[rng.permutation(len(all_ids))[:int(round(len(all_ids) / 9.0))]] = True
    else:
        stopping_mask = assign_folds(all_ids, STOPPING_SLICE_FOLDS) == 0
    train_mask = ~stopping_mask
    sider_train_dc = CombinedDataset(
        [all_ids[i] for i in range(len(all_ids)) if train_mask[i]], all_y[train_mask], all_w[train_mask])
    eval_dc = CombinedDataset(
        [all_ids[i] for i in range(len(all_ids)) if stopping_mask[i]], all_y[stopping_mask], all_w[stopping_mask])
    return tasks, sider_train_dc, eval_dc


def _build_group_data(config, aux_dataset_names, cache_dir, num_workers=None, carve_heldout_test=False):
    tasks, sider_train_dc, eval_dc = _split_sider(config, carve_heldout_test=carve_heldout_test)
    num_sider_tasks = len(tasks)
    sider_augment_copies = config["dataset"].get("sider_augment_copies", 0)

    combined_train, total_tasks, sider_mask = build_combined_train_set(
        sider_train_dc, aux_dataset_names, config["dataset"]["root"], sider_augment_copies)
    train_ds = MoleculeGraphDataset(combined_train, cache_dir=cache_dir, num_workers=num_workers)
    total_pos_rate = compute_pos_rate(combined_train.y, combined_train.w)

    eval_ds = MoleculeGraphDataset(eval_dc, cache_dir=config["dataset"]["processed_root"])

    # The same weighted sampling as prepare_data() in src/train.py. Without
    # it, SIDER is diluted among the large auxiliary datasets (Tox21/HIV/MUV)
    # and drops to about 5.7% of each batch; sider_batch_fraction guarantees
    # 50% SIDER per batch on average.
    sider_fraction = config["dataset"].get("sider_batch_fraction")
    sampler = None
    if sider_fraction is not None and sider_mask.sum() < len(sider_mask):
        n_sider = int(sider_mask.sum())
        n_aux = len(sider_mask) - n_sider
        weights = np.where(
            sider_mask,
            sider_fraction / max(n_sider, 1),
            (1 - sider_fraction) / max(n_aux, 1),
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(sider_mask), replacement=True)
        print(f"Weighted sampling: {sider_fraction*100:.0f}% SIDER per batch on average "
              f"(actual share: {n_sider/len(sider_mask)*100:.1f}%)")

    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=config["train"]["batch_size"])

    degenerate_tasks, _ = find_degenerate_tasks(tasks, sider_train_dc)
    return tasks, train_loader, eval_loader, degenerate_tasks, total_tasks, num_sider_tasks, total_pos_rate
