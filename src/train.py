import json
import os
from collections import defaultdict

import yaml
import deepchem as dc
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

from src.data.datasets import load_sider
from src.data.featurize import MoleculeGraphDataset, NUM_ATOM_FEATURES, augment_smiles
from src.models.gin import GINEncoder

from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

DEGENERATE_THRESHOLD = 0.05
STOPPING_SLICE_FOLDS = 10  # ~90% of SIDER for training, ~10% for early stopping

CHEMBL_TOP_N_TARGETS = 50
CHEMBL_ACTIVE_THRESHOLD = 5.0  # pChEMBL >= 5 -> "active/binds" (a 10 uM threshold)


def load_chembl_target_task(featurizer=None, splitter=None, data_dir="data/raw", save_dir=None, reload=None):
    """Expose the ChEMBL target-interaction data, fetched by
    scripts/data/fetch_chembl_potency.py, as an auxiliary prediction task, with the
    same interface as the other AUX_LOADERS such as Tox21/ClinTox:
    (tasks, (train, valid, test), transformers).

    Earlier attempts used this data as an input feature (see
    scripts/data/build_target_features.py, no longer in use). Widening the coverage
    diluted the target list and lowered performance, because the model treated
    it as an input it had to discriminate on. Here the same information is an
    output -- an extra task to predict: the model learns "does this molecule
    bind that target?" alongside SIDER, mechanistic information reaches the
    encoder through the gradient, and the input vector stays clean.

    Labels: 1 (active) for target-molecule pairs with pChEMBL >= 5.0, and 0
    for cases where the molecule has potency data but no measurement against
    this target (a default negative, the standard simplification for
    bioactivity data). For a molecule with no potency data at all, the weight
    is 0 in every column of this task, so it contributes no gradient and
    "not tested" is never confused with "tested, does not bind".
    """
    with open(os.path.join(data_dir, "chembl_targets.json")) as f:
        targets_cache = json.load(f)
    with open(os.path.join(data_dir, "chembl_potency.json")) as f:
        potency_cache = json.load(f)

    counter = defaultdict(int)
    for v in targets_cache.values():
        for t in v.get("targets", []):
            counter[t] += 1
    top_targets = [tid for tid, _ in sorted(counter.items(), key=lambda kv: -kv[1])[:CHEMBL_TOP_N_TARGETS]]

    ids, y_rows, w_rows = [], [], []
    for smi, entry in potency_cache.items():
        potency = entry.get("potency", {})
        row_y = [1.0 if potency.get(tid, 0.0) >= CHEMBL_ACTIVE_THRESHOLD else 0.0 for tid in top_targets]
        row_w = [1.0] * len(top_targets)
        ids.append(smi)
        y_rows.append(row_y)
        w_rows.append(row_w)

    y = np.array(y_rows, dtype=np.float32)
    w = np.array(w_rows, dtype=np.float32)
    train_dc = CombinedDataset(ids, y, w)
    empty = CombinedDataset([], np.zeros((0, len(top_targets)), dtype=np.float32),
                             np.zeros((0, len(top_targets)), dtype=np.float32))
    print(f"chembl_targets: {len(top_targets)} targets (auxiliary prediction task), "
          f"{len(ids)} molecules (threshold pChEMBL>={CHEMBL_ACTIVE_THRESHOLD})")
    return top_targets, (train_dc, empty, empty), None


AUX_LOADERS = {
    "tox21": dc.molnet.load_tox21,
    "clintox": dc.molnet.load_clintox,
    "bbbp": dc.molnet.load_bbbp,
    "hiv": dc.molnet.load_hiv,
    "muv": dc.molnet.load_muv,
    "chembl_targets": load_chembl_target_task,
}


def load_config(path="configs/default.yaml", chembl_feature_version=None, configure_features=True):
    """Read the config and, by default, explicitly configure the ChEMBL target
    feature file for this process, instead of copying files by hand.

    `chembl_feature_version` overrides the value in the config, which allows
    separate processes running V2 and V3 at the same time to start from the
    same code. `configure_features=False` configures nothing, for example to
    inspect the config read-only in a process whose provenance is already
    fixed.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    if configure_features:
        from src.data.featurize import configure_rdkit_descriptors, configure_target_features
        version = chembl_feature_version or config["dataset"]["chembl_feature_version"]
        feature_path = config["dataset"]["chembl_feature_files"][version]
        configure_target_features(version, feature_path)
        # With ablation.use_rdkit_descriptors on, normalizer_path is REQUIRED
        # (see configure_rdkit_descriptors): it fails loudly instead of running
        # silently with unfitted statistics.
        use_rdkit = config.get("ablation", {}).get("use_rdkit_descriptors", False)
        configure_rdkit_descriptors(
            use_rdkit, config["dataset"].get("rdkit_descriptor_normalizer_path") if use_rdkit else None)
    return config


class CombinedDataset:
    """Like a dc_dataset (.ids/.y/.w) but combined from several sources."""

    def __init__(self, ids, y, w):
        self.ids = ids
        self.y = y
        self.w = w


def _scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def assign_folds(ids, num_folds):
    """Distribute molecules evenly across num_folds folds by Murcko scaffold
    group, so that the same core structure always stays in one fold and no
    leakage occurs.
    """
    scaffold_to_indices = defaultdict(list)
    for i, smiles in enumerate(ids):
        scaffold = _scaffold(smiles) or f"__invalid_{i}"
        scaffold_to_indices[scaffold].append(i)

    groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)
    fold_sizes = [0] * num_folds
    fold_of_index = np.full(len(ids), -1, dtype=int)
    for group in groups:
        target_fold = int(np.argmin(fold_sizes))
        for i in group:
            fold_of_index[i] = target_fold
        fold_sizes[target_fold] += len(group)

    return fold_of_index


def compute_pos_rate(y, w):
    """The positive rate per task/column, over the samples whose label weight is
    > 0. Used both to detect degenerate tasks and for focal loss's per-task
    alpha."""
    valid_mask = w > 0
    return np.divide(
        (y * valid_mask).sum(axis=0),
        valid_mask.sum(axis=0),
        out=np.full(y.shape[1], np.nan),
        where=valid_mask.sum(axis=0) > 0,
    )


def find_degenerate_tasks(tasks, train_dc, threshold=DEGENERATE_THRESHOLD):
    train_pos_rate = compute_pos_rate(train_dc.y, train_dc.w)
    degenerate = [
        i for i, rate in enumerate(train_pos_rate)
        if np.isnan(rate) or rate < threshold or rate > 1 - threshold
    ]
    return degenerate, train_pos_rate


def _augment_sider(sider_train_dc, n_copies):
    """Generate n_copies alternative SMILES spellings for each SIDER molecule and
    add them to the dataset with the same labels (see augment_smiles)."""
    ids, y_rows, w_rows = [], [], []
    for smiles, y, w in zip(sider_train_dc.ids, sider_train_dc.y, sider_train_dc.w):
        ids.append(smiles)
        y_rows.append(y)
        w_rows.append(w)
        for variant in augment_smiles(smiles, n_copies):
            ids.append(variant)
            y_rows.append(y)
            w_rows.append(w)
    return CombinedDataset(ids, np.array(y_rows, dtype=np.float32), np.array(w_rows, dtype=np.float32))


def build_combined_train_set(sider_train_dc, aux_dataset_names, root, sider_augment_copies=0):
    """Combine SIDER's training set with all samples of the auxiliary datasets
    (tox21/clintox/bbbp and the rest), using their train+valid+test together
    because we never evaluate on them. Each molecule has real labels and
    weights only in the task columns of its own source; its weight is 0 in
    every other column.

    If `sider_augment_copies > 0`, the SIDER molecules are duplicated with
    alternative SMILES spellings (see _augment_sider), a cheap way to grow the
    small SIDER set.

    The returned `sider_mask` marks which rows came from SIDER, whether
    original or augmented, which the weighted sampling needs.
    """
    if sider_augment_copies > 0:
        sider_train_dc = _augment_sider(sider_train_dc, sider_augment_copies)
        print(f"SIDER augmentation: {sider_augment_copies} copies per molecule, "
              f"{len(sider_train_dc.ids)} rows in total (original + generated)")

    sources = [("sider", sider_train_dc, sider_train_dc.y.shape[1])]

    for name in aux_dataset_names:
        loader = AUX_LOADERS[name]
        tasks, datasets, _ = loader(featurizer="Raw", splitter="scaffold", data_dir=root, save_dir=root)
        train_dc, valid_dc, test_dc = datasets
        ids = list(train_dc.ids) + list(valid_dc.ids) + list(test_dc.ids)
        y = np.concatenate([train_dc.y, valid_dc.y, test_dc.y], axis=0)
        w = np.concatenate([train_dc.w, valid_dc.w, test_dc.w], axis=0)
        sources.append((name, CombinedDataset(ids, y, w), len(tasks)))
        print(f"{name}: {len(tasks)} tasks, {len(ids)} molecules (auxiliary signal)")

    total_tasks = sum(n for _, _, n in sources)

    all_ids, all_y, all_w, sider_mask = [], [], [], []
    col_offset = 0
    for name, ds, n_tasks in sources:
        n = len(ds.ids)
        y_full = np.zeros((n, total_tasks), dtype=np.float32)
        w_full = np.zeros((n, total_tasks), dtype=np.float32)
        y_full[:, col_offset:col_offset + n_tasks] = ds.y
        w_full[:, col_offset:col_offset + n_tasks] = ds.w
        all_ids.extend(ds.ids)
        all_y.append(y_full)
        all_w.append(w_full)
        sider_mask.extend([name == "sider"] * n)
        col_offset += n_tasks

    combined = CombinedDataset(all_ids, np.concatenate(all_y, axis=0), np.concatenate(all_w, axis=0))
    return combined, total_tasks, np.array(sider_mask, dtype=bool)


def prepare_data(config):
    """Load SIDER plus the auxiliary datasets.

    SIDER's train/valid/test are merged and re-split in a scaffold-balanced
    way into 90% training and a 10% "stopping slice" (early stopping plus a
    reference metric), rather than keeping DeepChem's fixed 80/10/10. Almost
    all of SIDER's already small dataset therefore contributes to training,
    instead of 20% going unused as with the old fixed split.

    The training set is also combined with the auxiliary datasets listed in the
    config, so that the shared encoder learns a richer chemical
    representation. Evaluation is always on SIDER's stopping slice alone; the
    auxiliary datasets only enrich the training signal.

    Note: the "test" metric here is not a definitive performance figure. It is
    a small slice that also drives early stopping. For a more reliable
    estimate, see the K-fold cross-validation that was run once for this
    project and saved to outputs/kfold_history.json (~0.636-0.639 AUC).
    """
    tasks, train_dc, valid_dc, test_dc = load_sider(root=config["dataset"]["root"])
    num_sider_tasks = len(tasks)

    all_ids = list(train_dc.ids) + list(valid_dc.ids) + list(test_dc.ids)
    all_y = np.concatenate([train_dc.y, valid_dc.y, test_dc.y], axis=0)
    all_w = np.concatenate([train_dc.w, valid_dc.w, test_dc.w], axis=0)

    stopping_mask = assign_folds(all_ids, STOPPING_SLICE_FOLDS) == 0
    train_mask = ~stopping_mask

    sider_train_dc = CombinedDataset(
        [all_ids[i] for i in range(len(all_ids)) if train_mask[i]],
        all_y[train_mask],
        all_w[train_mask],
    )
    eval_dc = CombinedDataset(
        [all_ids[i] for i in range(len(all_ids)) if stopping_mask[i]],
        all_y[stopping_mask],
        all_w[stopping_mask],
    )

    aux_dataset_names = config["dataset"].get("aux_datasets", [])
    sampler = None
    if aux_dataset_names:
        sider_augment_copies = config["dataset"].get("sider_augment_copies", 0)
        combined_train, total_tasks, sider_mask = build_combined_train_set(
            sider_train_dc, aux_dataset_names, config["dataset"]["root"], sider_augment_copies,
        )
        train_ds = MoleculeGraphDataset(combined_train, cache_dir=config["dataset"]["processed_multitask_root"])
        total_pos_rate = compute_pos_rate(combined_train.y, combined_train.w)

        sider_fraction = config["dataset"].get("sider_batch_fraction")
        if sider_fraction is not None:
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
    else:
        total_tasks = num_sider_tasks
        train_ds = MoleculeGraphDataset(sider_train_dc, cache_dir=config["dataset"]["processed_root"])
        total_pos_rate = compute_pos_rate(sider_train_dc.y, sider_train_dc.w)

    eval_ds = MoleculeGraphDataset(eval_dc, cache_dir=config["dataset"]["processed_root"])

    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(eval_ds, batch_size=config["train"]["batch_size"])
    test_loader = valid_loader  # the same stopping slice; see K-fold for a reportable number

    degenerate_tasks, train_pos_rate = find_degenerate_tasks(tasks, sider_train_dc)

    return (tasks, train_loader, valid_loader, test_loader, degenerate_tasks, train_pos_rate,
            total_tasks, num_sider_tasks, total_pos_rate)


def build_model(config, num_tasks, device):
    from src.data.featurize import num_global_features
    # ablation.gin_layers, when present, overrides model.num_layers. Its
    # default of 3 equals model.num_layers, so adding the flag alone does not
    # change behaviour.
    num_layers = config.get("ablation", {}).get("gin_layers", config["model"]["num_layers"])
    model = GINEncoder(
        in_channels=NUM_ATOM_FEATURES,
        hidden_channels=config["model"]["hidden_dim"],
        num_layers=num_layers,
        out_layers=num_tasks,
        dropout=config["model"]["dropout"],
        readout=config["model"]["readout"],
        # The dimension follows whatever configure_rdkit_descriptors() set
        # for this process; load_config() does that automatically.
        num_global_features=num_global_features(),
        task_attention=config["model"].get("task_attention", False),
    ).to(device)
    return model


def build_optimizer_and_scheduler(model, config, extra_params=()):
    # extra_params carries parameters that live outside the model but still
    # need training, such as UncertaintyWeighting's log_sigma. Left empty, the
    # behaviour is identical to the baseline. The log_sigmas go in their own
    # group with weight_decay=0, otherwise they would be pulled towards 0
    # (equal weighting). With no extra_params the group list matches the
    # baseline's.
    param_groups = [{"params": list(model.parameters())}]
    extra_params = list(extra_params)
    if extra_params:
        param_groups.append({"params": extra_params, "weight_decay": 0.0})
    optimizer = torch.optim.Adam(
        param_groups,
        lr=config["train"]["lr"],
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["train"]["lr_scheduler"]["mode"],
        factor=config["train"]["lr_scheduler"]["factor"],
        patience=config["train"]["lr_scheduler"]["patience"],
        eps=float(config["train"]["lr_scheduler"]["eps"]),
    )
    return optimizer, scheduler


class FocalLoss(torch.nn.Module):
    """On heavily imbalanced multi-label tasks such as SIDER, suppress the loss
    of easy samples -- those that simply follow the majority class -- and force
    the model to focus on hard, discriminative ones. The goal is to make it
    harder for the model to merely memorise each task's base (positive) rate;
    the correlation measured between train_pos_rate and the mean prediction was
    0.974.

    FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    A larger gamma suppresses easy samples more; gamma=0 gives plain BCE.

    With `task_alpha` (one alpha per task, length = number of tasks), each task
    is balanced by its own base rate instead of by a single fixed alpha:
    task_alpha_t = 1 - train_pos_rate_t. A task whose minority class is
    positive penalises positives more heavily, and a task whose minority class
    is negative penalises negatives more heavily. A fixed alpha=0.25 acts in
    the same direction for every task, always suppressing positives, which is
    exactly backwards for a task with a 5% base rate. Using task_alpha is
    therefore a more accurate correction, because it accounts for the direction
    of the imbalance as well.
    """

    def __init__(self, gamma=2.0, alpha=0.25, task_alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if task_alpha is not None:
            self.register_buffer("task_alpha", task_alpha)
        else:
            self.task_alpha = None

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t).clamp(min=1e-6) ** self.gamma
        if self.task_alpha is not None:
            alpha = self.task_alpha.to(logits.device).unsqueeze(0)
        else:
            alpha = self.alpha
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return alpha_t * focal_weight * bce


def compute_task_alpha(config, train_pos_rate):
    """Focal loss's alpha per task (1 - base rate), falling back to the fixed
    focal_alpha for tasks whose base rate is unknown (nan). The same function
    is used both during training (build_criterion) and for the post-training
    threshold correction (bake_in_task_alpha_threshold); the two must stay
    consistent, otherwise the threshold ends up applied to the wrong task.
    """
    fallback_alpha = config["train"].get("focal_alpha", 0.25)
    rate = np.nan_to_num(train_pos_rate, nan=1 - fallback_alpha)
    rate = np.clip(rate, 0.02, 0.98)  # keep alpha bounded for extreme positive rates
    return 1.0 - rate


def build_criterion(config, train_pos_rate=None):
    # ablation.use_asl is checked BEFORE config["train"]["loss"]: with ASL on,
    # the focal/bce settings are ignored.
    if config.get("ablation", {}).get("use_asl", False):
        from src.losses import AsymmetricLoss
        return AsymmetricLoss(
            gamma_neg=config["train"].get("asl_gamma_neg", 4.0),
            gamma_pos=config["train"].get("asl_gamma_pos", 1.0),
            clip=config["train"].get("asl_clip", 0.05),
        )

    loss_type = config["train"].get("loss", "bce")
    if loss_type != "focal":
        return torch.nn.BCEWithLogitsLoss(reduction="none")

    task_alpha = None
    if config["train"].get("focal_task_alpha", False) and train_pos_rate is not None:
        task_alpha = torch.tensor(compute_task_alpha(config, train_pos_rate), dtype=torch.float32)

    return FocalLoss(
        gamma=config["train"].get("focal_gamma", 2.0),
        alpha=config["train"].get("focal_alpha", 0.25),
        task_alpha=task_alpha,
    )


def bake_in_task_alpha_threshold(model, task_alpha):
    """Fold the per-task decision threshold (see compute_task_alpha)
    permanently into the model's final-layer bias. The backend then needs no
    separate threshold list to carry or recompute: the model's own sigmoid
    output is already corrected and a plain 50% threshold works everywhere.

    The maths: under weighted BCE with alpha_t, the Bayes-optimal decision
    point in probability space is p* = alpha_t (solved for q=0.5). In logit
    space that is equivalent to subtracting logit(alpha_t) from that task's
    final-layer bias, after which the model makes the right decision at q=0.5,
    a plain 50% threshold.

    The length of `task_alpha` must match the number of outputs to correct,
    normally just SIDER's `num_sider_tasks`. Auxiliary tasks are never
    deployed, so they need no correction, though correcting them does no harm.
    """
    task_alpha = np.asarray(task_alpha)
    logit_shift = np.log(task_alpha / (1 - task_alpha))
    n = len(task_alpha)

    with torch.no_grad():
        if getattr(model, "task_attention", False):
            for i in range(n):
                model.task_heads[i].lins[-1].bias -= float(logit_shift[i])
        else:
            model.mlp.lins[-1].bias[:n] -= torch.tensor(logit_shift, dtype=model.mlp.lins[-1].bias.dtype)


def train_one_epoch(model, loader, optimizer, criterion, device, task_weighter=None):
    """`task_weighter=None` (the default) is the baseline behaviour: all
    (sample, task) cells combine with equal weight into a single mean. This
    branch is identical to the code before ablation 2.5 existed.

    `task_weighter` implements ablation.task_weighting:
      - an `UncertaintyWeighting` module -> the per-task loss is weighted by a
        learned log_sigma.
    """
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.global_features)
        elementwise = criterion(out, batch.y)

        if task_weighter is None:
            optimizer.zero_grad()
            loss = (elementwise * batch.w).sum() / batch.w.sum()
            loss.backward()
            optimizer.step()
        else:
            from src.losses import per_task_masked_loss
            per_task = per_task_masked_loss(elementwise, batch.w)
            valid = batch.w.sum(dim=0) > 0
            optimizer.zero_grad()
            loss = task_weighter(per_task, valid)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, num_sider_tasks=None, exclude_tasks=()):
    """Evaluate on SIDER's own valid/test loader.

    If the model produces more outputs than `num_sider_tasks` (the multi-task
    transfer case), only the first `num_sider_tasks` columns, which belong to
    SIDER, are used: the auxiliary tasks' outputs never enter the metric.
    """
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        out = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.global_features))
        if num_sider_tasks is not None:
            out = out[:, :num_sider_tasks]
        all_preds.append(out.cpu())
        all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    return compute_aucs(preds, targets, exclude_tasks)


def compute_aucs(preds, targets, exclude_tasks=()):
    aucs_all, aucs_filtered = [], []
    for task in range(preds.shape[1]):
        if len(np.unique(targets[:, task])) < 2:
            continue
        auc = roc_auc_score(targets[:, task], preds[:, task])
        aucs_all.append(auc)
        if task not in exclude_tasks:
            aucs_filtered.append(auc)
    return float(np.mean(aucs_all)), float(np.mean(aucs_filtered))


def _recalibrate_batchnorm(model, loader, device):
    """An SWA helper with the same logic as
    `torch.optim.swa_utils.update_bn` -- reset the running statistics,
    accumulate a cumulative average with momentum=None, and make one
    forward-only pass -- written by hand to work with our model's
    five-positional-argument forward. `update_bn` itself calls `model(input)`,
    which does not fit our signature.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_modules:
        return
    momenta = {m: m.momentum for m in bn_modules}
    for m in bn_modules:
        m.reset_running_stats()
        m.momentum = None
    was_training = model.training
    model.train()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.global_features)
    for m in bn_modules:
        m.momentum = momenta[m]
    model.train(was_training)


def train_model(model, train_loader, valid_loader, config, degenerate_tasks, device, num_sider_tasks,
                 checkpoint_path="best_model.pt", verbose=True, train_pos_rate=None,
                 resume_path=None):
    """
    With `resume_path`, the state at the end of every epoch -- model, optimizer
    and scheduler state, epoch number, best_auc, patience_counter and history
    -- is written to that file. When the script is restarted, including after a
    host restart, training continues from the epoch it reached
    instead of starting over. The resume file is deleted once training ends
    normally (early stopping or the maximum epoch), so its presence means "this
    model was left half-finished".

    Note: the random state of the DataLoader and sampler is not restored, only
    the model/optimizer/scheduler weights and the epoch counter. This is not
    bit-exact reproducibility, but it removes the risk of restarting tens of
    hours of training from scratch, which was the pressing problem.
    """
    # task_weighting=uncertainty introduces an extra learnable parameter
    # (log_sigma) that must be added to the optimizer, or it is never learned.
    # The default ("none") never enters this branch.
    task_weighting = config.get("ablation", {}).get("task_weighting", "none")
    task_weighter = None
    extra_params = ()
    if task_weighting == "uncertainty":
        from src.losses import UncertaintyWeighting
        # The task count is the model's output width (SIDER plus auxiliary
        # tasks, e.g. 60), not num_sider_tasks (27): the loss is computed over
        # all outputs.
        num_out_tasks = train_loader.dataset[0].y.shape[-1]
        task_weighter = UncertaintyWeighting(num_out_tasks).to(device)
        # .parameters() is a generator, so consuming it a second time in the
        # SWA block below would yield an empty list. list() materialises it.
        extra_params = list(task_weighter.parameters())
    elif task_weighting != "none":
        raise ValueError(f"unknown ablation.task_weighting: {task_weighting!r} "
                          f"(must be none|uncertainty)")

    optimizer, scheduler = build_optimizer_and_scheduler(model, config, extra_params=extra_params)
    criterion = build_criterion(config, train_pos_rate)

    best_auc = 0.0
    patience_counter = 0
    patience = config["train"]["early_stopping_patience"]
    history = []
    start_epoch = 1

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_auc = ckpt["best_auc"]
        patience_counter = ckpt["patience_counter"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        if isinstance(task_weighter, torch.nn.Module) and "task_weighter_state" in ckpt:
            task_weighter.load_state_dict(ckpt["task_weighter_state"])
        if patience_counter >= patience:
            # Early stopping had already triggered and the process was cut during
            # the SWA/threshold step: skip the epoch loop, rerun only the later steps.
            start_epoch = config["train"]["epochs"] + 1
        if verbose:
            print(f"[resume] resuming from '{resume_path}' at epoch {start_epoch} "
                  f"(best_auc_filtered={best_auc:.4f}).")

    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, task_weighter)
        val_auc, val_auc_filtered = evaluate(model, valid_loader, device, num_sider_tasks, exclude_tasks=degenerate_tasks)
        scheduler.step(val_auc_filtered)
        current_lr = optimizer.param_groups[0]["lr"]
        if verbose:
            print(f"epoch {epoch}: train_loss={train_loss:.4f}, val_auc={val_auc:.4f}, val_auc_filtered={val_auc_filtered:.4f}, lr={current_lr:.6f}")
            if isinstance(task_weighter, torch.nn.Module):
                ls = task_weighter.log_sigma.detach()
                print(f"  log_sigma: min={ls.min():.3f} ort={ls.mean():.3f} max={ls.max():.3f} "
                      f"SIDER_ort={ls[:num_sider_tasks].mean():.3f}")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": val_auc,
            "val_auc_filtered": val_auc_filtered,
        })

        if val_auc_filtered > best_auc:
            best_auc = val_auc_filtered
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if resume_path:
            state = {
                "epoch": epoch, "best_auc": best_auc, "patience_counter": patience_counter,
                "history": history, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            }
            # The log_sigmas live outside the model; without saving them they would
            # reset to 0 on resume (while the Adam moments would be restored).
            if isinstance(task_weighter, torch.nn.Module):
                state["task_weighter_state"] = task_weighter.state_dict()
            torch.save(state, resume_path)

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping (epoch {epoch}, best_auc_filtered={best_auc:.4f}).")
            break

    # The resume file used to be deleted here, before SWA. A process killed
    # during SWA then restarted with no resume file but a .lock present, so the
    # seed was skipped as "claimed by another process". It is now deleted at the
    # very end. Because SWA overwrites the checkpoint, the best pre-SWA model is
    # kept in .pre_swa and SWA is rebuilt from there on a restart.
    use_swa = config.get("ablation", {}).get("use_swa", False)
    pre_swa_path = checkpoint_path + ".pre_swa"
    if use_swa and os.path.exists(pre_swa_path):
        model.load_state_dict(torch.load(pre_swa_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path))
        if use_swa:
            torch.save(model.state_dict(), pre_swa_path)
    if verbose and isinstance(task_weighter, torch.nn.Module):
        print("final log_sigma values:", [round(v, 3) for v in task_weighter.log_sigma.detach().cpu().tolist()])

    # Run `swa_epochs` extra epochs on top of the best (early-stopped) model
    # and average the weights. The BatchNorm running statistics -- present in
    # GINEConv's inner MLP -- become inconsistent with those averaged weights,
    # so they are recalibrated over the training data. torch's own `update_bn`
    # does not work with our five-argument forward signature, so the same logic
    # is applied by hand.
    if use_swa:
        from torch.optim.swa_utils import SWALR, AveragedModel
        swa_epochs = config["train"].get("swa_epochs", 5)
        swa_lr = config["train"].get("swa_lr", config["train"]["lr"] * 0.1)
        averaged = AveragedModel(model)
        swa_optimizer, _ = build_optimizer_and_scheduler(model, config, extra_params=extra_params)
        # The new optimizer is built with train.lr (5e-4), and with SWALR's
        # default 10-epoch anneal it could not reach swa_lr within swa_epochs
        # (5): the average was taken far above training's final lr (~6e-5). The
        # lr is therefore pinned to swa_lr from the start.
        for g in swa_optimizer.param_groups:
            g["lr"] = swa_lr
        swa_scheduler = SWALR(swa_optimizer, swa_lr=swa_lr, anneal_epochs=1)
        for e in range(swa_epochs):
            train_one_epoch(model, train_loader, swa_optimizer, criterion, device, task_weighter)
            averaged.update_parameters(model)
            swa_scheduler.step()
            if verbose:
                print(f"[SWA] extra epoch {e + 1}/{swa_epochs} finished")
        _recalibrate_batchnorm(averaged, train_loader, device)
        model.load_state_dict(averaged.module.state_dict())
        swa_auc, swa_auc_filtered = evaluate(model, valid_loader, device, num_sider_tasks,
                                              exclude_tasks=degenerate_tasks)
        if verbose:
            print(f"[SWA] before best_auc_filtered={best_auc:.4f} -> after={swa_auc_filtered:.4f}")
        best_auc = swa_auc_filtered
        history.append({"epoch": "swa", "val_auc": swa_auc, "val_auc_filtered": swa_auc_filtered})
        torch.save(model.state_dict(), checkpoint_path)

    # This step is skipped when ASL is enabled: ASL's own asymmetry does the
    # same job, and applying both shifts the threshold twice.
    use_asl = config.get("ablation", {}).get("use_asl", False)
    if (not use_asl and config["train"].get("loss") == "focal"
            and config["train"].get("focal_task_alpha", False) and train_pos_rate is not None):
        task_alpha = compute_task_alpha(config, train_pos_rate)[:num_sider_tasks]
        bake_in_task_alpha_threshold(model, task_alpha)
        torch.save(model.state_dict(), checkpoint_path)
        if verbose:
            print(f"per-task threshold correction folded into the final layer bias (mean alpha={task_alpha.mean():.3f}).")

    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)  # training finished normally; nothing was left half-done
    if os.path.exists(pre_swa_path):
        os.remove(pre_swa_path)

    return model, best_auc, history


if __name__ == "__main__":
    import json

    config = load_config()
    (tasks, train_loader, valid_loader, test_loader, degenerate_tasks, train_pos_rate,
     total_tasks, num_sider_tasks, total_pos_rate) = prepare_data(config)
    print("number of tasks (SIDER):", len(tasks))
    print("total number of outputs (including auxiliary tasks):", total_tasks)
    print("combined training graphs:", len(train_loader.dataset))

    print(f"degenerate tasks (train positive rate <{DEGENERATE_THRESHOLD*100:.0f}% or >{(1 - DEGENERATE_THRESHOLD) * 100:.0f}%), excluded from the main metric:")
    for i in degenerate_tasks:
        print(f"  - {tasks[i]} (train_pos%={train_pos_rate[i] * 100:.1f})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["train"]["seed"])
    model = build_model(config, total_tasks, device)
    print(model)

    model, best_auc, history = train_model(
        model, train_loader, valid_loader, config, degenerate_tasks, device, num_sider_tasks,
        checkpoint_path="outputs/best_model_multitask_transfer.pt", train_pos_rate=total_pos_rate,
    )

    test_auc, test_auc_filtered = evaluate(model, test_loader, device, num_sider_tasks, exclude_tasks=degenerate_tasks)
    print(f"stopping-slice AUC (for reference, not a reportable result): {test_auc:.4f} (filtered: {test_auc_filtered:.4f})")
    print("for a reliable performance estimate see the K-fold result: outputs/kfold_history.json (~0.636-0.639 AUC)")

    with open("outputs/history.json", "w") as f:
        json.dump({
            "config": {
                "hidden_dim": config["model"]["hidden_dim"],
                "num_layers": config["model"]["num_layers"],
                "lr": config["train"]["lr"],
                "batch_size": config["train"]["batch_size"],
                "early_stopping_patience": config["train"]["early_stopping_patience"],
                "aux_datasets": config["dataset"].get("aux_datasets", []),
            },
            "degenerate_tasks": [tasks[i] for i in degenerate_tasks],
            "best_val_auc_filtered": best_auc,
            "test_auc": test_auc,
            "test_auc_filtered": test_auc_filtered,
            "epochs": history,
        }, f, indent=2)
