"""Training of one 5-seed GIN-v2 group (used by the scripts in scripts/train/).

The manifest is updated as soon as each model finishes, so that a killed
process resumes where it left off (finished models are skipped).
"""
import json
import os

import torch

from src.provenance import manifest_entry
from src.train import build_model, load_config, train_model
from src.train_ensemble_hetero import CKPT_DIR, _build_group_data


def _load_manifest(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_manifest(path, manifest):
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def _claim(lock_path):
    try:
        os.close(os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def run_group(config, aux_dataset_names, cache_dir, indices, ckpt_prefix, group_label, manifest_path,
              carve_heldout_test=False, num_workers=None):
    """`carve_heldout_test=True` removes the genuinely untouched ~14.3%
    scaffold test split, carved out by src/splits.py, from both training and
    the stopping slice (see scripts/train/scaffold204_baseline.py). The default is
    False, so the older group2a/3 calls in __main__ below are unaffected.

    `num_workers`: MoleculeGraphDataset's default is
    `min(os.cpu_count(), 1)`, i.e. always 1, even on a 16-core machine. With no
    cache available (a new molecule set or held-out split), featurizing ~150K
    molecules on one core takes hours. Passing a value here, e.g. 4, enables
    real multiprocessing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks, train_loader, valid_loader, degenerate_tasks, total_tasks, num_sider_tasks, total_pos_rate = \
        _build_group_data(config, aux_dataset_names, cache_dir, num_workers=num_workers,
                          carve_heldout_test=carve_heldout_test)

    for i in indices:
        ckpt_path = os.path.join(CKPT_DIR, f"{ckpt_prefix}{i}.pt")
        resume_path = ckpt_path + ".resume"
        lock_path = os.path.join(CKPT_DIR, f"{ckpt_prefix}{i}.lock")
        manifest = _load_manifest(manifest_path)
        if any(e["path"] == ckpt_path and e["group"] == group_label for e in manifest):
            print(f"[{ckpt_prefix}{i}] already finished, skipping")
            continue
        # When a resume file exists, a stale .lock -- left by an earlier
        # process that most likely died with a host restart --
        # must not block us from continuing. Without a resume file the lock is
        # still a valid "another process claimed this" signal.
        if not os.path.exists(resume_path) and not _claim(lock_path):
            print(f"[{ckpt_prefix}{i}] claimed by another process, skipping")
            continue

        torch.manual_seed(1000 + i)
        model = build_model(json.loads(json.dumps(config)), total_tasks, device)
        print(f"[{ckpt_prefix}{i}] training starts ...")
        model, best_auc, _ = train_model(
            model, train_loader, valid_loader, config, degenerate_tasks, device, num_sider_tasks,
            checkpoint_path=ckpt_path, verbose=True, train_pos_rate=total_pos_rate,
            resume_path=resume_path,
        )
        print(f"[{ckpt_prefix}{i}] finished: best_val_auc_filtered={best_auc:.4f}")

        manifest = _load_manifest(manifest_path)
        manifest.append(manifest_entry(seed=1000 + i, config=config, extra={
            "path": ckpt_path, "arch": "gin", "group": group_label, 
            "total_tasks": total_tasks, "num_sider_tasks": num_sider_tasks,
            "best_val_auc_filtered": best_auc,
        }))
        _save_manifest(manifest_path, manifest)
        del model
        torch.cuda.empty_cache()

    print(f"Group {group_label} finished.")
