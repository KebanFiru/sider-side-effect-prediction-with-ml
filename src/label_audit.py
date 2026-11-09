"""Label audit.

Scope:
  (3.A) SIDER label consistency: whether several copies of the same molecule
        (by canonical SMILES) appear with different labels across the merged
        train/valid/test set, and per-task positive-rate anomalies (very rare
        or very common tasks -- a wider check than the "single class" case
        `find_degenerate_tasks` already detects).
  (3.B) The OOD (literature) set: for the B-LIT/C-LitTop30/C-Top30World groups
        in `/tmp/final_mols.json`, (a) whether their canonical SMILES overlap
        SIDER, which is a leakage check, and (b) whether each entry has a
        source and confidence field, which is a provenance completeness check.

Important: this module never invents literature sources. Check (b) reports a
missing source field as "missing" rather than assigning a fabricated one.
Filling the sources in is a separate, manual research step (see
data/raw/ood_provenance_template.json).
"""
import json
from collections import defaultdict

import numpy as np
from rdkit import Chem


def canonical_smiles(smi):
    """The RDKit canonical SMILES, or None if it cannot be parsed. The caller
    should report those in a separate "unparseable" list rather than skipping
    them silently."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def find_duplicate_label_conflicts(ids, y, w):
    """Finds molecules sharing a canonical SMILES that carry DIFFERENT labels
    for at least one task (among cells with w=1).

    Returns (list of dicts {canonical, raw_ids, n_duplicates, conflicting_tasks},
    unparsed SMILES). conflicting_tasks: task indices whose y values disagree
    where w=1.
    """
    canon_to_idxs = defaultdict(list)
    unparsed = []
    for i, smi in enumerate(ids):
        c = canonical_smiles(smi)
        if c is None:
            unparsed.append(smi)
            continue
        canon_to_idxs[c].append(i)

    conflicts = []
    for canon, idxs in canon_to_idxs.items():
        if len(idxs) < 2:
            continue
        n_tasks = y.shape[1]
        conflicting_tasks = []
        for t in range(n_tasks):
            observed = {float(y[i, t]) for i in idxs if w[i, t] > 0}
            if len(observed) > 1:
                conflicting_tasks.append(t)
        if conflicting_tasks:
            conflicts.append({
                "canonical": canon,
                "raw_ids": [ids[i] for i in idxs],
                "n_duplicates": len(idxs),
                "conflicting_tasks": conflicting_tasks,
            })
    return conflicts, unparsed


def per_task_label_anomalies(tasks, y, w, min_rate=0.01, max_rate=0.99):
    """The number of labelled samples (w=1) and the positive rate per task. A
    rate below `min_rate` or above `max_rate` is flagged as unusually
    imbalanced, unless the task is single-class, which is find_degenerate_tasks'
    job.
    """
    report = []
    for t, name in enumerate(tasks):
        mask = w[:, t] > 0
        n_labeled = int(mask.sum())
        if n_labeled == 0:
            report.append({"task": name, "n_labeled": 0, "pos_rate": None, "anomalous": True,
                            "reason": "no labelled sample at all"})
            continue
        pos_rate = float(y[mask, t].mean())
        n_classes = len(np.unique(y[mask, t]))
        anomalous = n_classes > 1 and (pos_rate < min_rate or pos_rate > max_rate)
        report.append({
            "task": name, "n_labeled": n_labeled, "pos_rate": round(pos_rate, 4),
            "anomalous": anomalous,
            "reason": f"positive rate {pos_rate*100:.2f}%, outside the thresholds" if anomalous else None,
        })
    return report


def check_ood_sider_leakage(ood_smiles, sider_ids):
    """Finds OOD / literature molecules whose canonical SMILES matches any
    SIDER molecule (train+valid+test); such a match would be leakage that
    breaks the "truly unseen" claim.
    """
    sider_canon = set()
    for smi in sider_ids:
        c = canonical_smiles(smi)
        if c is not None:
            sider_canon.add(c)

    leaks = []
    for entry_name, smi in ood_smiles:
        c = canonical_smiles(smi)
        if c is not None and c in sider_canon:
            leaks.append({"name": entry_name, "smi": smi, "canonical": c})
    return leaks


def audit_ood_provenance(ood_entries, required_fields=("source", "confidence")):
    """Check whether each OOD entry carries the `required_fields`. A missing
    field is reported as missing; no value is ever fabricated.
    """
    missing = []
    for entry in ood_entries:
        absent = [f for f in required_fields if not entry.get(f)]
        if absent:
            missing.append({"name": entry.get("name"), "group": entry.get("group"),
                             "missing_fields": absent})
    return missing
