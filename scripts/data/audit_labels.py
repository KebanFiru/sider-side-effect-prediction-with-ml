"""Label audit report.

(A) Internal consistency of SIDER (duplicate / conflicting labels, per-task
positive-rate anomalies).
(B) For the OOD / literature set (if /tmp/final_mols.json exists): leakage
check against SIDER and completeness of provenance (source / confidence).

Usage: PYTHONPATH=. python -m scripts.data.audit_labels
"""
import json
import os

import numpy as np

from src.data.datasets import load_sider
from src.label_audit import (
    audit_ood_provenance,
    check_ood_sider_leakage,
    find_duplicate_label_conflicts,
    per_task_label_anomalies,
)

OOD_MOLS_PATH = "/tmp/final_mols.json"
OUT_PATH = "/tmp/label_audit_report.json"


def main():
    tasks, train_dc, valid_dc, test_dc = load_sider(root="data/raw")
    all_ids = list(train_dc.ids) + list(valid_dc.ids) + list(test_dc.ids)
    all_y = np.concatenate([train_dc.y, valid_dc.y, test_dc.y], axis=0)
    all_w = np.concatenate([train_dc.w, valid_dc.w, test_dc.w], axis=0)

    print(f"=== A: SIDER internal consistency ({len(all_ids)} molecules, {len(tasks)} tasks) ===")
    conflicts, unparsed = find_duplicate_label_conflicts(all_ids, all_y, all_w)
    print(f"SMILES RDKit cannot parse: {len(unparsed)}")
    print(f"Duplicate groups with conflicting labels: {len(conflicts)}")
    for c in conflicts[:10]:
        print(f"  {c['canonical'][:60]}...  ({c['n_duplicates']} copies, "
              f"conflicts in {len(c['conflicting_tasks'])} tasks)")

    anomalies = per_task_label_anomalies(tasks, all_y, all_w)
    anomalous = [a for a in anomalies if a["anomalous"]]
    print(f"\nPer-task positive-rate anomalies (outside 1%-99%): {len(anomalous)}/{len(tasks)}")
    for a in anomalous:
        print(f"  {a['task']}: n={a['n_labeled']}, pos_rate={a['pos_rate']}  -- {a['reason']}")

    ood_leaks, ood_missing_provenance = [], []
    if os.path.exists(OOD_MOLS_PATH):
        ood_data = json.load(open(OOD_MOLS_PATH))
        ood_entries = [m for m in ood_data["mols"] if m["group"] not in ("A-SIDER", "Q")]
        print(f"\n=== B: OOD / literature set audit ({len(ood_entries)} molecules) ===")

        ood_smiles = [(m["name"], m["smi"]) for m in ood_entries]
        ood_leaks = check_ood_sider_leakage(ood_smiles, all_ids)
        print(f"Canonical SMILES shared with SIDER (leakage): {len(ood_leaks)}")
        for leak in ood_leaks:
            print(f"  {leak['name']}: {leak['smi'][:60]}...")

        ood_missing_provenance = audit_ood_provenance(ood_entries)
        print(f"\nEntries missing provenance (source / confidence): "
              f"{len(ood_missing_provenance)}/{len(ood_entries)}")
        if len(ood_missing_provenance) == len(ood_entries):
            print("  ALL missing -- the current final_mols.json format has no provenance fields.")
            print("  see data/raw/ood_provenance_template.json (to be filled in by hand, never invented).")
    else:
        print(f"\n=== B skipped: {OOD_MOLS_PATH} not found ===")

    report = {
        "n_sider_molecules": len(all_ids),
        "n_tasks": len(tasks),
        "n_unparsed_smiles": len(unparsed),
        "n_duplicate_label_conflicts": len(conflicts),
        "duplicate_label_conflicts": conflicts,
        "n_task_anomalies": len(anomalous),
        "task_anomalies": anomalous,
        "n_ood_sider_leaks": len(ood_leaks),
        "ood_sider_leaks": ood_leaks,
        "n_ood_missing_provenance": len(ood_missing_provenance),
        "ood_missing_provenance": ood_missing_provenance,
    }
    json.dump(report, open(OUT_PATH, "w"), indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
