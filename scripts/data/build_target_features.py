"""ChEMBL target features, version 2 (the version used by GIN-v2), built from
chembl_potency.json (smiles -> {chembl_id, potency: {target: pChEMBL}}):
target selection and coverage restricted to SIDER molecules, potency as raw
pChEMBL / 10, the 100 most frequent targets + a has_data flag (101 features).

V2 was chosen among the variants below on the earlier validation protocol
(before the held-out test set existed; numbers are not comparable with the
clean-test results):
  V1) binary,                       SIDER-only (703)  -> AUC 0.656
  V2) potency raw /10,              SIDER-only (703)  -> AUC 0.648/0.682  <- this file
  V3) potency clipped to 5-9,       expanded (5255)   -> AUC 0.645
  V4) binary,                       expanded (5255)   -> AUC 0.634
  V5) potency 5-9 + SIDER targets,  expanded (5255)   -> AUC 0.649

Output: data/raw/chembl_target_features.json
"""
import json

import pandas as pd
from rdkit import Chem

PCHEMBL_SCALE = 10.0

with open("data/raw/chembl_targets.json") as f:
    targets_cache = json.load(f)
with open("data/raw/chembl_potency.json") as f:
    potency_cache = json.load(f)

sider_smiles = set(pd.read_csv("data/raw/sider.csv.gz")["smiles"].tolist())
print(f"SIDER molecules: {len(sider_smiles)} (coverage and target selection fixed to these)")

from collections import Counter
counter = Counter()
for smi in sider_smiles:
    v = targets_cache.get(smi)
    if v:
        for t in v.get("targets", []):
            counter[t] += 1

TOP_N = 100
top_targets = [tid for tid, _ in counter.most_common(TOP_N)]
target_index = {tid: i for i, tid in enumerate(top_targets)}

inchikey_to_vector = {}
resolved_count = 0
with_any_top_target = 0

for smi in sider_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    inchikey = Chem.MolToInchiKey(mol)
    entry = targets_cache.get(smi, {})

    vec = [0.0] * (TOP_N + 1)  # last element: has_data flag
    has_data = bool(entry.get("resolved") and entry.get("targets"))
    if has_data:
        resolved_count += 1
        pot_entry = potency_cache.get(smi)
        potency = pot_entry["potency"] if pot_entry else {}
        hit = False
        for tid, idx in target_index.items():
            if tid in potency:
                vec[idx] = potency[tid] / PCHEMBL_SCALE
                hit = True
        if hit:
            with_any_top_target += 1
        vec[TOP_N] = 1.0

    inchikey_to_vector[inchikey] = vec

out = {
    "top_targets": top_targets,
    "num_features": TOP_N + 1,
    "feature_type": "pchembl_potency_v2_sider_only",
    "pchembl_scale": PCHEMBL_SCALE,
    "inchikey_to_vector": inchikey_to_vector,
}
with open("data/raw/chembl_target_features.json", "w") as f:
    json.dump(out, f)

print(f"Vectors (one per molecule): {len(inchikey_to_vector)}")
print(f"With target data (has_data=1): {resolved_count}")
print(f"With a potency value for at least one top-100 target: {with_any_top_target}")
print(f"Feature dimension: {TOP_N + 1}")
