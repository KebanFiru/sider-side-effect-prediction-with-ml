"""Fetches ChEMBL target + potency data for the Tox21 + ClinTox + BBBP molecules
as well. APPENDS to the existing data/raw/chembl_targets.json and
chembl_potency.json (the 1427 SIDER molecules are untouched); thanks to the
shared key scheme (SMILES), featurize.py uses the new molecules without any
code change.
"""
import json
import os
import time
import warnings
import urllib.request
import urllib.error

warnings.filterwarnings("ignore")
import deepchem as dc
from rdkit import Chem

TARGETS_PATH = "data/raw/chembl_targets.json"
POTENCY_PATH = "data/raw/chembl_potency.json"
SLEEP = 0.12


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def resolve_and_fetch(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"resolved": False, "chembl_id": None, "targets": []}, {}
    inchikey = Chem.MolToInchiKey(mol)
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_structures__standard_inchi_key={inchikey}"
    data = fetch_json(url)
    if not data or not data.get("molecules"):
        return {"resolved": False, "chembl_id": None, "targets": []}, {}
    chembl_id = data["molecules"][0]["molecule_chembl_id"]

    act_url = (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        f"?molecule_chembl_id={chembl_id}&target_organism=Homo+sapiens"
        "&pchembl_value__isnull=false&limit=100"
    )
    act_data = fetch_json(act_url)
    targets, potency = [], {}
    if act_data:
        for a in act_data.get("activities", []):
            tid = a.get("target_chembl_id")
            pval = a.get("pchembl_value")
            if not tid:
                continue
            if tid not in targets:
                targets.append(tid)
            if pval is not None:
                try:
                    pval = float(pval)
                    if tid not in potency or pval > potency[tid]:
                        potency[tid] = pval
                except (TypeError, ValueError):
                    pass
    return {"resolved": True, "chembl_id": chembl_id, "targets": sorted(targets)}, potency


def main():
    dataset_loaders = [
        ("tox21", dc.molnet.load_tox21),
        ("clintox", dc.molnet.load_clintox),
        ("bbbp", dc.molnet.load_bbbp),
    ]

    smiles_list = []
    seen = set()
    for name, loader in dataset_loaders:
        _, datasets, _ = loader(featurizer="Raw", splitter=None, reload=True, data_dir="data/raw")
        ids = datasets[0].ids
        for smi in ids:
            if smi not in seen:
                seen.add(smi)
                smiles_list.append(smi)
        print(f"{name}: {len(ids)} molecules loaded (cumulative unique: {len(smiles_list)})")

    targets_cache = load_json(TARGETS_PATH)
    potency_cache = load_json(POTENCY_PATH)

    total = len(smiles_list)
    todo = [s for s in smiles_list if s not in targets_cache]
    print(f"Unique molecules: {total}, already fetched (from SIDER): {total - len(todo)}, to fetch: {len(todo)}")

    for i, smi in enumerate(todo):
        t_entry, potency = resolve_and_fetch(smi)
        targets_cache[smi] = t_entry
        if potency:
            potency_cache[smi] = {"chembl_id": t_entry["chembl_id"], "potency": potency}
        time.sleep(SLEEP)

        if (i + 1) % 100 == 0 or (i + 1) == len(todo):
            save_json(TARGETS_PATH, targets_cache)
            save_json(POTENCY_PATH, potency_cache)
            resolved = sum(1 for v in targets_cache.values() if v.get("resolved"))
            print(f"[{i+1}/{len(todo)}] saved (cache total={len(targets_cache)}, resolved={resolved})", flush=True)

    save_json(TARGETS_PATH, targets_cache)
    save_json(POTENCY_PATH, potency_cache)
    print("DONE.")


if __name__ == "__main__":
    main()
