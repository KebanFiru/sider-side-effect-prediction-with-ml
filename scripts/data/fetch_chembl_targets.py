"""Fetches the ChEMBL InChIKey lookup + bioactivity target list for all SIDER
molecules and saves it to data/raw/chembl_targets.json. After an interruption
or error it reads the existing cache and only completes the missing molecules
(resumable).
"""
import json
import os
import time
import urllib.request
import urllib.error

import pandas as pd
from rdkit import Chem

CACHE_PATH = "data/raw/chembl_targets.json"
SIDER_PATH = "data/raw/sider.csv.gz"
SLEEP = 0.12


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_PATH)


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def get_targets_for_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"resolved": False, "chembl_id": None, "targets": []}
    inchikey = Chem.MolToInchiKey(mol)

    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_structures__standard_inchi_key={inchikey}"
    data = fetch_json(url)
    if not data or not data.get("molecules"):
        return {"resolved": False, "chembl_id": None, "targets": []}
    chembl_id = data["molecules"][0]["molecule_chembl_id"]

    act_url = (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        f"?molecule_chembl_id={chembl_id}&target_organism=Homo+sapiens"
        "&pchembl_value__isnull=false&limit=100"
    )
    act_data = fetch_json(act_url)
    targets = []
    if act_data:
        targets = sorted(set(
            a.get("target_chembl_id") for a in act_data.get("activities", [])
            if a.get("target_chembl_id")
        ))
    return {"resolved": True, "chembl_id": chembl_id, "targets": targets}


def main():
    df = pd.read_csv(SIDER_PATH)
    smiles_list = df["smiles"].tolist()
    cache = load_cache()

    total = len(smiles_list)
    done_before = len(cache)
    print(f"Molecules: {total}, already in cache: {done_before}")

    for i, smi in enumerate(smiles_list):
        if smi in cache:
            continue
        result = get_targets_for_smiles(smi)
        cache[smi] = result
        time.sleep(SLEEP)

        if (i + 1) % 25 == 0 or (i + 1) == total:
            save_cache(cache)
            resolved = sum(1 for v in cache.values() if v["resolved"])
            with_targets = sum(1 for v in cache.values() if v["targets"])
            print(f"[{i+1}/{total}] cache={len(cache)} resolved={resolved} with_targets={with_targets}", flush=True)

    save_cache(cache)
    resolved = sum(1 for v in cache.values() if v["resolved"])
    with_targets = sum(1 for v in cache.values() if v["targets"])
    print(f"DONE. total={len(cache)} resolved={resolved} with_targets={with_targets}")


if __name__ == "__main__":
    main()
