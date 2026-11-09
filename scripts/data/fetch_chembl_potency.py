"""Extension of fetch_chembl_targets.py: for the already resolved ChEMBL IDs
(skipping the molecule lookup step, so it is fast) it also fetches pChEMBL
POTENCY values and saves them to data/raw/chembl_potency.json.

Format: {smiles: {chembl_id: {target_id: max_pchembl_value, ...}, ...}}
Instead of a binary "targets" list, the highest measurement (strongest
binding) per target is kept, so that weak and strong binding can be
distinguished as a feature.
"""
import json
import os
import time
import urllib.request
import urllib.error

CACHE_PATH = "data/raw/chembl_targets.json"
OUT_PATH = "data/raw/chembl_potency.json"
SLEEP = 0.12


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


def load_out():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            return json.load(f)
    return {}


def save_out(d):
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, OUT_PATH)


def get_potency(chembl_id):
    url = (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        f"?molecule_chembl_id={chembl_id}&target_organism=Homo+sapiens"
        "&pchembl_value__isnull=false&limit=100"
    )
    data = fetch_json(url)
    if not data:
        return {}
    per_target = {}
    for a in data.get("activities", []):
        tid = a.get("target_chembl_id")
        pval = a.get("pchembl_value")
        if not tid or pval is None:
            continue
        try:
            pval = float(pval)
        except (TypeError, ValueError):
            continue
        if tid not in per_target or pval > per_target[tid]:
            per_target[tid] = pval
    return per_target


def main():
    with open(CACHE_PATH) as f:
        cache = json.load(f)

    resolved = {smi: v["chembl_id"] for smi, v in cache.items() if v.get("resolved") and v.get("targets")}
    out = load_out()
    total = len(resolved)
    print(f"Molecules to fetch potency for: {total}, already done: {len(out)}")

    for i, (smi, cid) in enumerate(resolved.items()):
        if smi in out:
            continue
        out[smi] = {"chembl_id": cid, "potency": get_potency(cid)}
        time.sleep(SLEEP)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            save_out(out)
            print(f"[{i+1}/{total}] saved", flush=True)

    save_out(out)
    with_potency = sum(1 for v in out.values() if v["potency"])
    print(f"DONE. total={len(out)} with_potency={with_potency}")


if __name__ == "__main__":
    main()
