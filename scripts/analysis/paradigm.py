"""Paradigm experiment analysis: on the same test molecules (scaffold-204 test +
OGB/Hu test = 328), compares models trained with our balanced scaffold
paradigm (parbal) and with the literature's sequential scaffold paradigm
(parseq).

For each model pair (e.g. 2_3_parbal vs 2_3_parseq):
  - subsets: full test (328), scaffold-204 test only, OGB/Hu test only
  - molecule-size tertiles (heavy-atom count): small / medium / large
  - macro-AUC in each, and the paired bootstrap p (parbal - parseq)
Also describes the training and stopping pools of both paradigms (size,
fraction of singleton scaffolds).

Usage: PYTHONPATH=. python -m scripts.analysis.paradigm <tag_stem> [...]
  e.g. ... analyze_paradigm 2_3 classic_rf_ecfp_rdkit
  (tag_stem + "_parbal" / "_parseq" -> outputs/eval/ginv2_heldout_ensemble_{...}_probs.npz)
Output: outputs/eval/paradigm_analysis.json
"""
import json
import sys
from collections import Counter

import numpy as np
from rdkit import Chem

from src.eval_metrics import macro_auc, paired_bootstrap_pvalue
from src.splits import HELDOUT_TEST_PATH, load_heldout_test_set, ogb_split_masks
from src.train import _scaffold, load_config
from src.train_ensemble_hetero import _split_sider

OUT = "outputs/eval/paradigm_analysis.json"


def _config(paradigm):
    c = load_config(chembl_feature_version="v2")
    c["dataset"]["heldout_split"] = "paradigm"
    c["dataset"]["stopping_paradigm"] = paradigm
    return c


def _heavy(ids):
    return np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in ids])


def describe_pools():
    out = {}
    for par in ("balanced", "sequential"):
        _, tr, st = _split_sider(_config(par), carve_heldout_test=True)
        counts = Counter(_scaffold(s) for s in list(tr.ids) + list(st.ids))
        part = {}
        for name, ids in (("training", list(tr.ids)), ("stopping", list(st.ids))):
            ha = _heavy(ids)
            part[name] = {"n": len(ids), "heavy_atoms_mean": float(ha.mean()), "heavy_atoms_median": float(np.median(ha)),
                          "singleton_scaffold_fraction": float(np.mean([counts[_scaffold(s)] == 1 for s in ids]))}
        out[par] = part
    return out


def main():
    stems = sys.argv[1:]
    test = load_heldout_test_set(_config("balanced"))
    ids = list(test.ids)
    ours = set(json.load(open(HELDOUT_TEST_PATH)))
    ogb_test = set(np.array(ids)[ogb_split_masks(ids)[2]])
    ha = _heavy(ids)
    t1, t2 = np.quantile(ha, [1 / 3, 2 / 3])
    subsets = {
        "full test (328)": np.ones(len(ids), dtype=bool),
        "scaffold-204 test": np.array([s in ours for s in ids]),
        "OGB/Hu test": np.array([s in ogb_test for s in ids]),
        f"small molecules (<= {t1:.0f} heavy atoms)": ha <= t1,
        f"medium ({t1:.0f}-{t2:.0f})": (ha > t1) & (ha <= t2),
        f"large molecules (> {t2:.0f})": ha > t2,
    }
    result = {"pools": describe_pools(), "subset_sizes": {k: int(m.sum()) for k, m in subsets.items()},
              "models": {}}
    for stem in stems:
        a = np.load(f"outputs/eval/ginv2_heldout_ensemble_{stem}_parbal_probs.npz")
        b = np.load(f"outputs/eval/ginv2_heldout_ensemble_{stem}_parseq_probs.npz")
        assert np.array_equal(a["targets"], b["targets"]) and len(a["targets"]) == len(ids)
        rows = {}
        for name, m in subsets.items():
            y, pa, pb = a["targets"][m], a["probs"][m], b["probs"][m]
            pv = paired_bootstrap_pvalue(y, pa, pb, macro_auc, n_boot=2000)
            rows[name] = {"n": int(m.sum()), "balanced (ours)": macro_auc(y, pa), "sequential (literature)": macro_auc(y, pb),
                          "diff": pv["observed_diff"], "p": pv["p_value"]}
            print(f"{stem:28s} {name:34s} n={m.sum():3d}  balanced {rows[name]['balanced (ours)']:.3f}  "
                  f"sequential {rows[name]['sequential (literature)']:.3f}  diff {pv['observed_diff']:+.3f}  p={pv['p_value']:.3f}")
        result["models"][stem] = rows
    json.dump(result, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(result["pools"], indent=1, ensure_ascii=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
