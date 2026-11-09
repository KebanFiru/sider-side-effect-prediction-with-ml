"""Classical models (random forest, gradient boosting) on the SAME split as the
GIN-v2 ablation: training = the ~1100 SIDER training molecules of
`_split_sider(config, carve_heldout_test=True)`, test = the same 204-molecule
clean test set. The stopping slice is only reported (no setting was chosen on
it or on the test set).

Features:
  ecfp        : Morgan radius 2 (ECFP4), 2048 bits
  ecfp_rdkit  : ECFP4 + RDKit's 217 2D descriptors (same list as
                src/data/featurize.py; raw values since trees are scale
                invariant, NaN/inf -> 0, clipped at |x| = 1e6)
  ecfp_rdkit_aux    : ecfp_rdkit + auxiliary-data stacking (probabilities of
                RFs trained on the 33 Tox21/ClinTox/BBBP/HIV/MUV tasks;
                scripts/analysis/build_rf_aux_stack.py). _auxsmall: Tox21/ClinTox/BBBP
                only (15 tasks)
  ecfp_rdkit_target : ecfp_rdkit + the 101 ChEMBL target features used by
                GIN-v2 (V2: pChEMBL/10 for the 100 most frequent targets +
                has_data flag)

Models (fixed settings close to common literature defaults):
  rf  : RandomForestClassifier, 500 trees, class_weight="balanced"
        (the MoleculeNet RF setting), one multi-output model for all tasks
  hgb : HistGradientBoostingClassifier, one model per task, 200 iterations,
        lr 0.05
  xgb : xgboost.XGBClassifier, one model per task, 500 trees, lr 0.05,
        max_depth 6, subsample 0.8, colsample_bytree 0.5, tree_method="hist";
        no class weights (as hgb). Settings are fixed, not tuned on the test
        set or the stopping slice.

Output has the same format as the GIN-v2 test script:
outputs/eval/ginv2_heldout_ensemble_{tag}.json and _probs.npz, including the
paired bootstrap p-value against the baseline.

Usage: PYTHONPATH=. python -m scripts.eval.classic [model:features ...]
  Without arguments the four basic combinations (rf/hgb x ecfp/ecfp_rdkit) run.
  Example: ... eval_classic_heldout rf:ecfp_rdkit_target
"""
import sys
import json
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from src.data.featurize import _target_features, compute_rdkit_descriptors_raw
from src.eval_metrics import (
    bootstrap_ci,
    macro_auc,
    macro_aupr,
    macro_balanced_accuracy,
    macro_f1,
    macro_report,
    paired_bootstrap_pvalue,
)
from src.splits import load_heldout_test_set
from src.train import load_config
from src.train_ensemble_hetero import _split_sider

RDLogger.DisableLog("rdApp.*")
N_BOOT = 2000
N_JOBS = 4  # leaves CPU for training jobs running in parallel
OUT_DIR = os.environ.get("EVAL_OUT_DIR", "outputs/eval")  # EVAL_OUT_DIR: write elsewhere (reproducibility test)
BASELINE_TAG = "baseline"
# Random split (literature comparison): HELDOUT_SPLIT=random RANDOM_SPLIT_SEED=0..2,
# tag suffix "_random_seed{N}". OGB / Hu et al. scaffold split:
# HELDOUT_SPLIT=ogb_scaffold, tag suffix "_ogb". Outside the scaffold-204 split
# no paired comparison is made, since the scaffold baseline covers other molecules.
HELDOUT_SPLIT = os.environ.get("HELDOUT_SPLIT", "scaffold")
RANDOM_SPLIT_SEED = int(os.environ.get("RANDOM_SPLIT_SEED", 0))
STOPPING_PARADIGM = os.environ.get("STOPPING_PARADIGM", "balanced")  # for HELDOUT_SPLIT=paradigm
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


_AUX_STACK = "outputs/eval/rf_aux_stack_features.npz"  # scripts/analysis/build_rf_aux_stack.py
_AUX_SMALL = ("tox21", "clintox", "bbbp")


def aux_stack_features(smiles_list, sources):
    """Auxiliary-task RF probabilities for SIDER molecules (stacking)."""
    d = np.load(_AUX_STACK)
    cols = np.isin(d["sources"], sources)
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
    table = {canon(s): row[cols] for s, row in zip(d["smiles"], d["feats"])}
    return np.stack([table[canon(s)] for s in smiles_list])


def featurize(smiles_list, with_rdkit, with_target=False, aux=None):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"cannot parse SMILES: {smi}")
        fp = _MORGAN.GetFingerprintAsNumPy(mol).astype(np.float32)
        if with_rdkit:
            desc = np.nan_to_num(np.asarray(compute_rdkit_descriptors_raw(mol), dtype=np.float64),
                                 nan=0.0, posinf=0.0, neginf=0.0)
            fp = np.concatenate([fp, np.clip(desc, -1e6, 1e6).astype(np.float32)])
        if with_target:
            fp = np.concatenate([fp, _target_features(mol)])
        rows.append(fp)
    X = np.stack(rows)
    if aux:
        X = np.concatenate([X, aux_stack_features(smiles_list, aux)], axis=1)
    return X


def fit_predict(model_name, X_tr, y_tr, X_list):
    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=N_JOBS, random_state=0)
        clf.fit(X_tr, y_tr)
        return [np.stack([p[:, 1] for p in clf.predict_proba(X)], axis=1) for X in X_list]
    outs = [np.zeros((X.shape[0], y_tr.shape[1])) for X in X_list]
    for t in range(y_tr.shape[1]):
        if model_name == "xgb":
            clf = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                                colsample_bytree=0.5, tree_method="hist", n_jobs=N_JOBS, random_state=0)
        else:
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
        clf.fit(X_tr, y_tr[:, t])
        for o, X in zip(outs, X_list):
            o[:, t] = clf.predict_proba(X)[:, 1]
    return outs


def report(tag, targets, probs, extra):
    rep = macro_report(targets, probs, threshold=0.5)
    print(f"\n=== [{tag}] held-out test (n={targets.shape[0]} molecules) ===")
    print(f"macro_AUC={rep.macro_auc:.3f}  macro_AUPR={rep.macro_aupr:.3f}  "
          f"balanced_acc={rep.balanced_accuracy:.3f}  F1_macro={rep.f1_macro:.3f}")
    cis = {}
    for name, fn in [("macro_auc", macro_auc), ("macro_aupr", macro_aupr),
                     ("balanced_accuracy", macro_balanced_accuracy), ("f1_macro", macro_f1)]:
        cis[name] = bootstrap_ci(targets, probs, fn, n_boot=N_BOOT)
        print(f"  {name}: {cis[name]['point']:.3f}  95% CI=[{cis[name]['ci_lo']:.3f}, {cis[name]['ci_hi']:.3f}]")
    out = {"tag": tag, "n_molecules": int(targets.shape[0]),
           "n_tasks_total": rep.n_tasks_total, "n_tasks_excluded": rep.n_tasks_excluded,
           "macro_auc": rep.macro_auc, "macro_aupr": rep.macro_aupr,
           "balanced_accuracy": rep.balanced_accuracy, "f1_macro": rep.f1_macro,
           "ci": cis, **extra}
    np.savez_compressed(f"{OUT_DIR}/ginv2_heldout_ensemble_{tag}_probs.npz", probs=probs, targets=targets)
    out["heldout_split"] = HELDOUT_SPLIT
    if HELDOUT_SPLIT != "scaffold":
        if HELDOUT_SPLIT == "random":
            out["random_split_seed"] = RANDOM_SPLIT_SEED
        json.dump(out, open(f"{OUT_DIR}/ginv2_heldout_ensemble_{tag}.json", "w"), indent=2)
        return
    base = np.load(f"{OUT_DIR}/ginv2_heldout_ensemble_{BASELINE_TAG}_probs.npz")
    assert np.array_equal(base["targets"], targets), "targets / molecule order differ from the baseline"
    out["vs_baseline"] = {}
    for name, fn in [("macro_auc", macro_auc), ("macro_aupr", macro_aupr)]:
        paired = paired_bootstrap_pvalue(targets, probs, base["probs"], fn, n_boot=N_BOOT)
        out["vs_baseline"][name] = paired
        print(f"  diff vs baseline [{name}]: {paired['observed_diff']:+.3f}  paired bootstrap p={paired['p_value']:.3f}")
    json.dump(out, open(f"{OUT_DIR}/ginv2_heldout_ensemble_{tag}.json", "w"), indent=2)


def main():
    config = load_config(chembl_feature_version="v2")
    config["dataset"]["heldout_split"] = HELDOUT_SPLIT
    config["dataset"]["random_split_seed"] = RANDOM_SPLIT_SEED
    config["dataset"]["stopping_paradigm"] = STOPPING_PARADIGM
    _, train_dc, stop_dc = _split_sider(config, carve_heldout_test=True)
    test_dc = load_heldout_test_set(config)
    for dc in (train_dc, stop_dc, test_dc):
        assert np.all(np.asarray(dc.w) > 0), "missing labels found; this script assumes complete labels"
    y_tr = np.asarray(train_dc.y).astype(int)
    print(f"training {len(train_dc.ids)}, stopping slice {len(stop_dc.ids)}, clean test {len(test_dc.ids)} molecules")

    os.makedirs(OUT_DIR, exist_ok=True)
    combos = [a.split(":") for a in sys.argv[1:]] or [
        [m, f] for f in ("ecfp", "ecfp_rdkit") for m in ("rf", "hgb")]
    for feat in dict.fromkeys(f for _, f in combos):
        assert feat in ("ecfp", "ecfp_rdkit", "ecfp_rdkit_target",
                        "ecfp_rdkit_aux", "ecfp_rdkit_auxsmall"), feat
        aux = {"ecfp_rdkit_aux": ("tox21", "clintox", "bbbp", "hiv", "muv"),
               "ecfp_rdkit_auxsmall": _AUX_SMALL}.get(feat)
        X_tr, X_stop, X_test = (featurize(list(dc.ids), feat != "ecfp", feat.endswith("_target"), aux)
                                for dc in (train_dc, stop_dc, test_dc))
        for model_name in [m for m, f in combos if f == feat]:
            tag = f"classic_{model_name}_{feat}"
            if HELDOUT_SPLIT == "random":
                tag += f"_random_seed{RANDOM_SPLIT_SEED}"
            elif HELDOUT_SPLIT == "ogb_scaffold":
                tag += "_ogb"
            elif HELDOUT_SPLIT == "paradigm":
                tag += "_parbal" if STOPPING_PARADIGM == "balanced" else "_parseq"
            p_stop, p_test = fit_predict(model_name, X_tr, y_tr, [X_stop, X_test])
            stop_auc = macro_auc(np.asarray(stop_dc.y), p_stop)
            print(f"\n[{tag}] stopping-slice macro-AUC (for information only): {stop_auc:.3f}")
            report(tag, np.asarray(test_dc.y), p_test,
                   {"model": model_name, "features": feat, "n_train": len(train_dc.ids),
                    "stopping_slice_macro_auc": stop_auc})


if __name__ == "__main__":
    main()
