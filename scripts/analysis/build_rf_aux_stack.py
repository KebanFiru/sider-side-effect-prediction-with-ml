"""Auxiliary data (Tox21, ClinTox, BBBP, HIV, MUV) for the RF: STACKING features.

The GNN uses auxiliary data through multi-task training (shared encoder, extra
outputs). A sklearn RF has no direct equivalent: auxiliary molecules have no
SIDER labels, and Tox21/MUV also have missing labels. The classical
equivalent is stacking:

  1. For each auxiliary task (12+2+1+1+17 = 33 in total) a separate RF is
     trained on that task's labelled molecules only (ECFP4, 2048 bits).
  2. The probabilities these RFs predict for SIDER molecules are given to the
     SIDER RF as extra columns (scripts/eval/classic.py, features
     "ecfp_rdkit_aux" and "ecfp_rdkit_auxsmall").

Leakage / distribution-shift guard: ALL SIDER molecules (training + stopping +
test, ~1427) are removed from the auxiliary training sets (matched on the
InChIKey connectivity block of the largest fragment, i.e. ignoring salts and
stereo). Otherwise the probabilities would be in-sample (sharp) for SIDER
training molecules that also appear in an auxiliary set and out-of-sample
(blurred) for test molecules. This also makes the auxiliary models
split-independent: they are trained once and every split (scaffold-204,
OGB/Hu, random 0-2) uses the same matrix.

NOTE: the GNN's auxiliary-data protocol differs -- build_combined_train_set in
src/train.py uses ALL molecules of the auxiliary sets (including those shared
with SIDER; not leakage, since the labels are not SIDER labels, but less
strict than here).

Output: outputs/eval/rf_aux_stack_features.npz
  smiles (as written in the SIDER csv), feats (n x 33), task_names, sources

Usage: PYTHONPATH=. python -m scripts.analysis.build_rf_aux_stack
"""
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import RandomForestClassifier

from src.train import AUX_LOADERS, load_config

RDLogger.DisableLog("rdApp.*")
AUX_NAMES = ["tox21", "clintox", "bbbp", "hiv", "muv"]
N_TREES = 300
N_JOBS = 4  # leaves CPU for a parallel GNN training job
OUT = "outputs/eval/rf_aux_stack_features.npz"
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_LARGEST = rdMolStandardize.LargestFragmentChooser()


def mol_key(mol):
    """Molecule key that ignores salts and stereochemistry."""
    return Chem.MolToInchiKey(_LARGEST.choose(mol))[:14]


def ecfp(mols):
    return np.stack([_MORGAN.GetFingerprintAsNumPy(m).astype(np.uint8) for m in mols])


def main():
    config = load_config(chembl_feature_version="v2")
    root = config["dataset"]["root"]

    import pandas as pd
    sider_smiles = list(pd.read_csv(os.path.join(root, "sider.csv.gz"))["smiles"])
    sider_mols = [Chem.MolFromSmiles(s) for s in sider_smiles]
    keep = [i for i, m in enumerate(sider_mols) if m is not None]
    sider_smiles = [sider_smiles[i] for i in keep]
    sider_mols = [sider_mols[i] for i in keep]
    sider_keys = {mol_key(m) for m in sider_mols}
    X_sider = ecfp(sider_mols)
    print(f"SIDER: {len(sider_smiles)} molecules, {len(sider_keys)} unique keys")

    feats, names, sources = [], [], []
    for name in AUX_NAMES:
        tasks, datasets, _ = AUX_LOADERS[name](featurizer="Raw", splitter="scaffold",
                                                data_dir=root, save_dir=root)
        ids = [s for d in datasets for s in d.ids]
        y = np.concatenate([d.y for d in datasets])
        w = np.concatenate([d.w for d in datasets])
        mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else s for s in ids]
        ok = np.array([m is not None and mol_key(m) not in sider_keys for m in mols])
        n_overlap = sum(1 for m in mols if m is not None and mol_key(m) in sider_keys)
        mols = [m for m, k in zip(mols, ok) if k]
        y, w = y[ok], w[ok]
        X = ecfp(mols)
        print(f"{name}: {len(tasks)} tasks, {len(ids)} molecules, removed {n_overlap} shared with SIDER")
        for t, task in enumerate(tasks):
            lab = w[:, t] > 0
            y_t = y[lab, t].astype(int)
            clf = RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced",
                                         n_jobs=N_JOBS, random_state=0)
            clf.fit(X[lab], y_t)
            feats.append(clf.predict_proba(X_sider)[:, 1])
            names.append(f"{name}:{task}")
            sources.append(name)
            print(f"  {task}: {lab.sum()} labelled, {y_t.sum()} positive")
            del clf

    np.savez_compressed(OUT, smiles=np.array(sider_smiles), feats=np.stack(feats, axis=1).astype(np.float32),
                        task_names=np.array(names), sources=np.array(sources))
    print(f"wrote: {OUT}  ({len(sider_smiles)} x {len(names)})")


if __name__ == "__main__":
    main()
