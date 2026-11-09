"""Additional analyses for the paper. All numbers are written to a single JSON,
outputs/eval/paper_chemometrics.json, which the paper's number-generation
tooling reads (numbers are never copied by hand).

Sections (selectable via argv; default all):
  seeds   : RF and XGBoost on the OGB/Hu split, random_state 0..9
            (ecfp, ecfp_rdkit). Same statistic as Hu et al.: mean +- population
            std over seeds (ddof=0).
  yrand   : Y-randomization. The RF (ecfp_rdkit) is retrained 10 times with the
            ROWS of the training label matrix shuffled (preserving the joint
            distribution across tasks); AUC on the scaffold-204 and OGB/Hu tests.
  ad      : Applicability domain. Nearest-neighbour Tanimoto similarity (ECFP4)
            of each test molecule to the training set; the test set is split
            into three equal-sized similarity bins, macro-AUC per bin.
  srd     : Sum of Ranking Differences on the per-task AUC matrix
            (scaffold-204, 11 models x 27 tasks; reference: row mean), with
            the random-ranking distribution (CRRN); Friedman test with the
            Nemenyi critical difference, and Wilcoxon signed-rank tests for
            selected pairs.

Usage :
  PYTHONPATH=. python -m scripts.analysis.paper_chemometric_analyses [seeds yrand ad srd]
"""
import json
import os
import sys

import numpy as np
from rdkit import DataStructs
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon
from sklearn.metrics import roc_auc_score

from scripts.eval.classic import _MORGAN, featurize, fit_predict
from src.splits import load_heldout_test_set
from src.train import load_config
from src.train_ensemble_hetero import _split_sider
from rdkit import Chem

OUT_DIR = "outputs/eval"
OUT_PATH = f"{OUT_DIR}/paper_chemometrics.json"
N_SEEDS = 10
N_YRAND = 10

# models of the scaffold-204 ablation table (tag -> name in the paper)
SRD_MODELS = {
    "baseline": "GIN baseline", "2_1": "(A1) capacity", "2_2": "(A2) ASL",
    "2_3": "(A3) RDKit", "2_5": "(A4) uncertainty", "2_6": "(A5) SWA",
    "karma_scaffold": "(A3+A5)", "classic_rf_ecfp": "RF ECFP4",
    "classic_rf_ecfp_rdkit": "RF ECFP4+RDKit", "classic_xgb_ecfp_rdkit": "XGB ECFP4+RDKit",
    "blend_2_3_rf": "GIN(A3)+RF blend",
}
WILCOXON_PAIRS = [("2_3", "baseline"), ("2_6", "baseline"), ("2_1", "baseline"),
                  ("karma_scaffold", "classic_rf_ecfp_rdkit"), ("blend_2_3_rf", "classic_rf_ecfp_rdkit")]
AD_MODELS = {"scaffold": ["2_3", "classic_rf_ecfp_rdkit", "blend_2_3_rf"],
             "ogb_scaffold": ["2_3_ogb", "classic_rf_ecfp_rdkit_ogb", "blend_2_3_rf_ogb"]}


def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, t], p[:, t]) for t in range(y.shape[1]) if 0 < y[:, t].sum() < len(y)]
    return float(np.mean(aucs))


def per_task_auc(y, p):
    return np.array([roc_auc_score(y[:, t], p[:, t]) if 0 < y[:, t].sum() < len(y) else np.nan
                     for t in range(y.shape[1])])


def load_split(split):
    config = load_config(chembl_feature_version="v2")
    config["dataset"]["heldout_split"] = split
    config["dataset"]["random_split_seed"] = 0
    config["dataset"]["stopping_paradigm"] = "balanced"
    _, train_dc, _ = _split_sider(config, carve_heldout_test=True)
    test_dc = load_heldout_test_set(config)
    return (list(train_dc.ids), np.asarray(train_dc.y).astype(int),
            list(test_dc.ids), np.asarray(test_dc.y).astype(int))


def fit_predict_seeded(model_name, X_tr, y_tr, X_te, seed):
    # same settings as eval_classic_heldout; only random_state changes
    import scripts.eval.classic as ech
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=ech.N_JOBS,
                                     random_state=seed).fit(X_tr, y_tr)
        return np.stack([p[:, 1] for p in clf.predict_proba(X_te)], axis=1)
    out = np.zeros((X_te.shape[0], y_tr.shape[1]))
    for t in range(y_tr.shape[1]):
        clf = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.5, tree_method="hist", n_jobs=ech.N_JOBS, random_state=seed)
        out[:, t] = clf.fit(X_tr, y_tr[:, t]).predict_proba(X_te)[:, 1]
    return out


def run_seeds(res):
    tr_ids, y_tr, te_ids, y_te = load_split("ogb_scaffold")
    for feat in ("ecfp", "ecfp_rdkit"):
        X_tr, X_te = featurize(tr_ids, feat != "ecfp"), featurize(te_ids, feat != "ecfp")
        for m in ("rf", "xgb"):
            aucs = [macro_auc(y_te, fit_predict_seeded(m, X_tr, y_tr, X_te, s)) for s in range(N_SEEDS)]
            key = f"ogb_{m}_{feat}"
            res[key] = {"per_seed": aucs, "mean": float(np.mean(aucs)), "std_ddof0": float(np.std(aucs))}
            print(key, f"{np.mean(aucs):.3f} +- {np.std(aucs):.3f}", flush=True)


def run_yrand(res):
    rng = np.random.default_rng(12345)
    res["yrand"] = {}
    for split in ("scaffold", "ogb_scaffold"):
        tr_ids, y_tr, te_ids, y_te = load_split(split)
        X_tr, X_te = featurize(tr_ids, True), featurize(te_ids, True)
        true_auc = macro_auc(y_te, fit_predict("rf", X_tr, y_tr, [X_te])[0])
        perm_aucs = []
        for _ in range(N_YRAND):
            y_perm = y_tr[rng.permutation(len(y_tr))]
            perm_aucs.append(macro_auc(y_te, fit_predict("rf", X_tr, y_perm, [X_te])[0]))
        res["yrand"][split] = {"true_auc": true_auc, "perm_aucs": perm_aucs,
                               "perm_mean": float(np.mean(perm_aucs)), "perm_std": float(np.std(perm_aucs)),
                               "perm_max": float(np.max(perm_aucs))}
        print("yrand", split, f"true={true_auc:.3f} perm={np.mean(perm_aucs):.3f}+-{np.std(perm_aucs):.3f}"
              f" max={np.max(perm_aucs):.3f}", flush=True)


def nn_similarity(tr_ids, te_ids):
    fps_tr = [_MORGAN.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr_ids]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(_MORGAN.GetFingerprint(Chem.MolFromSmiles(s)), fps_tr))
                     for s in te_ids])


def run_ad(res):
    res["ad"] = {}
    for split, tags in AD_MODELS.items():
        tr_ids, _, te_ids, y_te = load_split(split)
        sim = nn_similarity(tr_ids, te_ids)
        edges = np.quantile(sim, [1 / 3, 2 / 3])
        bins = np.digitize(sim, edges)  # 0 = least similar, 2 = most similar
        entry = {"nn_sim_mean": float(sim.mean()), "edges": edges.tolist(),
                 "bin_n": [int((bins == b).sum()) for b in range(3)],
                 "bin_sim_mean": [float(sim[bins == b].mean()) for b in range(3)], "models": {}}
        for tag in tags:
            z = np.load(f"{OUT_DIR}/ginv2_heldout_ensemble_{tag}_probs.npz")
            assert np.array_equal(z["targets"].astype(int), y_te), f"{tag}: molecule order differs"
            p = z["probs"]
            entry["models"][tag] = {"all": macro_auc(y_te, p),
                                    "bins": [macro_auc(y_te[bins == b], p[bins == b]) for b in range(3)]}
        res["ad"][split] = entry
        print("ad", split, json.dumps(entry, indent=1), flush=True)


def srd(matrix, ref):
    """matrix: (n_obj, n_methods). Raw SRD against the reference ranking, and the maximum SRD."""
    n = matrix.shape[0]
    r_ref = rankdata(ref)
    raw = np.array([np.abs(rankdata(matrix[:, j]) - r_ref).sum() for j in range(matrix.shape[1])])
    # maximum SRD: the ranking fully reversed
    srd_max = int(np.abs(np.arange(1, n + 1) - np.arange(n, 0, -1)).sum())
    return raw, srd_max


def run_srd(res):
    tags = list(SRD_MODELS)
    zs = {t: np.load(f"{OUT_DIR}/ginv2_heldout_ensemble_{t}_probs.npz") for t in tags}
    y = zs[tags[0]]["targets"].astype(int)
    for t in tags:
        assert np.array_equal(zs[t]["targets"].astype(int), y), t
    A = np.stack([per_task_auc(y, zs[t]["probs"]) for t in tags], axis=1)  # (27, 11)
    A = A[~np.isnan(A).any(axis=1)]
    ref = A.mean(axis=1)
    raw, srd_max = srd(A, ref)
    rng = np.random.default_rng(0)
    n = A.shape[0]
    r_ref = rankdata(ref)
    rand = np.array([np.abs(rng.permutation(n) + 1 - r_ref).sum() for _ in range(100000)])
    q05 = float(np.quantile(rand, 0.05))
    p_vs_random = [float((rand <= v).mean()) for v in raw]
    # stability of the SRD ranking: bootstrap over tasks
    boot_rank = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        b_raw, _ = srd(A[idx], A[idx].mean(axis=1))
        boot_rank.append(rankdata(b_raw))
    boot_rank = np.array(boot_rank)
    fr = friedmanchisquare(*[A[:, j] for j in range(A.shape[1])])
    mean_rank = rankdata(-A, axis=1).mean(axis=0)  # 1 = best
    k = A.shape[1]  # Nemenyi critical difference (Demsar 2006), alpha = 0.05
    nemenyi_cd = float(studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n)))
    wil = {}
    for a, b in WILCOXON_PAIRS:
        ja, jb = tags.index(a), tags.index(b)
        w = wilcoxon(A[:, ja], A[:, jb])
        wil[f"{a}_vs_{b}"] = {"median_diff": float(np.median(A[:, ja] - A[:, jb])),
                              "n_better": int((A[:, ja] > A[:, jb]).sum()), "n_tasks": int(n),
                              "statistic": float(w.statistic), "p_value": float(w.pvalue)}
    res["srd"] = {
        "n_tasks": int(n), "models": {t: {"name": SRD_MODELS[t], "srd_raw": int(raw[j]),
                                          "srd_pct": float(100 * raw[j] / srd_max),
                                          "p_vs_random": p_vs_random[j],
                                          "boot_rank_median": float(np.median(boot_rank[:, j])),
                                          "boot_rank_q05": float(np.quantile(boot_rank[:, j], 0.05)),
                                          "boot_rank_q95": float(np.quantile(boot_rank[:, j], 0.95)),
                                          "friedman_mean_rank": float(mean_rank[j]),
                                          "mean_task_auc": float(A[:, j].mean())}
                                      for j, t in enumerate(tags)},
        "srd_max": srd_max, "random_q05_pct": 100 * q05 / srd_max,
        "random_mean_pct": float(100 * rand.mean() / srd_max),
        "friedman": {"statistic": float(fr.statistic), "p_value": float(fr.pvalue), "k": len(tags),
                     "nemenyi_cd": nemenyi_cd},
        "wilcoxon": wil, "per_task_auc": A.tolist(), "tags": tags,
    }
    print(json.dumps({k: v for k, v in res["srd"].items() if k != "per_task_auc"}, indent=1), flush=True)


def main():
    parts = sys.argv[1:] or ["srd", "ad", "yrand", "seeds"]
    res = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {}
    for part in parts:
        {"seeds": run_seeds, "yrand": run_yrand, "ad": run_ad, "srd": run_srd}[part](res)
        json.dump(res, open(OUT_PATH, "w"), indent=1)  # save after every section


if __name__ == "__main__":
    main()
