"""Probability sets for the conformal / calibration study. No model is trained
except the Random Forests refitted here (random_state fixed).

For each split (scaffold-204 = "scaffold", OGB/Hu = "ogb_scaffold") this writes
outputs/conformal/<part>_<split>.npz:

  rf     (--part rf)
    oof_ids, oof_y, oof_p, oof_fold : out-of-fold RF (ECFP4+RDKit) probabilities for
        the TRAINING molecules from a 5-fold scaffold-grouped CV (src.train.assign_folds,
        one repeat). No model-selection decision touched these molecules' labels as
        calibration data: this is the calibration set of the primary estimator.
    stop_ids, stop_y, stop_p : RF fitted on the full training set, scored on the
        early-stopping slice (the RF never saw it; it is used for a like-for-like
        comparison with the GIN, whose epoch was SELECTED on this slice).
    test_ids, test_y, test_p : the held-out test molecules (same RF as above;
        test_p_max_abs_diff_vs_eval = largest deviation from the stored
        outputs/eval/*_probs.npz of the paper).
  gin    (--part gin)
    stop_*  : GIN A3+A5 five-seed ensemble on the early-stopping slice, per seed
        (stop_p_seeds, shape [5, n, 27]) and mean (stop_p). WARNING: the stopping
        slice drove early stopping and epoch selection of these networks, so
        these probabilities are optimistically biased for calibration.
    test_*  : the same on the test molecules (test_p_max_abs_diff_vs_eval checks
        the mean against the paper's stored ensemble probabilities).
  yrand  (--part yrand)
    RF fitted on label-PERMUTED training data (10 permutations, rng 12345 as in
    paper_chemometric_analyses.run_yrand), scored on the stopping slice and the
    test set; true labels are kept for stop/test.

Usage (repository root, PYTHONPATH=.; GPU for --part gin):
  python -m scripts.analysis.conformal_probs --part rf|gin|yrand [--splits scaffold ogb_scaffold]
"""
import argparse
import json
import os
import time

import numpy as np

OUT = "outputs/conformal"
SPLITS = {"scaffold": dict(gin_tag="karma_scaffold", rf_tag="classic_rf_ecfp_rdkit"),
          "ogb_scaffold": dict(gin_tag="karma_ogb", rf_tag="classic_rf_ecfp_rdkit_ogb")}
N_FOLDS = 5
N_YRAND = 10
YRAND_SEED = 12345
P = "outputs/eval/ginv2_heldout_ensemble_"


def load_split_sets(split):
    from src.splits import load_heldout_test_set
    from src.train import load_config
    from src.train_ensemble_hetero import _split_sider
    config = load_config(chembl_feature_version="v2")
    config["dataset"]["heldout_split"] = split
    config["dataset"]["random_split_seed"] = 0
    config["dataset"]["stopping_paradigm"] = "balanced"
    _, train_dc, stop_dc = _split_sider(config, carve_heldout_test=True)
    test_dc = load_heldout_test_set(config)
    f = lambda d: (list(d.ids), np.asarray(d.y).astype(int))
    return f(train_dc), f(stop_dc), f(test_dc)


def run_rf(split):
    import scripts.eval.classic as ec
    from src.train import assign_folds
    ec.N_JOBS = 12
    (tr_ids, y_tr), (st_ids, y_st), (te_ids, y_te) = load_split_sets(split)
    X_tr, X_st, X_te = (ec.featurize(ids, True) for ids in (tr_ids, st_ids, te_ids))
    folds = assign_folds(tr_ids, N_FOLDS)
    oof = np.zeros(y_tr.shape)
    for k in range(N_FOLDS):
        t0 = time.time()
        te, tr = folds == k, folds != k
        oof[te] = ec.fit_predict("rf", X_tr[tr], y_tr[tr], [X_tr[te]])[0]
        print(f"[{split}] OOF fold {k}: {te.sum()} held-out molecules, {time.time() - t0:.0f}s", flush=True)
    p_st, p_te = ec.fit_predict("rf", X_tr, y_tr, [X_st, X_te])
    ref = np.load(f"{P}{SPLITS[split]['rf_tag']}_probs.npz")
    assert np.array_equal(ref["targets"], y_te), "test molecule order differs from the paper's"
    diff = float(np.max(np.abs(ref["probs"] - p_te)))
    print(f"[{split}] refit test probs vs paper's stored probs: max abs diff {diff:.2e}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/rf_{split}.npz", oof_ids=np.array(tr_ids), oof_y=y_tr, oof_p=oof, oof_fold=folds,
                        stop_ids=np.array(st_ids), stop_y=y_st, stop_p=p_st,
                        test_ids=np.array(te_ids), test_y=y_te, test_p=p_te, test_p_paper=ref["probs"],
                        test_p_max_abs_diff_vs_eval=diff)
    print(f"[{split}] calibration set sizes: OOF n={len(tr_ids)}, stopping slice n={len(st_ids)}, test n={len(te_ids)}", flush=True)


def run_gin(split):
    import torch
    from torch_geometric.loader import DataLoader
    from scripts.eval.ensemble import _apply_overrides
    from src.data.featurize import MoleculeGraphDataset, configure_rdkit_descriptors
    from src.splits import load_heldout_test_set
    from src.train import build_model, load_config
    from src.train_ensemble_hetero import _split_sider

    tag = SPLITS[split]["gin_tag"]
    meta = json.load(open(f"{P}{tag}.json"))
    config = load_config(chembl_feature_version="v2")
    _apply_overrides(config, meta.get("config_overrides", []))
    config["dataset"]["stopping_paradigm"] = "balanced"
    use_rdkit = config["ablation"].get("use_rdkit_descriptors", False)
    configure_rdkit_descriptors(use_rdkit, config["dataset"].get("rdkit_descriptor_normalizer_path") if use_rdkit else None)
    _, _, stop_dc = _split_sider(config, carve_heldout_test=True)
    test_dc = load_heldout_test_set(config)
    ds_st = MoleculeGraphDataset(stop_dc, cache_dir=config["dataset"]["processed_root"])
    ds_te = MoleculeGraphDataset(test_dc, cache_dir=config["dataset"]["processed_multitask_root"])
    assert len(ds_st) == len(stop_dc.ids) and len(ds_te) == len(test_dc.ids), "molecules dropped during featurization"
    bs = config["train"]["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed = {"stop": [], "test": []}
    for i in range(int(meta["n_seeds"])):
        model = build_model(json.loads(json.dumps(config)), 60, device)
        model.load_state_dict(torch.load(f"{meta['ckpt_prefix']}{i}.pt", map_location=device))
        model.eval()
        for key, ds in (("stop", ds_st), ("test", ds_te)):
            out = []
            with torch.no_grad():
                for b in DataLoader(ds, batch_size=bs):
                    b = b.to(device)
                    out.append(torch.sigmoid(model(b.x, b.edge_index, b.edge_attr, b.batch, b.global_features))[:, :27].cpu())
            per_seed[key].append(torch.cat(out).numpy())
        print(f"[{split}] GIN seed {i} scored", flush=True)
    ref = np.load(f"{P}{tag}_probs.npz")
    y_te, y_st = np.asarray(test_dc.y).astype(int), np.asarray(stop_dc.y).astype(int)
    assert np.array_equal(ref["targets"], y_te)
    ps, pt = np.stack(per_seed["stop"]), np.stack(per_seed["test"])
    diff = float(np.max(np.abs(ref["probs"] - pt.mean(0))))
    print(f"[{split}] GIN test ensemble vs paper's stored probs: max abs diff {diff:.2e}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/gin_{split}.npz", stop_ids=np.array(list(stop_dc.ids)), stop_y=y_st, stop_p=ps.mean(0),
                        stop_p_seeds=ps, test_ids=np.array(list(test_dc.ids)), test_y=y_te, test_p=pt.mean(0),
                        test_p_seeds=pt, test_p_max_abs_diff_vs_eval=diff,
                        note=np.array("stopping slice was used for early stopping / epoch selection: probabilities are biased"))


def run_yrand(split):
    import scripts.eval.classic as ec
    ec.N_JOBS = 12
    (tr_ids, y_tr), (st_ids, y_st), (te_ids, y_te) = load_split_sets(split)
    X_tr, X_st, X_te = (ec.featurize(ids, True) for ids in (tr_ids, st_ids, te_ids))
    rng = np.random.default_rng(YRAND_SEED)
    ps, pt = [], []
    for k in range(N_YRAND):
        t0 = time.time()
        p_st, p_te = ec.fit_predict("rf", X_tr, y_tr[rng.permutation(len(y_tr))], [X_st, X_te])
        ps.append(p_st); pt.append(p_te)
        print(f"[{split}] Y-rand permutation {k}: {time.time() - t0:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/yrand_{split}.npz", stop_ids=np.array(st_ids), stop_y=y_st, stop_p=np.stack(ps),
                        test_ids=np.array(te_ids), test_y=y_te, test_p=np.stack(pt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["rf", "gin", "yrand"])
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    a = ap.parse_args()
    for s in a.splits:
        {"rf": run_rf, "gin": run_gin, "yrand": run_yrand}[a.part](s)
