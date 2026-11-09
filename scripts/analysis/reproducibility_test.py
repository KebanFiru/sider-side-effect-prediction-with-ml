"""Reproducibility test: do the CURRENT code and the stored checkpoints give the
same results as the files the paper was built from?

Nothing in outputs/eval/ is modified. New results go to EVAL_OUT_DIR
(default outputs/eval_repro/); the reference is read from outputs/eval/.

What is checked, per model (model list = every result JSON in outputs/eval/):
  * GIN ensembles   -> re-evaluate the stored checkpoints (scripts.eval.ensemble)
                       with the overrides recorded in the old JSON.
  * classic models  -> RE-TRAIN (scripts.eval.classic). random_state=0 is fixed
                       for RF / XGBoost / HGB and the splits are read from disk,
                       so the result should be (near-)identical.
  * blends          -> recomputed (scripts.eval.blend) from the NEW probabilities
                       of their two components when both were re-run.
For each one: new probs vs old probs (np.allclose), macro-AUC / macro-AUPR
difference, and the per-endpoint AUC (sklearn, 27 endpoints) of old vs new.

Tolerances (PASS/FAIL):
  GIN / blend : probs rtol=1e-3 atol=1e-4 (GPU kernels are not bit-exact),
                metrics |diff| <= 5e-4
  classic     : probs rtol=1e-5 atol=1e-6, metrics |diff| <= 1e-6
  endpoint AUC: |diff| <= 5e-4 (GIN, blend) / 1e-6 (classic)

Tiers: 1 = the models of paper Table A.10 (RF ECFP+RDKit on scaffold-204, OGB/Hu,
random 0/1/2; GIN A3+A5 5-seed on the same splits; GIN A3 on OGB/Hu) and the
blends built only from them, incl. the new blend_karma_rf_ogb; 2 = everything else.

Usage (repository root, PYTHONPATH=.):
  python -m scripts.analysis.reproducibility_test --dry-run          # list + estimate
  python -m scripts.analysis.reproducibility_test --tier 1           # Table A.10 only
  python -m scripts.analysis.reproducibility_test --tier 2           # everything
  python -m scripts.analysis.reproducibility_test --only karma_ogb classic_rf_ecfp_rdkit_ogb
  options: --force (re-run models already in EVAL_OUT_DIR), --compare-only
           (no re-run; compare whatever is already in EVAL_OUT_DIR)
Outputs in EVAL_OUT_DIR: repro_report.csv (model level), repro_endpoint_auc.csv
(split, model, endpoint, eski_auc, yeni_auc, fark; includes random_mean3 rows),
repro_summary.json, repro_logs/<tag>.log, repro_timing.csv.
A missing checkpoint / reference file gives SKIP, a crashed run ERROR, a model
without an old result (blend_karma_rf_ogb) NEW; the test
continues either way.
"""
import argparse
import csv
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

OLD = "outputs/eval"
NEW = os.environ.get("EVAL_OUT_DIR", "outputs/eval_repro")
P = "ginv2_heldout_ensemble_"

# 2_3_ogb (A3, 5 seeds) is the GIN component of the paper's OGB/Hu blend blend_2_3_rf_ogb.
TIER1_GIN = ["karma_scaffold", "karma_ogb", "karma_random", "karma_random_s1", "karma_random_s2", "2_3_ogb"]
TIER1_RF = ["classic_rf_ecfp_rdkit", "classic_rf_ecfp_rdkit_ogb"] + \
           [f"classic_rf_ecfp_rdkit_random_seed{i}" for i in range(3)]
TIER1 = set(TIER1_GIN + TIER1_RF)
# Blends of Table A.10's "strongest model" column: scaffold-204 blend_karma_rf, OGB/Hu
# blend_2_3_rf_ogb (A3 GIN), random blend_karma_rf_random[_s1|_s2] -- all tier 1 because
# their components are. NEW_BLENDS do not exist in outputs/eval: they are produced for
# the first time (no old value to compare with, status NEW). blend_karma_rf_ogb puts
# the same GIN variant (A3+A5) on the OGB/Hu split as on the other two families.
NEW_BLENDS = {"blend_karma_rf_ogb": ("karma_ogb", "classic_rf_ecfp_rdkit_ogb")}

TOL = {"gin": dict(rtol=1e-3, atol=1e-4, metric=5e-4, ep=5e-4),
       "blend": dict(rtol=1e-3, atol=1e-4, metric=5e-4, ep=5e-4),
       "classic": dict(rtol=1e-5, atol=1e-6, metric=1e-6, ep=1e-6)}
# rough minutes per model, from the timestamps of the original runs (GPU for GIN)
MINUTES = {"gin": 3.0, "classic_rf": 3.0, "classic_xgb": 5.0, "classic_hgb": 5.0, "blend": 0.5}


def old_path(tag, ext):
    return f"{OLD}/{P}{tag}.{ext}"


def new_path(tag, ext):
    return f"{NEW}/{P}{tag}.{ext}"


def kind_of(meta):
    if "ckpt_prefix" in meta:
        return "gin"
    if "gnn_tag" in meta:
        return "blend"
    if "model" in meta:
        return "classic"
    return None


def split_label(meta):
    s, seed = meta.get("heldout_split", "scaffold"), meta.get("random_split_seed", 0)
    for o in meta.get("config_overrides", []):  # GIN JSONs record the split only here
        if o.startswith("dataset.heldout_split="):
            s = json.loads(o.split("=", 1)[1])
        elif o.startswith("dataset.random_split_seed="):
            seed = int(o.split("=", 1)[1])
    if s == "random":
        return f"random{seed}"
    return {"scaffold": "scaffold204", "ogb_scaffold": "ogb"}.get(s, s)


def load_models():
    models = {}
    for f in sorted(glob.glob(f"{OLD}/{P}*.json")):
        meta = json.load(open(f))
        tag = meta.get("tag")
        k = kind_of(meta)
        if k and tag and os.path.exists(old_path(tag, "npz").replace(".npz", "_probs.npz")):
            models[tag] = dict(meta=meta, kind=k)
    for tag, (g, r) in NEW_BLENDS.items():
        if g in models:
            models[tag] = dict(meta=dict(tag=tag, gnn_tag=g, rf_tag=r, heldout_split=models[g]["meta"].get("heldout_split", "scaffold"),
                                         config_overrides=models[g]["meta"].get("config_overrides", [])),
                               kind="blend", new_only=True)
    for tag, m in models.items():
        meta = m["meta"]
        if m["kind"] == "blend":
            deps = [meta["gnn_tag"], meta["rf_tag"]]
            m["deps"] = deps
            m["tier"] = 1 if all(d in TIER1 for d in deps) else 2
        else:
            m["tier"] = 1 if tag in TIER1 else 2
        m["split"] = split_label(meta) if m["kind"] != "blend" else split_label(models.get(meta["gnn_tag"], m)["meta"]
                                                                                 if meta["gnn_tag"] in models else meta)
    return models


def est_minutes(tag, m):
    if m["kind"] == "gin":
        return MINUTES["gin"] * m["meta"].get("n_seeds", 5) / 5 + 1.5
    if m["kind"] == "blend":
        return MINUTES["blend"]
    return MINUTES.get("classic_" + m["meta"]["model"], 4.0)


def run(cmd, env_extra, log_path):
    env = dict(os.environ, PYTHONPATH=".", EVAL_OUT_DIR=NEW, EVAL_DIR=NEW, **env_extra)
    t0 = time.time()
    with open(log_path, "w") as log:
        r = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0


def run_gin(tag, m):
    meta = m["meta"]
    n = int(meta.get("n_seeds", 5))
    idx = meta.get("seed_indices") or list(range(n))
    missing = [f"{meta['ckpt_prefix']}{i}.pt" for i in idx if not os.path.exists(f"{meta['ckpt_prefix']}{i}.pt")]
    if missing:
        return "SKIP", f"missing checkpoint: {missing[0]}", 0.0
    # paired-comparison reference exactly as in the original run; a tag that does not
    # exist means "no pairing" (otherwise the scaffold baseline copied into EVAL_OUT_DIR
    # for the classic runs would be compared against other splits' molecules)
    ref = (meta.get("vs_baseline") or {}).get("reference_tag", "__no_reference__")
    env = {"N_SEEDS": str(n), "REF_TAG": ref}
    if idx != list(range(n)):
        env["SEED_INDICES"] = ",".join(map(str, idx))
    cmd = [sys.executable, "-u", "-m", "scripts.eval.ensemble", meta["ckpt_prefix"], tag] + list(meta.get("config_overrides", []))
    rc, dt = run(cmd, env, f"{NEW}/repro_logs/{tag}.log")
    return ("OK" if rc == 0 else "ERROR"), ("" if rc == 0 else f"exit {rc}, see repro_logs/{tag}.log"), dt


def run_classic_group(tags, models):
    """One process per (split, seed): featurization is shared between models."""
    groups = {}
    for t in tags:
        meta = models[t]["meta"]
        key = (meta.get("heldout_split", "scaffold"), meta.get("random_split_seed", 0))
        groups.setdefault(key, []).append(t)
    res = {}
    for (split, seed), ts in groups.items():
        combos = [f"{models[t]['meta']['model']}:{models[t]['meta']['features']}" for t in ts]
        env = {"HELDOUT_SPLIT": split, "RANDOM_SPLIT_SEED": str(seed)}
        cmd = [sys.executable, "-u", "-m", "scripts.eval.classic"] + combos
        rc, dt = run(cmd, env, f"{NEW}/repro_logs/classic_{split}_{seed}.log")
        for t in ts:
            res[t] = ("OK" if rc == 0 else "ERROR", "" if rc == 0 else f"exit {rc}", dt / len(ts))
    return res


def run_blend(tag, m):
    meta = m["meta"]
    for d in m["deps"]:
        if not os.path.exists(new_path(d, "json")):
            return "SKIP", f"component {d} not re-run", 0.0
    rc, dt = run([sys.executable, "-u", "-m", "scripts.eval.blend", meta["gnn_tag"], meta["rf_tag"], tag], {},
                 f"{NEW}/repro_logs/{tag}.log")
    return ("OK" if rc == 0 else "ERROR"), ("" if rc == 0 else f"exit {rc}"), dt


def endpoint_names():
    with gzip.open("data/raw/sider.csv.gz", "rt") as f:
        return next(csv.reader(f))[1:]


def per_endpoint_auc(y, p):
    out = []
    for k in range(y.shape[1]):
        yy = y[:, k]
        out.append(np.nan if yy.min() == yy.max() else roc_auc_score(yy, p[:, k]))
    return np.array(out)


def compare(tag, m, names, ep_rows, ep_store):
    kind = m["kind"]
    tol = TOL[kind]
    if not os.path.exists(new_path(tag, "json")) or not os.path.exists(new_path(tag, "npz").replace(".npz", "_probs.npz")):
        return None
    if m.get("new_only"):  # first-time product: no old file, report the endpoint AUCs only
        n = np.load(new_path(tag, "npz").replace(".npz", "_probs.npz"))
        nm = json.load(open(new_path(tag, "json")))
        en = per_endpoint_auc(n["targets"], n["probs"])
        for k, nm_ in enumerate(names):
            ep_rows.append([m["split"], tag, nm_, "", en[k], ""])
        ep_store[tag] = (np.full(len(names), np.nan), en)
        return dict(model=tag, kind="blend", split=m["split"], tier=m["tier"], status="NEW",
                    macro_auc_new=nm["macro_auc"], macro_aupr_new=nm["macro_aupr"],
                    note="no reference in outputs/eval; produced by this test")
    o = np.load(old_path(tag, "npz").replace(".npz", "_probs.npz"))
    n = np.load(new_path(tag, "npz").replace(".npz", "_probs.npz"))
    om, nm = m["meta"], json.load(open(new_path(tag, "json")))
    same_targets = o["targets"].shape == n["targets"].shape and np.array_equal(o["targets"], n["targets"])
    res = dict(model=tag, kind=kind, split=m["split"], tier=m["tier"], same_targets=same_targets)
    if not same_targets:
        res.update(status="FAIL", note="targets / molecule order differ")
        return res
    res["probs_max_abs_diff"] = float(np.max(np.abs(o["probs"] - n["probs"])))
    res["probs_allclose"] = bool(np.allclose(o["probs"], n["probs"], rtol=tol["rtol"], atol=tol["atol"]))
    ok = res["probs_allclose"]
    for met in ("macro_auc", "macro_aupr"):
        d = nm[met] - om[met]
        res[f"{met}_old"], res[f"{met}_new"], res[f"{met}_diff"] = om[met], nm[met], d
        ok &= abs(d) <= tol["metric"]
    eo, en = per_endpoint_auc(o["targets"], o["probs"]), per_endpoint_auc(n["targets"], n["probs"])
    diff = en - eo
    finite = ~np.isnan(diff)
    res["endpoint_auc_max_abs_diff"] = float(np.max(np.abs(diff[finite]))) if finite.any() else 0.0
    ok &= (np.isnan(eo) == np.isnan(en)).all() and res["endpoint_auc_max_abs_diff"] <= tol["ep"]
    res["tol"] = f"probs rtol={tol['rtol']} atol={tol['atol']}; metric {tol['metric']}; endpoint {tol['ep']}"
    res["status"] = "PASS" if ok else "FAIL"
    for k, nm_ in enumerate(names):
        ep_rows.append([m["split"], tag, nm_, eo[k], en[k], diff[k]])
    ep_store[tag] = (eo, en)
    return res


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[1, 2], default=1, help="1 = Table A.10 models, 2 = all")
    ap.add_argument("--only", nargs="*", help="only these tags")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    assert os.path.abspath(NEW) != os.path.abspath(OLD), "EVAL_OUT_DIR must differ from outputs/eval"
    models = load_models()
    sel = [t for t, m in models.items() if (a.only and t in a.only) or (not a.only and m["tier"] <= a.tier)]
    order = {"gin": 0, "classic": 1, "blend": 2}
    sel.sort(key=lambda t: (models[t]["tier"], order[models[t]["kind"]], t))

    if a.dry_run:
        tot = 0.0
        print(f"{'tier':4} {'kind':8} {'split':12} {'min':>5}  tag")
        for t in sel:
            mi = est_minutes(t, models[t]); tot += mi
            print(f"{models[t]['tier']:4} {models[t]['kind']:8} {models[t]['split']:12} {mi:5.1f}  {t}")
        print(f"\n{len(sel)} models, estimated {tot:.0f} min (~{tot/60:.1f} h); tier-1 only: "
              f"{sum(est_minutes(t, m) for t, m in models.items() if m['tier'] == 1):.0f} min")
        return

    os.makedirs(f"{NEW}/repro_logs", exist_ok=True)
    status, timing = {}, []
    if not a.compare_only:
        # the scaffold-204 classic runs compare against the baseline probs in EVAL_OUT_DIR
        bp = new_path("baseline", "npz").replace(".npz", "_probs.npz")
        if not os.path.exists(bp) and os.path.exists(old_path("baseline", "npz").replace(".npz", "_probs.npz")):
            shutil.copy(old_path("baseline", "npz").replace(".npz", "_probs.npz"), bp)  # copy: old file untouched

        def todo(t):
            return a.force or not os.path.exists(new_path(t, "json"))
        for t in [t for t in sel if models[t]["kind"] == "gin"]:
            if todo(t):
                status[t] = run_gin(t, models[t]); timing.append((t, status[t][2]))
        ct = [t for t in sel if models[t]["kind"] == "classic" and todo(t)]
        if ct:
            for t, r in run_classic_group(ct, models).items():
                status[t] = r; timing.append((t, r[2]))
        for t in [t for t in sel if models[t]["kind"] == "blend"]:
            if todo(t):
                status[t] = run_blend(t, models[t]); timing.append((t, status[t][2]))
        write_csv(f"{NEW}/repro_timing.csv", ["model", "seconds"], timing)

    names = endpoint_names()
    ep_rows, ep_store, report = [], {}, []
    for t in sel:
        m = models[t]
        r = compare(t, m, names, ep_rows, ep_store)
        if r is None:
            st = status.get(t, ("SKIP", "not re-run (no output in EVAL_OUT_DIR)", 0.0))
            r = dict(model=t, kind=m["kind"], split=m["split"], tier=m["tier"], status="ERROR" if st[0] == "ERROR" else "SKIP", note=st[1])
        elif t in status and status[t][0] != "OK":
            r["note"] = status[t][1]
        report.append(r)

    # random: mean of the three partitions, as in Table A.10
    def mean3(tags, label):
        if all(t in ep_store for t in tags):
            eo = np.nanmean([ep_store[t][0] for t in tags], axis=0)
            en = np.nanmean([ep_store[t][1] for t in tags], axis=0)
            for k, nm_ in enumerate(names):
                ep_rows.append(["random_mean3", label, nm_, eo[k], en[k], en[k] - eo[k]])
            return float(np.nanmax(np.abs(en - eo)))
    summary = {"random_mean3_max_endpoint_diff": {
        "GIN A3+A5": mean3(["karma_random", "karma_random_s1", "karma_random_s2"], "karma_random(mean3)"),
        "RF ECFP+RDKit": mean3([f"classic_rf_ecfp_rdkit_random_seed{i}" for i in range(3)], "classic_rf_ecfp_rdkit(mean3)"),
        "GIN+RF blend": mean3(["blend_karma_rf_random", "blend_karma_rf_random_s1", "blend_karma_rf_random_s2"],
                              "blend_karma_rf_random(mean3)")}}

    cols = ["tier", "split", "model", "kind", "status", "macro_auc_old", "macro_auc_new", "macro_auc_diff",
            "macro_aupr_old", "macro_aupr_new", "macro_aupr_diff", "probs_max_abs_diff",
            "endpoint_auc_max_abs_diff", "tol", "note"]
    write_csv(f"{NEW}/repro_report.csv", cols, [[r.get(c, "") for c in cols] for r in report])
    write_csv(f"{NEW}/repro_endpoint_auc.csv", ["split", "model", "endpoint", "eski_auc", "yeni_auc", "fark"], ep_rows)
    summary["counts"] = {s: sum(r["status"] == s for r in report) for s in ("PASS", "FAIL", "SKIP", "ERROR", "NEW")}
    summary["tier1_all_pass"] = all(r["status"] in ("PASS", "NEW") for r in report if r["tier"] == 1)
    json.dump(summary, open(f"{NEW}/repro_summary.json", "w"), indent=2)

    print(f"\n{'status':6} {'split':12} {'dAUC':>9} {'dAUPR':>9} {'max|dP|':>9} {'max|dEP|':>9}  model")
    for r in report:
        f = lambda k: f"{r[k]:9.2e}" if k in r and r[k] != "" else f"{'-':>9}"
        print(f"{r['status']:6} {r['split']:12} {f('macro_auc_diff')} {f('macro_aupr_diff')} {f('probs_max_abs_diff')} "
              f"{f('endpoint_auc_max_abs_diff')}  {r['model']} {r.get('note', '')}")
    print("\n", json.dumps(summary, indent=2))
    print(f"report: {NEW}/repro_report.csv, {NEW}/repro_endpoint_auc.csv")


if __name__ == "__main__":
    main()
