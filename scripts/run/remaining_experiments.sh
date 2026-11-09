#!/bin/sh
# Runs the remaining experiments of the paper IN ORDER
# (from the repository root). Safe to rerun after an interruption: training
# skips the seeds already in the manifest (and resumes a half-finished seed from
# its .resume file); evaluations simply rerun (cheap).
#
#   sh scripts/run/remaining_experiments.sh
#
# Progress: outputs/eval/remaining_experiments.log     (step by step),
#           outputs/eval/remaining_experiments.status  (last step started)
set -u
cd .
export PYTHONPATH=.

LOG=outputs/eval/remaining_experiments.log
STATUS=outputs/eval/remaining_experiments.status
mkdir -p outputs/eval

say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; echo "$*" > "$STATUS"; }
fix_perms() { chown -R 1000:1000 outputs 2>/dev/null || true; }

say "STARTED"

# ------------------------------------------------------------ 1) OGB baseline
say "1/4 baseline_ogb training (5 seeds)"
python -u -m scripts.train.ogb_baseline 0 1 2 3 4 >> "$LOG" 2>&1 || { say "ERROR baseline_ogb training"; exit 1; }
say "1/4 baseline_ogb evaluation"
REF_TAG=classic_rf_ecfp_rdkit_ogb python -m scripts.eval.ensemble \
  outputs/ensemble_hetero/ginv2_baseline_ogb_ baseline_ogb \
  dataset.heldout_split='"ogb_scaffold"' >> "$LOG" 2>&1 || { say "ERROR baseline_ogb eval"; exit 1; }
fix_perms

# ------------------------------------------------------------ 2) OGB combined
say "2/4 karma_ogb training (5 seeds)"
python -u -m scripts.train.ogb_a3a5 0 1 2 3 4 >> "$LOG" 2>&1 || { say "ERROR karma_ogb training"; exit 1; }
say "2/4 karma_ogb evaluation"
REF_TAG=baseline_ogb python -m scripts.eval.ensemble \
  outputs/ensemble_hetero/ginv2_karma_ogb_ karma_ogb \
  ablation.use_rdkit_descriptors=true ablation.use_swa=true \
  dataset.heldout_split='"ogb_scaffold"' \
  dataset.rdkit_descriptor_normalizer_path='"data/raw/ginv2_rdkit_descriptor_normalizer_ogb.json"' \
  >> "$LOG" 2>&1 || { say "ERROR karma_ogb eval"; exit 1; }
fix_perms

# --------------------------------------------------- 3) OGB 2.3, seeds 1005-1009
# The existing 5-seed outputs (2_3_ogb, blend_2_3_rf_ogb) are kept; the 10-seed
# results are written under separate tags.
say "3/4 2_3_ogb seeds 1005-1009 training"
python -u -m scripts.train.ogb_a3 5 6 7 8 9 >> "$LOG" 2>&1 || { say "ERROR 2_3_ogb training"; exit 1; }
for I in 5 6 7 8 9; do
  say "3/4 2_3_ogb single-seed evaluation (index $I)"
  SEED_INDICES=$I REF_TAG=classic_rf_ecfp_rdkit_ogb python -m scripts.eval.ensemble \
    outputs/ensemble_hetero/ginv2_2_3_ogb_ "2_3_ogb_seed$((1000+I))" \
    ablation.use_rdkit_descriptors=true dataset.heldout_split='"ogb_scaffold"' \
    dataset.rdkit_descriptor_normalizer_path='"data/raw/ginv2_rdkit_descriptor_normalizer_ogb.json"' \
    >> "$LOG" 2>&1 || { say "ERROR 2_3_ogb eval index $I"; exit 1; }
done
say "3/4 2_3_ogb 10-seed ensemble"
N_SEEDS=10 REF_TAG=classic_rf_ecfp_rdkit_ogb python -m scripts.eval.ensemble \
  outputs/ensemble_hetero/ginv2_2_3_ogb_ 2_3_ogb_10seed \
  ablation.use_rdkit_descriptors=true dataset.heldout_split='"ogb_scaffold"' \
  dataset.rdkit_descriptor_normalizer_path='"data/raw/ginv2_rdkit_descriptor_normalizer_ogb.json"' \
  >> "$LOG" 2>&1 || { say "ERROR 2_3_ogb 10-seed eval"; exit 1; }
python -m scripts.eval.blend 2_3_ogb_10seed classic_rf_ecfp_rdkit_ogb blend_2_3_rf_ogb_10seed \
  >> "$LOG" 2>&1 || { say "ERROR blend 10 seeds"; exit 1; }
fix_perms

# ------------------------------------------------ 4) Random split, splits 1 and 2
for N in 1 2; do
  say "4/4 karma_random split $N training (5 seeds)"
  RANDOM_SPLIT_SEED=$N python -u -m scripts.train.random_a3a5 0 1 2 3 4 \
    >> "$LOG" 2>&1 || { say "ERROR karma_random s$N training"; exit 1; }
  say "4/4 karma_random split $N evaluation"
  REF_TAG="classic_rf_ecfp_rdkit_random_seed${N}" python -m scripts.eval.ensemble \
    "outputs/ensemble_hetero/ginv2_karma_random_s${N}_" "karma_random_s${N}" \
    ablation.use_rdkit_descriptors=true ablation.use_swa=true \
    dataset.heldout_split='"random"' "dataset.random_split_seed=${N}" \
    "dataset.rdkit_descriptor_normalizer_path=\"data/raw/ginv2_rdkit_descriptor_normalizer_random_seed${N}.json\"" \
    >> "$LOG" 2>&1 || { say "ERROR karma_random s$N eval"; exit 1; }
  python -m scripts.eval.blend "karma_random_s${N}" \
    "classic_rf_ecfp_rdkit_random_seed${N}" "blend_karma_rf_random_s${N}" \
    >> "$LOG" 2>&1 || { say "ERROR karma_random s$N blend"; exit 1; }
  fix_perms
done

fix_perms
say "FINISHED"
