#!/bin/sh
# Once a GIN-v2 ablation training run finishes, measure the 5-seed ensemble on
# the held-out test. It waits for an EXIT_CODE line after the last "=== RESUME"
# marker in the training log; on success it runs
# scripts.eval.ensemble, otherwise it writes TRAINING_FAILED to
# the log.
#
# Usage (from the repository root):
#   sh scripts/run/eval_after_training.sh <train_log> <ckpt_prefix> <tag> <group_label> [override ...]
# Example (2.2):
#   sh scripts/run/eval_after_training.sh /tmp/scaffold204_a2.log \
#       outputs/ensemble_hetero/ginv2_2_2_asl_ 2_2 2.2_asl ablation.use_asl=true
set -u
TRAIN_LOG=$1
PREFIX=$2
TAG=$3
GROUP=$4
shift 4
OUT=outputs/eval/eval_${TAG}.log

last_run() {
    awk '/=== RESUME/{buf=""} {buf=buf"\n"$0} END{print buf}' "$TRAIN_LOG"
}

while ! last_run | grep -q "EXIT_CODE="; do
    sleep 60
done

if last_run | grep -q "EXIT_CODE=0" && [ -f "${PREFIX}4.pt" ] && last_run | grep -q "Group ${GROUP} finished"; then
    PYTHONPATH=. python -m scripts.eval.ensemble "$PREFIX" "$TAG" "$@" > "$OUT" 2>&1
    echo "EVAL_EXIT=$?" >> "$OUT"
else
    echo "TRAINING_FAILED" > "$OUT"
    tail -30 "$TRAIN_LOG" >> "$OUT"
fi
chown 1000:1000 outputs/eval/* outputs/ensemble_hetero/manifest_*.json
