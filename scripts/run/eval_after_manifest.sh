#!/bin/sh
# Watcher for parallel training: waits until all training processes of the
# group have exited, then scores the 5-seed ensemble on the clean test set if
# the manifest holds 5 seeds. (eval_after_training.sh watches a single training
# log; with two processes sharing the seeds it could start the test while a
# seed still in training had only a partial checkpoint.)
#
# Usage (from the repository root):
#   sh scripts/run/eval_after_manifest.sh <manifest> <training_module> <ckpt_prefix> <tag> [override ...]
set -u
MANIFEST=$1
MODULE=$2
PREFIX=$3
TAG=$4
shift 4
OUT=outputs/eval/eval_${TAG}.log

n_done() {
    python -c "import json,os,sys; p=sys.argv[1]; print(len(json.load(open(p))) if os.path.exists(p) else 0)" "$MANIFEST"
}

while pgrep -f "python -u -m $MODULE" > /dev/null; do
    sleep 60
done

if [ "$(n_done)" -ge 5 ]; then
    PYTHONPATH=. python -m scripts.eval.ensemble "$PREFIX" "$TAG" "$@" > "$OUT" 2>&1
    echo "EVAL_EXIT=$?" >> "$OUT"
else
    echo "TRAINING_FAILED ($(n_done)/5 seeds in the manifest)" > "$OUT"
fi
chown 1000:1000 outputs/eval/* outputs/ensemble_hetero/manifest_*.json
