#!/bin/bash
# The "strongest model" of the paper: GIN A3+A5 (5-seed ensemble) + RF ECFP4+RDKit,
# averaged 0.5/0.5, on all three split families -- training, evaluation and blend
# in one place, with the same GIN variant (A3+A5) everywhere.
#
#   sh scripts/run/best_blend.sh scaffold      # scaffold-204        -> blend_karma_rf
#   sh scripts/run/best_blend.sh ogb           # OGB/Hu              -> blend_karma_rf_ogb
#   sh scripts/run/best_blend.sh random 0|1|2  # random partition N  -> blend_karma_rf_random[_s1|_s2]
#
# Add TRAIN=0 to skip the GIN training (re-use the existing checkpoints) and
# EVAL_OUT_DIR=<dir> to write the evaluation outputs elsewhere than outputs/eval
# (the blend step then needs EVAL_DIR=<same dir> as well; both are set below).
# Existing results are overwritten only if the tag already exists in the output
# directory. The paper's OGB/Hu blend (blend_2_3_rf_ogb) uses the A3 GIN instead;
# its recipe is in scripts/run/remaining_experiments.sh and docs/reproduction.md.
# Cost: GIN training ~3.5 h per split (5 seeds), RF ~3 min, evaluation and blend ~5 min.
set -e
export PYTHONPATH=.
: "${TRAIN:=1}"
OUT="${EVAL_OUT_DIR:-outputs/eval}"
export EVAL_OUT_DIR="$OUT" EVAL_DIR="$OUT"
mkdir -p "$OUT"
FAM="$1"; N="${2:-0}"
COMMON="ablation.use_rdkit_descriptors=true ablation.use_swa=true"

case "$FAM" in
scaffold)
  GIN=karma_scaffold; RF=classic_rf_ecfp_rdkit; BLEND=blend_karma_rf
  [ "$TRAIN" = 1 ] && python -u -m scripts.train.scaffold204_a3a5 0 1 2 3 4
  python -m scripts.eval.ensemble outputs/ensemble_hetero/ginv2_karma_scaffold_ $GIN \
    ablation.use_rdkit_descriptors=true \
    dataset.rdkit_descriptor_normalizer_path='"data/raw/ginv2_rdkit_descriptor_normalizer.json"'
  python -m scripts.eval.classic rf:ecfp_rdkit
  ;;
ogb)
  GIN=karma_ogb; RF=classic_rf_ecfp_rdkit_ogb; BLEND=blend_karma_rf_ogb
  [ "$TRAIN" = 1 ] && python -u -m scripts.train.ogb_a3a5 0 1 2 3 4   # needs data/raw/..._normalizer_ogb.json (written by train.ogb_a3)
  REF_TAG=$RF python -m scripts.eval.ensemble outputs/ensemble_hetero/ginv2_karma_ogb_ $GIN $COMMON \
    dataset.heldout_split='"ogb_scaffold"' \
    dataset.rdkit_descriptor_normalizer_path='"data/raw/ginv2_rdkit_descriptor_normalizer_ogb.json"'
  HELDOUT_SPLIT=ogb_scaffold python -m scripts.eval.classic rf:ecfp_rdkit
  ;;
random)
  if [ "$N" = 0 ]; then SG=""; SB=""; else SG="_s$N"; SB="_s$N"; fi
  GIN=karma_random$SG; RF=classic_rf_ecfp_rdkit_random_seed$N; BLEND=blend_karma_rf_random$SB
  [ "$TRAIN" = 1 ] && RANDOM_SPLIT_SEED=$N python -u -m scripts.train.random_a3a5 0 1 2 3 4
  PFX=outputs/ensemble_hetero/ginv2_karma_random_; [ "$N" = 0 ] || PFX=${PFX}s${N}_
  REF_TAG=$RF python -m scripts.eval.ensemble $PFX $GIN $COMMON \
    dataset.heldout_split='"random"' dataset.random_split_seed=$N \
    dataset.rdkit_descriptor_normalizer_path="\"data/raw/ginv2_rdkit_descriptor_normalizer_random_seed$N.json\""
  HELDOUT_SPLIT=random RANDOM_SPLIT_SEED=$N python -m scripts.eval.classic rf:ecfp_rdkit
  ;;
*) echo "usage: $0 scaffold | ogb | random 0|1|2"; exit 2;;
esac
python -m scripts.eval.blend $GIN $RF $BLEND
