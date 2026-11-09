#!/bin/bash
# RF + auxiliary-data stacking: all splits, then the comparison.
set -e
export PYTHONPATH=.
C="rf:ecfp_rdkit_aux rf:ecfp_rdkit_auxsmall"
[ -f outputs/eval/rf_aux_stack_features.npz ] || python -u -m scripts.analysis.build_rf_aux_stack
python -u -m scripts.eval.classic $C
HELDOUT_SPLIT=ogb_scaffold python -u -m scripts.eval.classic $C
for s in 0 1 2; do HELDOUT_SPLIT=random RANDOM_SPLIT_SEED=$s python -u -m scripts.eval.classic $C; done
python -u -m scripts.analysis.compare_rf_aux
