#!/usr/bin/env bash
# Usage:
#   DOMAIN=math bash scripts/train.sh               # DMax-Math-16B, seed 0
#   DOMAIN=code SEED=3 bash scripts/train.sh        # DMax-Coder-16B, seed 3 (the paper uses 0-4)
#   DOMAIN=math bash scripts/train.sh optimizer.lr=1e-6   # override any key of src/configs/base.yaml
#
# Output: outputs/dmax_<domain>_mmd/seed<SEED>/ with final_model/ (the checkpoint), training.jsonl
# and resolved_config.yaml. Evaluate the checkpoint with
#   DOMAIN=math MODEL_PATH=outputs/dmax_math_mmd/seed0/final_model bash scripts/eval.sh


set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DOMAIN=${DOMAIN:-math}
SEED=${SEED:-0}
NUM_GPUS=${NUM_GPUS:-8}
case "$DOMAIN" in
  math|code) ;;
  *) echo 'DOMAIN must be math or code' >&2; exit 2 ;;
esac
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dmax_${DOMAIN}_mmd/seed$SEED}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"  # vendor/ is imported from the repository root
export TOKENIZERS_PARALLELISM=false

exec torchrun --standalone --nproc_per_node="$NUM_GPUS" src/main.py \
  config="src/configs/dmax_${DOMAIN}_mmd.yaml" \
  training.seed="$SEED" \
  output_dir="$OUTPUT_DIR" \
  "$@"
