#!/usr/bin/env bash
# Usage:
#   DOMAIN=math bash scripts/eval.sh               # released yresearch/DMax-Math-MMD, seed 0
#   DOMAIN=code SEED=3 bash scripts/eval.sh        # released yresearch/DMax-Coder-MMD, seed 3
#   DOMAIN=math MODEL_PATH=outputs/dmax_math_mmd/seed0/final_model bash scripts/eval.sh
#
# Optional: THRESHOLDS='0.75 0.9' (default 0.85 for math, 0.9 for code), TASKS='gsm8k_llada_mini',
# NUM_GPUS (tensor-parallel size, 2 as in the paper), CUDA_VISIBLE_DEVICES, OUTPUT_DIR.
# DMAX_DIR: where https://github.com/czg1225/DMax at commit 82bc29e is (or will be) cloned;
# default ./DMax, cloned automatically on the first run.
#
# Results go to OUTPUT_DIR/threshold_<t>/<task>/, by default outputs/eval/<domain>/<model>/seed<SEED>,
# where <model> is the HF repo name (DMax-Math-MMD) or the training run (dmax_math_mmd_seed0).
# At the end the script prints the results table, averaged over every seed evaluated so far for
# this model (scripts/aggregate.py).


set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
DOMAIN=${DOMAIN:-math}
SEED=${SEED:-0}
NUM_GPUS=${NUM_GPUS:-2}
DMAX_DIR=${DMAX_DIR:-$ROOT/DMax}
# The evaluator itself: DMax at the commit used in the paper, cloned on first use.
if [[ ! -d $DMAX_DIR/dInfer/evaluations ]]; then
  echo "=== Cloning DMax (commit 82bc29e) into $DMAX_DIR"
  git clone --quiet https://github.com/czg1225/DMax.git "$DMAX_DIR"
  git -C "$DMAX_DIR" checkout --quiet 82bc29e
fi

case "$DOMAIN" in
  math)
    RELEASED=yresearch/DMax-Math-MMD
    TASKS=${TASKS:-gsm8k_llada_mini minerva_math500 minerva_math_algebra asdiv_llada_mini}
    THRESHOLDS=${THRESHOLDS:-0.85} ;;
  code)
    RELEASED=yresearch/DMax-Coder-MMD
    TASKS=${TASKS:-humaneval_instruct mbpp_sanitized_llada_mini}
    THRESHOLDS=${THRESHOLDS:-0.9} ;;
  *) echo 'DOMAIN must be math or code' >&2; exit 2 ;;
esac

MODEL_PATH=${MODEL_PATH:-$RELEASED}
# Name results after the model: the HF repo name, or <experiment>_<seed> for a training run
# (outputs/dmax_math_mmd/seed0/final_model -> dmax_math_mmd_seed0).
MODEL_NAME=$(basename "$MODEL_PATH")
if [[ $MODEL_NAME == final_model ]]; then
  run=$(dirname "$MODEL_PATH")
  MODEL_NAME=$(basename "$(dirname "$run")")_$(basename "$run")
fi
if [[ -n ${OUTPUT_DIR:-} ]]; then
  OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR"); SUMMARY_RUNS=$OUTPUT_DIR
else
  OUTPUT_DIR=$ROOT/outputs/eval/$DOMAIN/$MODEL_NAME/seed$SEED; SUMMARY_RUNS="$(dirname "$OUTPUT_DIR")/seed*"
fi
# The evaluator reads weights from a local directory: download Hugging Face models first.
if [[ ! -d $MODEL_PATH ]]; then
  if [[ ! $MODEL_PATH =~ ^[^/]+/[^/]+$ ]]; then
    echo "Checkpoint not found: $MODEL_PATH" >&2; exit 2
  fi
  MODEL_PATH=$("$PYTHON" -c 'import sys; from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))' "$MODEL_PATH")
fi
MODEL_PATH=$(realpath "$MODEL_PATH")

DINFER=$(realpath "$DMAX_DIR/dInfer")
export PYTHONPATH="$DINFER/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 TRANSFORMERS_TRUST_REMOTE_CODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
cd "$DINFER/evaluations"

for threshold in $THRESHOLDS; do
  for task in $TASKS; do
    # Math accuracy comes from the original DMax answer checkers; code uses lm-eval pass@1.
    case "$task" in
      gsm8k_llada_mini) checker=val_gsm8k.py ;;
      minerva_math500) checker=val_math.py ;;
      minerva_math_algebra) checker=val_algebra.py ;;
      asdiv_llada_mini) checker=val_asdiv.py ;;
      humaneval_instruct|mbpp_sanitized_llada_mini) checker= ;;
      *) echo "Unsupported task: $task" >&2; exit 2 ;;
    esac
    out="$OUTPUT_DIR/threshold_$threshold/$task"
    mkdir -p "$out"
    # The evaluator scores whatever rank_0.jsonl it finds; never leave a stale one behind.
    rm -f "$out"/rank_*.jsonl "$out/results.txt"
    # A free port per evaluation, so concurrent jobs on one node do not collide.
    port=$("$PYTHON" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    model_args="model_path=$MODEL_PATH,gen_length=2048,block_length=32,threshold=$threshold"
    model_args+=",parallel_decoding=threshold,cache=prefix,use_bd=True,model_type=llada2,parallel=tp"
    model_args+=",gpus=$(seq -s ';' 0 $((NUM_GPUS - 1))),master_port=$port,save_dir=$out"

    echo "=== $task, threshold $threshold -> $out"
    "$PYTHON" -u eval_dinfer_sglang.py --tasks "$task" --model dInfer_eval --model_args "$model_args" \
      --batch_size 1 --apply_chat_template --confirm_run_unsafe_code --include_path tasks \
      --seed "$SEED" --output_path "$out" 2>&1 | tee "$out/eval.log"
    if [[ -n $checker ]]; then
      "$PYTHON" "$checker" --pred-path "$out/rank_0.jsonl" 2>&1 | tee "$out/postprocess.log"
    fi
  done
done

echo "=== Results"
"$PYTHON" "$ROOT/scripts/aggregate.py" $SUMMARY_RUNS
