#!/usr/bin/env bash
# Launcher for ELF post-training and evaluation. Run from the repository root.
#
# Single GPU / CPU:
#     bash scripts/launch.sh train src/configs/mmd/owt_t5.yml
#     bash scripts/launch.sh eval  src/configs/mmd/owt_t5.yml --checkpoint_path outputs/owt_t5-mmd/checkpoint_10000
#
# Multi-GPU (single host); pick the GPUs with CUDA_VISIBLE_DEVICES:
#     CUDA_VISIBLE_DEVICES=0,1 NGPU=2 bash scripts/launch.sh train src/configs/mmd/tinygsm_gpt2.yml
#
# Multi-host (torchrun rendezvous):
#     NGPU=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=node-0 MASTER_PORT=29500 \
#         bash scripts/launch.sh train src/configs/mmd/owt_gpt2.yml
#
# Extra arguments are passed through, e.g. --config_override lr=1e-4.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: bash scripts/launch.sh <train|eval> <config.yml> [extra args...]"
    exit 1
fi

MODE=$1
CONFIG=$2
shift 2

case "$MODE" in
    train) ENTRY=src/train.py ;;
    eval)  ENTRY=src/eval.py ;;
    *) echo "Unknown mode: $MODE (expected 'train' or 'eval')"; exit 1 ;;
esac

NGPU=${NGPU:-1}
NNODES=${NNODES:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if [[ "$NGPU" == "1" && "$NNODES" == "1" ]]; then
    echo "[launch] single-process: python $ENTRY --config $CONFIG $*"
    exec python "$ENTRY" --config "$CONFIG" "$@"
elif [[ "$NNODES" == "1" ]]; then
    # --standalone picks a free port, so several runs can share one host.
    echo "[launch] torchrun --standalone nproc_per_node=$NGPU $ENTRY"
    exec torchrun --standalone --nproc_per_node="$NGPU" "$ENTRY" --config "$CONFIG" "$@"
else
    echo "[launch] torchrun nproc_per_node=$NGPU nnodes=$NNODES node_rank=${NODE_RANK:-0} $ENTRY"
    exec torchrun \
        --nproc_per_node="$NGPU" \
        --nnodes="$NNODES" \
        --node_rank="${NODE_RANK:-0}" \
        --master_addr="${MASTER_ADDR:-127.0.0.1}" \
        --master_port="${MASTER_PORT:-29500}" \
        "$ENTRY" --config "$CONFIG" "$@"
fi
