#!/usr/bin/env bash
# Paper setting: 500 frames, 50-frame DA partitions, reuse chunk 0.
set -euo pipefail
cd "$(dirname "$0")"

DATASET_DIR="${DATASET_DIR:-/path/to/scannet}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/3d_scannet}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON" eval/eval_chunked_3d_scannet.py \
    --dataset_dir "$DATASET_DIR" \
    --n_frames 500 --chunk_size 50 \
    --sampling_method da_partitioning --local_search_iters 5 \
    --rechunk_remaining_only \
    --output_dir "$OUTPUT_DIR" "$@"
