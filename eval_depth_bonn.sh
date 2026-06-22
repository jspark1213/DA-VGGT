#!/bin/bash
# Bonn RGB-D depth evaluation
# Method: random_ls_revsim + pose-weighted re-chunking (mode E, gamma=0.001, eps=0.005)
set -e
cd "$(dirname "$0")"   # repo root

DATASET_DIR=/workspace/dataset/bonn
OUTPUT_DIR=./results/depth_bonn
export CUDA_VISIBLE_DEVICES=0

python eval/eval_chunked_depth_bonn.py \
    --dataset_dir "$DATASET_DIR" \
    --n_frames 500 \
    --chunk_size 50 \
    --sampling_method random_ls_revsim \
    --local_search_iters 5 \
    --poseweight_mode pseudo \
    --rechunk_remaining_only \
    --output_dir "$OUTPUT_DIR"
