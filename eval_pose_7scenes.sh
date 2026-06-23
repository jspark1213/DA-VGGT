#!/bin/bash
# 7-Scenes camera-pose evaluation
# Method: da_partitioning (diversity-aware) + pose-weighted re-chunking (gamma=0.001, eps=0.005)
set -e
cd "$(dirname "$0")"   # repo root

DATASET_DIR=/path/to/7scenes
OUTPUT_DIR=./results/pose_7scenes
export CUDA_VISIBLE_DEVICES=0

python eval/eval_chunked_pose_7scenes.py \
    --dataset_dir "$DATASET_DIR" \
    --n_frames 500 \
    --chunk_size 50 \
    --sampling_method da_partitioning \
    --local_search_iters 5 \
    --rechunk_remaining_only \
    --output_dir "$OUTPUT_DIR"
