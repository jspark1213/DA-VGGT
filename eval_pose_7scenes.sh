#!/bin/bash
# 7-Scenes camera-pose evaluation
# Method: random_ls_revsim + pose-weighted re-chunking (mode E, gamma=0.001, eps=0.005)
set -e
cd "$(dirname "$0")"   # repo root

DATASET_DIR=/workspace/dataset/7scenes
OUTPUT_DIR=./results/pose_7scenes
export CUDA_VISIBLE_DEVICES=0

python eval/eval_chunked_pose_7scenes.py \
    --dataset_dir "$DATASET_DIR" \
    --n_frames 500 \
    --chunk_size 50 \
    --sampling_method random_ls_revsim \
    --local_search_iters 5 \
    --poseweight_mode pseudo \
    --rechunk_remaining_only \
    --output_dir "$OUTPUT_DIR"
