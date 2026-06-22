# DA-VGGT

Official repository for **Diversity-Aware View Partitioning for Scalable VGGT**.

DA-VGGT is a training-free, plug-and-play inference framework for VGGT that
organizes input views into diversity-aware balanced subsets, constructed via
combinatorial graph partitioning over visual dissimilarity and spatial
dispersion. This lets the transformer focus attention on geometrically
informative views while reducing redundant attention, improving camera pose
estimation, multi-view depth prediction, and 3D reconstruction at lower memory
and latency. This repository provides the code to reproduce the main
evaluations on 7-Scenes (pose, 3D) and Bonn (depth).

## News

- **2026-06-17** — Accepted to **ECCV 2026** 🎉
- **2026-06-22** — Code released.

## Layout

```
eval_pose_7scenes.sh             # run: 7-Scenes camera-pose evaluation
eval_3d_7scenes.sh               # run: 7-Scenes 3D-reconstruction evaluation
eval_depth_bonn.sh               # run: Bonn RGB-D depth evaluation
eval/
  eval_chunked_pose_7scenes.py   # 7-Scenes camera-pose evaluation
  eval_chunked_3d_7scenes.py     # 7-Scenes 3D-reconstruction evaluation
  eval_chunked_depth_bonn.py     # Bonn RGB-D depth evaluation
requirements.txt                 # python dependencies
vggt/                            # VGGT model package (incl. our aggregator splits)
```

## Setup

```bash
pip install -r requirements.txt
```

The VGGT-1B checkpoint is fetched automatically from Hugging Face on first run
(`facebook/VGGT-1B`).

## Run

One launcher script per evaluation lives at the repo root. Edit `DATASET_DIR` /
`CUDA_VISIBLE_DEVICES` at the top of each, then run:

```bash
bash eval_pose_7scenes.sh    # 7-Scenes camera pose
bash eval_3d_7scenes.sh      # 7-Scenes 3D reconstruction
bash eval_depth_bonn.sh      # Bonn RGB-D depth
```

Each script invokes the matching `eval/eval_chunked_*.py` with the main method
`--sampling_method random_ls_revsim` and pose-weighted re-chunking
(`--poseweight_mode pseudo`, `--combine_mode E`, `--gamma 0.001`,
`--epsilon 0.005`, `--rechunk_remaining_only`).

To run a script directly (e.g. for a custom config):

```bash
python eval/eval_chunked_pose_7scenes.py \
    --dataset_dir /path/to/7scenes \
    --n_frames 500 --chunk_size 50 \
    --sampling_method random_ls_revsim --local_search_iters 5 \
    --poseweight_mode pseudo --rechunk_remaining_only \
    --output_dir ./results/pose_7scenes
```

`--sampling_method origin` (single-batch, no chunking) is available as a baseline
in every script. See `python eval/<script>.py --help` for the full option list.

## Acknowledgements

This work is built on top of [VGGT](https://github.com/facebookresearch/vggt),
whose model and pretrained weights (`facebook/VGGT-1B`) we use directly. Our
chunked evaluation harness and the evaluation-metric code (pose AUC, 3D
accuracy/completion, and depth metrics) are adapted from
[FastVGGT](https://github.com/mystorm16/FastVGGT)
([arXiv:2509.02560](https://arxiv.org/abs/2509.02560)). We thank the authors of
both projects for releasing their code.
