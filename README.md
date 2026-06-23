<h1 align="center">Diversity-Aware View Partitioning for Scalable VGGT</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://JSP-ywu.github.io/DA-VGGT/"><img src="https://img.shields.io/badge/Project_Page-DA--VGGT-2088c1" alt="Project Page"></a>
</p>

<p align="center">
  Jinsoo Park<sup>1</sup>,
  Donggyu Choi<sup>2</sup>,
  <a href="https://ahyunseo.github.io/">Ahyun Seo</a><sup>3</sup>,
  <a href="https://cvlab.postech.ac.kr/~mcho/">Minsu Cho</a><sup>1,4</sup>,
  <a href="https://jeanyson.github.io/">Jeany Son</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>POSTECH&nbsp;&nbsp; <sup>2</sup>GIST&nbsp;&nbsp; <sup>3</sup>KAIST&nbsp;&nbsp; <sup>4</sup>RLWRLD
</p>

DA-VGGT is a training-free, plug-and-play inference framework for VGGT that
organizes input views into diversity-aware balanced subsets, constructed via
combinatorial graph partitioning over visual dissimilarity and spatial
dispersion. This lets the transformer focus attention on geometrically
informative views while reducing redundant attention, improving camera pose
estimation, multi-view depth prediction, and 3D reconstruction at lower memory
and latency. This repository provides the code to reproduce the main
evaluations on 7-Scenes (pose, 3D) and Bonn (depth).

## News

- **`2026-06-17`** — Accepted to **ECCV 2026** 🎉
- **`2026-06-22`** — Code released.

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

Each script invokes the matching `eval/eval_chunked_*.py` with our method
`--sampling_method da_partitioning` (diversity-aware partitioning), which splits
the views and applies pose-weighted re-chunking (`--gamma 0.001`,
`--epsilon 0.005`, `--rechunk_remaining_only`).

To run a script directly (e.g. for a custom config):

```bash
python eval/eval_chunked_pose_7scenes.py \
    --dataset_dir /path/to/7scenes \
    --n_frames 500 --chunk_size 50 \
    --sampling_method da_partitioning --local_search_iters 5 \
    --rechunk_remaining_only \
    --output_dir ./results/pose_7scenes
```

Two baselines are available in every script: `--sampling_method
random_partitioning` (random partitioning without local search) and `origin`
(no partitioning — a single full-sequence pass). See `python eval/<script>.py
--help` for the full option list.

## Options

The CLI is intentionally minimal — only the knobs needed to reproduce the paper
are exposed. All three scripts share a common core; the 3D and depth scripts add
a few task-specific options.

**Common (all scripts)**

| Argument | Default | Description |
|---|---|---|
| `--dataset_dir` | *(required)* | Dataset root |
| `--n_frames` | `200` | Frames sampled per sequence |
| `--chunk_size` | `50` | Max frames per chunk |
| `--sampling_method` | `da_partitioning` | `da_partitioning` (ours), `random_partitioning` (random, no local search), or `origin` (no partitioning, single full pass) |
| `--local_search_iters` | `5` | 2-opt local-search iterations (`da_partitioning` only) |
| `--scenes` | all | Subset of scenes to evaluate |
| `--model_path` | HF `facebook/VGGT-1B` | Custom checkpoint |
| `--dtype` | `bfloat16` | `float32` / `bfloat16` / `float16` |
| `--dino_batch_size` | `256` | DINOv2 mini-batch size |
| `--seed` | `42` | Random seed |
| `--output_dir` | `./results_*` | Output directory |

**Pose-weighted re-chunking** — applied automatically for `--sampling_method da_partitioning`:

| Argument | Default | Description |
|---|---|---|
| `--gamma` | `0.001` | Softmax temperature for pseudo-pose soft assignment |
| `--tau` | auto | Pose-weight distance decay (auto = median pairwise distance) |
| `--epsilon` | `0.005` | ε for the appearance−ε·pose score combination |
| `--rechunk_remaining_only` | off | Freeze chunk-0 and re-chunk only the remaining frames |

**3D-only** (`eval_chunked_3d_7scenes.py`): `--conf_thresh`, `--export_ply`, `--ply_dir`.
**Depth-only** (`eval_chunked_depth_bonn.py`): `--min_depth`, `--max_depth`.

Cross-chunk alignment is single-anchor rigid **SE3** (no scale). The
score-combination and pseudo-pose pipeline have a single fixed configuration; the
various ablation knobs used during development (alternative samplers, combine
modes, multi-anchor / Sim3 alignment, oracle GT pose-weighting, τ/rotation
sweeps) are not part of this release.

## Acknowledgements

This work is built on top of [VGGT](https://github.com/facebookresearch/vggt),
whose model and pretrained weights (`facebook/VGGT-1B`) we use directly. Our
chunked evaluation harness and the evaluation-metric code (pose AUC, 3D
accuracy/completion, and depth metrics) are adapted from
[FastVGGT](https://github.com/mystorm16/FastVGGT)
([arXiv:2509.02560](https://arxiv.org/abs/2509.02560)). We thank the authors of
both projects for releasing their code.

## Citation

If you find this work useful, please consider citing:

```bibtex
TBD
```
