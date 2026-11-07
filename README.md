<h1 align="center">Diversity-Aware View Partitioning for Scalable VGGT</h1>

<p align="center">
  <a href="https://arxiv.org/pdf/2607.01885"><img src="https://img.shields.io/badge/Paper-PDF-EC1C24?logo=adobeacrobatreader&logoColor=white" alt="Paper"></a>
  <a href="https://arxiv.org/abs/2607.01885"><img src="https://img.shields.io/badge/arXiv-2607.01885-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://jspark1213.github.io/DA-VGGT/"><img src="https://img.shields.io/badge/Project_Page-DA--VGGT-2088c1" alt="Project Page"></a>
</p>

<p align="center">
  <a href="https://jspark1213.github.io/">Jinsoo Park</a><sup>1</sup>,
  Donggyu Choi<sup>2</sup>,
  <a href="https://ahyunseo.github.io/">Ahyun Seo</a><sup>3</sup>,
  <a href="https://cvlab.postech.ac.kr/~mcho/">Minsu Cho</a><sup>1,4</sup>,
  <a href="https://jeanyson.github.io/">Jeany Son</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>POSTECH&nbsp;&nbsp; <sup>2</sup>GIST&nbsp;&nbsp; <sup>3</sup>KAIST&nbsp;&nbsp; <sup>4</sup>RLWRLD
</p>

DA-VGGT is a training-free, plug-and-play inference framework for VGGT that
organizes input views into diversity-aware balanced chunks, constructed via
combinatorial graph partitioning over visual dissimilarity and spatial
dispersion. This lets the transformer focus attention on geometrically
informative views while reducing redundant attention, improving camera pose
estimation, multi-view depth prediction, and 3D reconstruction at lower memory
and latency. This repository provides the code to reproduce the main
evaluations on ScanNet-50, 7-Scenes and NRGBD (3D), 7-Scenes (pose), and Bonn (depth).

## News

- **`2026-06-17`** — Accepted to **ECCV 2026** 🎉
- **`2026-06-22`** — Code released.

## Layout

```
eval_pose_7scenes.sh             # run: 7-Scenes camera-pose evaluation
eval_3d_7scenes.sh               # run: 7-Scenes 3D-reconstruction evaluation
eval_3d_scannet.sh               # run: ScanNet-50 3D-reconstruction evaluation
eval_3d_nrgbd.sh                 # run: NRGBD 3D-reconstruction evaluation
eval_depth_bonn.sh               # run: Bonn RGB-D depth evaluation
eval/
  eval_chunked_pose_7scenes.py   # 7-Scenes camera-pose evaluation
  eval_chunked_3d_7scenes.py     # 7-Scenes 3D-reconstruction evaluation
  eval_chunked_3d_scannet.py     # ScanNet-50 3D-reconstruction evaluation
  scannet50_scenes.txt          # exact 50 input scenes for Table 3
  eval_chunked_3d_nrgbd.py       # NRGBD 3D-reconstruction evaluation
  recon_{data,metrics,inference,eval}.py # shared reconstruction implementation
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
bash eval_3d_scannet.sh      # ScanNet-50 3D reconstruction
bash eval_3d_nrgbd.sh        # NRGBD 3D reconstruction
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
are exposed. The evaluation scripts share common options; the 3D and depth scripts add
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

**3D-only** (all three datasets): `--conf_thresh`, `--export_ply`, `--ply_dir`,
`--device`, `--dry_run`. Reconstruction defaults to **500 frames** and
`--rechunk_remaining_only`; `--no-rechunk_remaining_only` disables chunk-0 reuse.
`--model_path` accepts a PyTorch checkpoint, a `.safetensors` file, or a local
Hugging Face model directory. PLY output is only written with `--export_ply`.
**Depth-only** (`eval_chunked_depth_bonn.py`): `--min_depth`, `--max_depth`.

Cross-chunk alignment is single-anchor rigid **SE3** (no scale). The
score-combination and pseudo-pose pipeline have a single fixed configuration; the
various ablation knobs used during development (alternative samplers, combine
modes, multi-anchor / Sim3 alignment, oracle GT pose-weighting, τ/rotation
sweeps) are not part of this release.

## 3D reconstruction

The three entry points use the same evaluator: uniform frame selection, GT
depth in meters, a central 224×224 crop, paired Sim(3) alignment using up to
30,000 points, paired subsampling to at most 999,999 points, and rigid
point-to-point ICP with a 0.1 m threshold. Acc and Comp are mean nearest-neighbor
distances in meters; `cd=(acc+comp)/2`, `cd_sum=acc+comp`, and `nc=(nc1+nc2)/2`.
The summary averages sequences equally for 7-Scenes and scenes equally for
ScanNet/NRGBD. RGB is processed with the repository's VGGT preprocessor; GT
depth uses nearest-neighbor resizing and scaled intrinsics.

Expected dataset layouts:

```text
scannet/
  train/scene0000_00/
    color/0.jpg  depth/0.png  pose/0.txt
    intrinsic/intrinsic_depth.txt
7scenes/chess/
  TestSplit.txt
  seq-03/frame-000000.color.png
  seq-03/frame-000000.depth.proj.png   # projected depth, uint16 millimeters
  seq-03/frame-000000.pose.txt         # OpenCV camera-to-world
nrgbd/kitchen/
  images/img0.png  depth/depth0.png    # uint16 millimeters
  poses.txt                          # OpenGL camera-to-world; converted on load
```

The exact 50 input scenes for **ScanNet-50 (Table 3)** are listed in
[`eval/scannet50_scenes.txt`](eval/scannet50_scenes.txt), in evaluation order.
The ScanNet evaluator uses this list by default. Paths are relative to
`--dataset_dir`; both `train/scene*/` / `val/scene*/` and flat `scene*/` exports
are supported. To use a different list, pass `--scene_list /path/to/test_seq.txt`.
`--num_scenes` limits the list (default 50), and `--scenes` selects individual
scenes. 7-Scenes also accepts a sequence such as `chess/seq-03`.

```bash
# Validate the dataset and save exact selected frame IDs without loading a model.
python eval/eval_chunked_3d_scannet.py \
  --dataset_dir /path/to/scannet --dry_run

# Launchers accept environment overrides and additional evaluator options.
DATASET_DIR=/path/to/scannet CUDA_VISIBLE_DEVICES=0 \
  bash eval_3d_scannet.sh --model_path /path/to/VGGT-1B/model.safetensors
DATASET_DIR=/path/to/7scenes bash eval_3d_7scenes.sh
DATASET_DIR=/path/to/nrgbd bash eval_3d_nrgbd.sh

# A short execution check on one scene, including PLY export.
python eval/eval_chunked_3d_scannet.py \
  --dataset_dir /path/to/scannet --scenes train/scene0000_00 \
  --n_frames 4 --chunk_size 3 --export_ply
```

Each run writes a frame manifest and a results JSON under `--output_dir`,
including evaluation settings, per-sequence metrics, failures, inference time,
and peak allocated VRAM. Partial results are saved after every sequence; any
failed sequence gives a nonzero exit status. Use separate output directories
for separate experiment configurations. `--dry_run` checks frame paths and
poses; image decoding is checked during evaluation.

CPU validation: `python -m unittest discover -s eval/tests -v`.

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
@inproceedings{park2026diversity,
  title={Diversity-Aware View Partitioning for Scalable VGGT},
  author={Park, Jinsoo and Choi, Donggyu and Seo, Ahyun and Cho, Minsu and Son, Jeany},
  booktitle={European Conference on Computer Vision},
  pages={547--564},
  year={2026},
  organization={Springer}
}
```
