"""
Bonn RGBD Depth Estimation — In-model FL Chunked Inference (VGGT)

Evaluates depth prediction quality on the Bonn RGBD dataset using chunked
inference. For each scene, uniformly samples n_frames, runs VGGT with
depth_head enabled, and computes:
  - abs_rel: mean(|d_pred - d_gt| / d_gt)  over valid pixels
  - delta_1: fraction of pixels where max(d_pred/d_gt, d_gt/d_pred) < 1.25

Depth scale alignment across chunks uses the anchor frame's predicted depth
(prediction-to-prediction median ratio), no GT is involved.

Dataset format (TUM-style):
    <scene>/rgb.txt       — timestamp + rgb/filename
    <scene>/depth.txt     — timestamp + depth/filename
    <scene>/rgb/*.png     — 480×640 RGB images
    <scene>/depth/*.png   — 480×640 uint16 depth maps (factor=5000 → meters)

Usage:
    python eval_chunked_depth_bonn.py \
        --dataset_dir /workspace/dataset/bonn \
        --n_frames 200 --chunk_size 50 --scenes rgbd_bonn_balloon
"""

import os, sys, json, time, random, logging, warnings
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Allow running from the eval/ subdirectory: add repo root (parent of eval/) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vggt.models.vggt import VGGT
from vggt.models.aggregator import (
    random_balanced_ls_revsim_split,
    compute_pseudo_poses, compute_pose_weight_matrix, rechunk_with_pose_weights,
)
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.multi_anchor import select_anchors, multi_anchor_depth_scale

logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False

DEPTH_FACTOR = 5000.0  # Bonn/TUM: uint16 / 5000 = meters

ALL_SCENES = sorted([
    'rgbd_bonn_balloon', 'rgbd_bonn_balloon2',
    'rgbd_bonn_balloon_tracking', 'rgbd_bonn_balloon_tracking2',
    'rgbd_bonn_crowd', 'rgbd_bonn_crowd2', 'rgbd_bonn_crowd3',
    'rgbd_bonn_kidnapping_box', 'rgbd_bonn_kidnapping_box2',
    'rgbd_bonn_moving_nonobstructing_box', 'rgbd_bonn_moving_nonobstructing_box2',
    'rgbd_bonn_moving_obstructing_box', 'rgbd_bonn_moving_obstructing_box2',
    'rgbd_bonn_person_tracking', 'rgbd_bonn_person_tracking2',
    'rgbd_bonn_placing_nonobstructing_box', 'rgbd_bonn_placing_nonobstructing_box2',
    'rgbd_bonn_placing_nonobstructing_box3', 'rgbd_bonn_placing_obstructing_box',
    'rgbd_bonn_removing_nonobstructing_box', 'rgbd_bonn_removing_nonobstructing_box2',
    'rgbd_bonn_removing_obstructing_box',
    'rgbd_bonn_static', 'rgbd_bonn_static_close_far',
    'rgbd_bonn_synchronous', 'rgbd_bonn_synchronous2',
])


# =============================================================================
# Bonn Dataset Loading (TUM-style)
# =============================================================================

def parse_tum_list(filepath):
    """Parse a TUM-format list file (timestamp + path), skipping comments."""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts = float(parts[0])
            path = parts[1]
            entries.append((ts, path))
    return entries


def associate_nearest(src_entries, dst_entries, max_diff=0.02):
    """Associate src entries to nearest dst entries by timestamp.

    Returns list of (src_idx, dst_idx) pairs where time diff < max_diff.
    """
    associations = []
    dst_ts = np.array([e[0] for e in dst_entries])
    for i, (ts, _) in enumerate(src_entries):
        j = int(np.argmin(np.abs(dst_ts - ts)))
        if abs(ts - dst_ts[j]) < max_diff:
            associations.append((i, j))
    return associations


def load_bonn_frames(scene_path):
    """Load all RGB-Depth paired frames from a Bonn scene.

    Returns list of (rgb_path, depth_path) with absolute paths.
    """
    scene_path = Path(scene_path)
    rgb_list = parse_tum_list(scene_path / 'rgb.txt')
    depth_list = parse_tum_list(scene_path / 'depth.txt')

    associations = associate_nearest(rgb_list, depth_list)

    frames = []
    for rgb_idx, depth_idx in associations:
        rgb_path = scene_path / rgb_list[rgb_idx][1]
        depth_path = scene_path / depth_list[depth_idx][1]
        if rgb_path.exists() and depth_path.exists():
            frames.append((str(rgb_path), str(depth_path)))
    return frames


def sample_frames(frames, n_frames):
    """Uniformly sample up to n_frames from frames list."""
    actual_n = min(n_frames, len(frames))
    if actual_n == 0:
        return None
    if actual_n < n_frames:
        print(f"  Warning: only {len(frames)} frames available, using all (requested {n_frames})")
    indices = np.linspace(0, len(frames) - 1, actual_n, dtype=int)
    return [frames[i] for i in indices]


def load_gt_depth(depth_path, depth_factor=DEPTH_FACTOR):
    """Load GT depth map from uint16 PNG, convert to meters.

    Returns:
        depth_m: (H, W) float32 in meters, 0 = invalid.
    """
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Cannot read depth: {depth_path}")
    return d.astype(np.float32) / depth_factor


# =============================================================================
# Depth Metrics
# =============================================================================

def compute_depth_metrics(pred_depth, gt_depth, min_depth=0.1, max_depth=10.0):
    """Compute depth metrics with per-image median scaling.

    Applies median scaling: scale = median(gt / pred) over valid pixels,
    then evaluates on scaled prediction. This is standard protocol for
    monocular/multi-view depth evaluation (FlashVGGT, DUSt3R, etc.).

    Args:
        pred_depth: (H, W) predicted depth (arbitrary scale).
        gt_depth: (H, W) GT depth in meters (0 = invalid).
        min_depth: minimum valid depth.
        max_depth: maximum valid depth.

    Returns:
        dict with abs_rel, delta_1, median_scale, n_valid, or None if no valid pixels.
    """
    valid = (gt_depth > min_depth) & (gt_depth < max_depth) & (pred_depth > 1e-6)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None

    d_pred = pred_depth[valid]
    d_gt = gt_depth[valid]

    # Per-image median scaling
    median_scale = float(np.median(d_gt / d_pred))
    d_pred_scaled = d_pred * median_scale

    abs_rel = float(np.mean(np.abs(d_pred_scaled - d_gt) / d_gt))

    ratio = np.maximum(d_pred_scaled / d_gt, d_gt / d_pred_scaled)
    delta_1 = float(np.mean(ratio < 1.25))

    return {
        'abs_rel': abs_rel,
        'delta_1': delta_1,
        'median_scale': median_scale,
        'n_valid': n_valid,
    }


def compute_depth_metrics_with_scale(pred_depth, gt_depth, scale,
                                     min_depth=0.1, max_depth=10.0):
    """Compute depth metrics using a pre-computed (global) scale factor.

    Unlike compute_depth_metrics(), this does NOT compute a per-image scale.
    The caller provides the scale to apply, enabling global single-scale
    evaluation that properly measures inter-chunk depth consistency.

    Args:
        pred_depth: (H, W) predicted depth (arbitrary scale).
        gt_depth: (H, W) GT depth in meters (0 = invalid).
        scale: Pre-computed scale factor: pred_scaled = pred * scale.
        min_depth: minimum valid depth.
        max_depth: maximum valid depth.

    Returns:
        dict with abs_rel, delta_1, n_valid, or None if no valid pixels.
    """
    valid = (gt_depth > min_depth) & (gt_depth < max_depth) & (pred_depth > 1e-6)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None

    d_pred = pred_depth[valid] * scale
    d_gt = gt_depth[valid]

    abs_rel = float(np.mean(np.abs(d_pred - d_gt) / d_gt))

    ratio = np.maximum(d_pred / d_gt, d_gt / d_pred)
    delta_1 = float(np.mean(ratio < 1.25))

    return {
        'abs_rel': abs_rel,
        'delta_1': delta_1,
        'n_valid': n_valid,
    }


# =============================================================================
# Chunking quality metrics (same as pose eval)
# =============================================================================

def compute_chunking_quality_metrics(sim_matrix, chunks):
    """Compute chunking quality metrics from similarity matrix and chunks."""
    N = sim_matrix.shape[0]
    K = len(chunks)
    sim = np.clip(sim_matrix, 0, None)

    per_chunk_cov = np.array([
        sim[:, chunk].max(axis=1).sum() if len(chunk) > 0 else 0.0
        for chunk in chunks
    ])

    coverage_profiles = np.zeros((K, N), dtype=np.float32)
    for k, chunk in enumerate(chunks):
        if len(chunk) > 0:
            coverage_profiles[k] = sim[:, chunk].max(axis=1)
    per_frame_var = float(coverage_profiles.var(axis=0).mean())
    chunk_total_var = float(coverage_profiles.sum(axis=1).var())

    diversities = []
    for chunk in chunks:
        if len(chunk) < 2:
            diversities.append(0.0)
            continue
        chunk_sim = sim[np.ix_(chunk, chunk)]
        triu_idx = np.triu_indices(len(chunk), k=1)
        diversities.append(float(1.0 - chunk_sim[triu_idx].mean()))
    diversity = np.array(diversities)

    # --- Pair Utility Coverage: f_ucov(C_k) = sum_i max_{j in C_k} u(sim(i,j)) ---
    alpha_metric = 0.7
    sim_safe_m = np.clip(sim, 1e-8, 1.0 - 1e-8)
    U_m = np.power(sim_safe_m, alpha_metric) * np.power(1.0 - sim_safe_m, 1.0 - alpha_metric)
    np.fill_diagonal(U_m, 0.0)

    per_chunk_ucov = np.array([
        U_m[:, chunk].max(axis=1).sum() if len(chunk) > 0 else 0.0
        for chunk in chunks
    ])

    # --- Within-chunk Pair Quality ---
    pair_qualities = []
    for chunk in chunks:
        if len(chunk) < 2:
            pair_qualities.append(0.0)
            continue
        chunk_U = U_m[np.ix_(chunk, chunk)]
        triu_idx = np.triu_indices(len(chunk), k=1)
        pair_qualities.append(float(chunk_U[triu_idx].mean()))
    pair_quality = np.array(pair_qualities)

    return {
        'coverage_mean': float(per_chunk_cov.mean()),
        'coverage_std': float(per_chunk_cov.std()),
        'coverage_min': float(per_chunk_cov.min()),
        'per_frame_coverage_variance': per_frame_var,
        'chunk_total_coverage_variance': chunk_total_var,
        'diversity_mean': float(diversity.mean()),
        'utility_coverage_mean': float(per_chunk_ucov.mean()),
        'utility_coverage_min': float(per_chunk_ucov.min()),
        'pair_quality_mean': float(pair_quality.mean()),
        'pair_quality_min': float(pair_quality.min()),
    }


# =============================================================================
# Model
# =============================================================================

def load_model(device, model_path=None, chunk_size=50):
    print(f"Loading VGGT model (chunk_size={chunk_size}) ...")
    if model_path is None:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    else:
        model = VGGT()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Enable in-model chunked inference
    model.aggregator.sampling_max_frames = chunk_size

    # Keep depth_head enabled, disable unused heads
    model.point_head = None
    model.track_head = None

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()
    return model


# =============================================================================
# Poseweight depth helpers
# =============================================================================

def _poseweight_allchunk_inference_depth(model, images, rechunked, anchors, device, dtype,
                                          patch_tokens_cpu=None):
    """Run inference on each rechunked chunk using pre-computed patch tokens, return merged depth."""
    H, W = images.shape[-2:]
    K = len(rechunked)
    S = images.shape[1]

    torch.cuda.synchronize()
    t0 = time.time()

    chunk_depths = []
    chunk_depth_confs = []
    depth_chunk_scales = []

    for k in range(K):
        chunk_indices = rechunked[k]
        S_k = len(chunk_indices)
        idx = torch.tensor(chunk_indices, dtype=torch.long)
        pt_chunk = patch_tokens_cpu[idx].to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype, enabled=True):
                output_list, patch_start_idx = model.aggregator.forward_transformer(
                    pt_chunk, B=1, S=S_k, H=H, W=W, device=device)
            del pt_chunk
            chunk_imgs = images[:, chunk_indices]
            depth_k, depth_conf_k = model.depth_head(
                output_list, images=chunk_imgs, patch_start_idx=patch_start_idx)

        chunk_depths.append(depth_k.float()[0])        # (S_k, H_d, W_d, 1)
        chunk_depth_confs.append(depth_conf_k.float()[0])  # (S_k, H_d, W_d)

        del output_list, chunk_imgs, depth_k, depth_conf_k
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t_infer = time.time() - t0

    # Multi-anchor depth scale alignment (median of ratios across all anchors)
    t0 = time.time()
    anchor_set = set(anchors)
    H_d, W_d = chunk_depths[0].shape[1], chunk_depths[0].shape[2]
    all_depth = torch.zeros(S, H_d, W_d, 1, device=device)
    all_depth_conf = torch.zeros(S, H_d, W_d, device=device)

    ref_positions = [rechunked[0].index(a) for a in anchors]
    ref_anchor_depths = [chunk_depths[0][p] for p in ref_positions]

    for k in range(K):
        indices_k = rechunked[k]
        depth_k = chunk_depths[k]
        conf_k = chunk_depth_confs[k]

        if k == 0:
            scale_k = 1.0
        else:
            src_positions = [indices_k.index(a) for a in anchors]
            src_anchor_depths = [depth_k[p] for p in src_positions]
            scale_k = multi_anchor_depth_scale(ref_anchor_depths, src_anchor_depths)
        depth_chunk_scales.append(scale_k)

        for pos, idx in enumerate(indices_k):
            if k == 0 or idx not in anchor_set:
                all_depth[idx] = depth_k[pos] * scale_k
                all_depth_conf[idx] = conf_k[pos]

    t_align = time.time() - t0

    return (all_depth.unsqueeze(0).cpu(), all_depth_conf.unsqueeze(0).cpu(),
            depth_chunk_scales, t_infer, t_align)


def _poseweight_remaining_inference_depth(model, images, rechunked, anchors,
                                           chunk0_depth, chunk0_depth_conf,
                                           chunk0_indices, device, dtype,
                                           patch_tokens_cpu=None):
    """Reuse chunk0 depth, infer only K-1 remaining chunks."""
    H, W = images.shape[-2:]
    K = len(rechunked)
    S = images.shape[1]

    torch.cuda.synchronize()
    t0 = time.time()

    chunk_depths = [chunk0_depth]
    chunk_depth_confs = [chunk0_depth_conf]

    for k in range(1, K):
        chunk_indices = rechunked[k]
        S_k = len(chunk_indices)
        idx = torch.tensor(chunk_indices, dtype=torch.long)
        pt_chunk = patch_tokens_cpu[idx].to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype, enabled=True):
                output_list, patch_start_idx = model.aggregator.forward_transformer(
                    pt_chunk, B=1, S=S_k, H=H, W=W, device=device)
            del pt_chunk
            chunk_imgs = images[:, chunk_indices]
            depth_k, depth_conf_k = model.depth_head(
                output_list, images=chunk_imgs, patch_start_idx=patch_start_idx)

        chunk_depths.append(depth_k.float()[0])
        chunk_depth_confs.append(depth_conf_k.float()[0])

        del output_list, chunk_imgs, depth_k, depth_conf_k
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t_infer = time.time() - t0

    # Multi-anchor depth scale alignment (median of ratios across all anchors)
    t0 = time.time()
    anchor_set = set(anchors)
    depth_chunk_scales = []
    H_d, W_d = chunk_depths[0].shape[1], chunk_depths[0].shape[2]
    all_depth = torch.zeros(S, H_d, W_d, 1, device=device)
    all_depth_conf = torch.zeros(S, H_d, W_d, device=device)

    ref_positions = [rechunked[0].index(a) for a in anchors]
    ref_anchor_depths = [chunk_depths[0][p] for p in ref_positions]

    for k in range(K):
        indices_k = rechunked[k]
        depth_k = chunk_depths[k]
        conf_k = chunk_depth_confs[k]
        if k == 0:
            scale_k = 1.0
        else:
            src_positions = [indices_k.index(a) for a in anchors]
            src_anchor_depths = [depth_k[p] for p in src_positions]
            scale_k = multi_anchor_depth_scale(ref_anchor_depths, src_anchor_depths)
        depth_chunk_scales.append(scale_k)

        for pos, idx in enumerate(indices_k):
            if k == 0 or idx not in anchor_set:
                all_depth[idx] = depth_k[pos] * scale_k
                all_depth_conf[idx] = conf_k[pos]

    t_align = time.time() - t0
    return (all_depth.unsqueeze(0).cpu(), all_depth_conf.unsqueeze(0).cpu(),
            depth_chunk_scales, t_infer, t_align)


def run_inference_poseweight_depth(model, image_paths, device, dtype,
                                    combine_mode="A", alpha_combine=0.5,
                                    gamma=1.0, tau=None,
                                    score_type="revsim", ls_iters=5,
                                    n_anchors=1, chunk_size=50,
                                    rechunk_remaining_only=False,
                                    epsilon=None,
                                    anchor_select="uniform", seed=42):
    """Poseweight rechunking for depth evaluation.

    Pipeline:
      Phase 1: DINO all -> sim matrix -> initial chunking (step_ls_revsim)
      Phase 2: 1st chunk inference -> extract positions -> pseudo-poses
      Phase 3: Compute W_pose -> combine with appearance -> re-chunk via 2-opt LS
      Phase 4: Depth-specific inference on rechunked chunks

    If rechunk_remaining_only=True:
      - chunk0 result is kept as-is (no re-assignment of chunk0 frames)
      - Only remaining N-|chunk0| frames are re-chunked into K-1 chunks
      - chunk0 depth from Phase 2 is reused; only K-1 chunks are inferred in Phase 4

    Returns:
        (depth, depth_conf, img_shape, K, anchors, timing, initial_chunks,
         rechunked_chunks, sim_matrix, tau_used, depth_chunk_scales)
    """
    timing = {}
    images = load_and_preprocess_images(image_paths).to(device)
    if images.dim() == 4:
        images = images.unsqueeze(0)
    S = images.shape[1]
    img_shape = images.shape[-2:]

    torch.cuda.reset_peak_memory_stats(device)

    # ---- Phase 1a: DINOv2 on all images ----
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype, enabled=True):
            patch_tokens_cpu, pooled_tokens = model.aggregator.forward_dino(images)
    torch.cuda.synchronize()
    timing['dino'] = time.time() - t0
    timing['peak_vram_dino_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # ---- Phase 1b: Similarity matrix ----
    t0 = time.time()
    feats = F.normalize(pooled_tokens, dim=-1)
    sim_matrix = (feats @ feats.T).numpy()
    del pooled_tokens, feats
    timing['sim_matrix'] = time.time() - t0

    # ---- Phase 1c: Anchor selection (uniform / fps_sim / random / first) ----
    precomputed_anchors = select_anchors(S, n_anchors, mode=anchor_select,
                                         sim_matrix=sim_matrix, seed=seed)
    timing['anchor_select'] = anchor_select

    # ---- Phase 1d: Initial chunking (random + revsim LS) ----
    t0 = time.time()
    chunks, anchors_init, t_ls_initial = random_balanced_ls_revsim_split(
        sim_matrix, chunk_size, n_anchors=n_anchors,
        local_search_iters=ls_iters,
        anchors_override=precomputed_anchors,
    )
    timing['initial_chunking'] = time.time() - t0
    timing['initial_ls'] = t_ls_initial
    initial_chunks = [list(ch) for ch in chunks]  # deep copy for logging

    # ---- Phase 2a: First chunk inference (reuse DINOv2 tokens) ----
    torch.cuda.synchronize()
    t0 = time.time()

    chunk0_indices = chunks[0]
    S_k0 = len(chunk0_indices)
    H, W = img_shape
    idx_t = torch.tensor(chunk0_indices, dtype=torch.long)
    pt_chunk0 = patch_tokens_cpu[idx_t].to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype, enabled=True):
            output_list0, patch_start_idx0 = model.aggregator.forward_transformer(
                pt_chunk0, B=1, S=S_k0, H=H, W=W, device=device)
        del pt_chunk0
        with torch.amp.autocast('cuda', enabled=False):
            pose_enc_list0 = model.camera_head(output_list0)
            pose_enc_0 = pose_enc_list0[-1][0]
        chunk0_imgs = images[:, chunk0_indices]
        depth_out0, depth_conf_out0 = model.depth_head(
            output_list0, images=chunk0_imgs, patch_start_idx=patch_start_idx0)
        chunk0_depth_pred = depth_out0.float()       # (1, S_k, H_d, W_d, 1)
        chunk0_depth_conf_pred = depth_conf_out0.float()  # (1, S_k, H_d, W_d)
        del output_list0, pose_enc_list0, chunk0_imgs, depth_out0, depth_conf_out0

    # Decode chunk0 extrinsics (for camera positions)
    pe0_f = pose_enc_0.unsqueeze(0).float()
    with torch.amp.autocast('cuda', dtype=torch.float64, enabled=True):
        ext0_full, _ = pose_encoding_to_extri_intri(pe0_f.to(device), (H, W))
    ext0_full = ext0_full[0]  # (S_k, 3, 4)

    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    timing['first_chunk_inference'] = time.time() - t0

    # ---- Phase 2b: Pseudo-poses for all N frames ----
    t0 = time.time()
    # Camera center = -R^T @ t from chunk0 extrinsics
    ext_np = ext0_full.cpu().numpy()  # (S_k, 3, 4)
    R_w2c = ext_np[:, :3, :3]   # (S_k, 3, 3)
    t_w2c = ext_np[:, :3, 3]    # (S_k, 3)
    chunk0_positions = np.array([
        -R_w2c[i].T @ t_w2c[i] for i in range(len(t_w2c))
    ])  # (S_k, 3)
    pseudo_positions = compute_pseudo_poses(
        chunk0_positions, chunk0_indices, S, sim_matrix, gamma=gamma,
    )
    timing['pseudo_pose_computation'] = time.time() - t0

    # ---- Phase 3a: Pose weight matrix ----
    t0 = time.time()
    W_pose, tau_used = compute_pose_weight_matrix(pseudo_positions, tau=tau)
    timing['pose_weight_computation'] = time.time() - t0

    # ---- Phase 3b: Re-chunking ----
    t0 = time.time()
    if rechunk_remaining_only:
        # Freeze chunk0; re-chunk only remaining frames into K-1 chunks
        remaining_chunks = [ch for ch in chunks[1:]]  # K-1 chunks
        remaining_anchors = anchors_init  # anchor is shared

        rechunked_rest, anchors_new, t_ls_rechunk = rechunk_with_pose_weights(
            remaining_chunks, remaining_anchors, sim_matrix, W_pose,
            score_type=score_type, alpha=0.0,
            combine_mode=combine_mode, alpha_combine=alpha_combine,
            n_anchors=n_anchors, local_search_iters=ls_iters,
            epsilon=epsilon,
            anchors_override=precomputed_anchors,
        )
        # Prepend frozen chunk0
        rechunked = [list(chunk0_indices)] + rechunked_rest
        anchors_new = anchors_init  # anchor unchanged
    else:
        # Original: re-chunk ALL frames
        rechunked, anchors_new, t_ls_rechunk = rechunk_with_pose_weights(
            chunks, anchors_init, sim_matrix, W_pose,
            score_type=score_type, alpha=0.0,
            combine_mode=combine_mode, alpha_combine=alpha_combine,
            n_anchors=n_anchors, local_search_iters=ls_iters,
            epsilon=epsilon,
            anchors_override=precomputed_anchors,
        )
    timing['rechunking_total'] = time.time() - t0
    timing['rechunking_ls'] = t_ls_rechunk
    rechunked_chunks = [list(ch) for ch in rechunked]

    # ---- Phase 4: Depth inference on rechunked chunks ----
    if rechunk_remaining_only:
        # Reuse chunk0 depth; only infer K-1 remaining chunks
        (depth_pred, depth_conf_pred, depth_chunk_scales,
         t_infer, t_align) = _poseweight_remaining_inference_depth(
            model, images, rechunked, anchors_new,
            chunk0_depth_pred[0], chunk0_depth_conf_pred[0],
            chunk0_indices, device, dtype,
            patch_tokens_cpu=patch_tokens_cpu,
        )
    else:
        (depth_pred, depth_conf_pred, depth_chunk_scales,
         t_infer, t_align) = _poseweight_allchunk_inference_depth(
            model, images, rechunked, anchors_new, device, dtype,
            patch_tokens_cpu=patch_tokens_cpu,
        )
    timing['all_chunks_inference'] = t_infer
    timing['alignment'] = t_align

    timing['total'] = (timing['dino'] + timing['sim_matrix']
                       + timing['initial_chunking']
                       + timing['first_chunk_inference']
                       + timing['pseudo_pose_computation']
                       + timing['pose_weight_computation']
                       + timing['rechunking_total']
                       + timing['all_chunks_inference']
                       + timing['alignment'])
    timing['peak_vram_total_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del images, patch_tokens_cpu
    torch.cuda.empty_cache()

    K = len(rechunked)
    return (depth_pred, depth_conf_pred, img_shape, K, anchors_new, timing,
            initial_chunks, rechunked_chunks, sim_matrix, tau_used,
            depth_chunk_scales)


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, image_paths, device, dtype):
    images = load_and_preprocess_images(image_paths).to(device)
    img_shape = images.shape[-2:]  # (H_model, W_model)

    torch.cuda.synchronize()
    t_total_start = time.time()

    with torch.no_grad():
        if dtype in (torch.bfloat16, torch.float16):
            images = images.to(dtype)
        with torch.amp.autocast('cuda', dtype=dtype, enabled=True):
            pred = model(images)

    torch.cuda.synchronize()
    total_time = time.time() - t_total_start

    # Extract depth predictions
    depth = pred.get('depth', None)        # [1, S, H_d, W_d, 1] or None
    depth_conf = pred.get('depth_conf', None)

    num_chunks = pred.get('num_chunks', 1)
    anchor = pred.get('anchor', None)
    anchors = pred.get('anchors', [anchor] if anchor is not None else None)
    timing = pred.get('timing', None)
    sim_matrix = pred.get('sim_matrix', None)
    chunk_frame_indices = pred.get('chunk_frame_indices', None)
    depth_chunk_scales = pred.get('depth_chunk_scales', None)
    if timing is not None:
        timing['total'] = total_time
    else:
        timing = {'total': total_time}
    timing['peak_vram_total_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del images, pred
    torch.cuda.empty_cache()
    return depth, depth_conf, img_shape, num_chunks, anchor, anchors, timing, sim_matrix, chunk_frame_indices, depth_chunk_scales


# =============================================================================
# Evaluation per scene
# =============================================================================

def evaluate_scene(scene_name, scene_path, model, device, dtype, n_frames,
                   min_depth, max_depth,
                   poseweight_mode=None, combine_mode="E",
                   gamma=0.001, tau=None, epsilon=None,
                   rechunk_remaining_only=False):
    all_frames = load_bonn_frames(scene_path)
    print(f"  Loaded {len(all_frames)} RGB-Depth pairs")

    sampled = sample_frames(all_frames, n_frames)
    if sampled is None:
        print(f"  No frames found in {scene_name}, skipping.")
        return None

    image_paths = [f[0] for f in sampled]
    depth_paths = [f[1] for f in sampled]
    N = len(sampled)

    # Inference
    if poseweight_mode == "pseudo":
        chunk_size = model.aggregator.sampling_max_frames
        if chunk_size <= 0:
            chunk_size = 50
        ls_iters = model.aggregator.sampling_local_search_iters
        seed = getattr(model.aggregator, 'sampling_seed', 42)

        (depth_pred, depth_conf, img_shape, num_chunks, anchors,
         timing, initial_chunks, rechunked_chunks, sim_matrix, tau_used,
         depth_chunk_scales) = run_inference_poseweight_depth(
            model, image_paths, device, dtype,
            combine_mode=combine_mode, alpha_combine=0.5,
            gamma=gamma, tau=tau,
            score_type="revsim", ls_iters=ls_iters,
            n_anchors=1, chunk_size=chunk_size,
            rechunk_remaining_only=rechunk_remaining_only,
            epsilon=epsilon,
            anchor_select="uniform", seed=seed)
        anchor = anchors[0] if anchors else None
        chunk_frame_indices = rechunked_chunks
    else:
        depth_pred, depth_conf, img_shape, num_chunks, anchor, anchors, timing, \
            sim_matrix, chunk_frame_indices, depth_chunk_scales = \
            run_inference(model, image_paths, device, dtype)

    if depth_pred is None:
        print(f"  depth_head returned None — model may not support depth in chunked mode.")
        return None

    # depth_pred: [1, S, H_d, W_d, 1]
    depth_pred_np = depth_pred[0].cpu().numpy()  # [S, H_d, W_d, 1]

    # Per-frame depth evaluation (pass 1: per-image scaling + collect data for global scaling)
    per_frame_metrics = []
    all_gt_over_pred = []      # for global scale computation
    frame_valid_data = []      # store (pred_resized, gt_depth) for pass 2
    for i in range(N):
        gt_depth = load_gt_depth(depth_paths[i])  # (H_gt, W_gt) in meters
        H_gt, W_gt = gt_depth.shape

        # Resize prediction to GT resolution
        pred_i = depth_pred_np[i, :, :, 0]  # [H_d, W_d]
        pred_resized = cv2.resize(pred_i, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)

        metrics = compute_depth_metrics(pred_resized, gt_depth, min_depth, max_depth)
        per_frame_metrics.append(metrics)

        # Collect valid gt/pred ratios for global scale
        valid = (gt_depth > min_depth) & (gt_depth < max_depth) & (pred_resized > 1e-6)
        if valid.sum() > 0:
            all_gt_over_pred.append(gt_depth[valid] / pred_resized[valid])
        frame_valid_data.append((pred_resized, gt_depth))

    # Aggregate per-image-scaled metrics (skip frames with no valid pixels)
    valid_metrics = [m for m in per_frame_metrics if m is not None]
    if len(valid_metrics) == 0:
        print(f"  No valid depth pixels in any frame.")
        return None

    mean_abs_rel = float(np.mean([m['abs_rel'] for m in valid_metrics]))
    mean_delta_1 = float(np.mean([m['delta_1'] for m in valid_metrics]))
    mean_median_scale = float(np.mean([m['median_scale'] for m in valid_metrics]))
    std_median_scale = float(np.std([m['median_scale'] for m in valid_metrics]))
    total_valid_pixels = sum(m['n_valid'] for m in valid_metrics)

    # Pass 2: Global single-scale evaluation
    if all_gt_over_pred:
        all_ratios = np.concatenate(all_gt_over_pred)
        seq_global_scale = float(np.median(all_ratios))
        del all_ratios, all_gt_over_pred

        per_frame_global_metrics = []
        for i in range(N):
            pred_resized, gt_depth = frame_valid_data[i]
            gm = compute_depth_metrics_with_scale(
                pred_resized, gt_depth, seq_global_scale, min_depth, max_depth
            )
            per_frame_global_metrics.append(gm)

        valid_global = [m for m in per_frame_global_metrics if m is not None]
        if valid_global:
            global_abs_rel = float(np.mean([m['abs_rel'] for m in valid_global]))
            global_delta_1 = float(np.mean([m['delta_1'] for m in valid_global]))
        else:
            global_abs_rel = None
            global_delta_1 = None
    else:
        seq_global_scale = None
        global_abs_rel = None
        global_delta_1 = None
        per_frame_global_metrics = [None] * N

    del frame_valid_data, depth_pred, depth_conf
    torch.cuda.empty_cache()

    # Per-chunk depth evaluation
    per_chunk_eval = []
    if chunk_frame_indices is not None and len(chunk_frame_indices) > 1:
        for k, chunk_idx_list in enumerate(chunk_frame_indices):
            chunk_metrics = [per_frame_metrics[i] for i in chunk_idx_list if per_frame_metrics[i] is not None]
            chunk_global = [per_frame_global_metrics[i] for i in chunk_idx_list if per_frame_global_metrics[i] is not None]
            if chunk_metrics:
                c_abs_rel = float(np.mean([m['abs_rel'] for m in chunk_metrics]))
                c_delta_1 = float(np.mean([m['delta_1'] for m in chunk_metrics]))
            else:
                c_abs_rel = None
                c_delta_1 = None
            if chunk_global:
                c_g_abs_rel = float(np.mean([m['abs_rel'] for m in chunk_global]))
                c_g_delta_1 = float(np.mean([m['delta_1'] for m in chunk_global]))
            else:
                c_g_abs_rel = None
                c_g_delta_1 = None
            per_chunk_eval.append({
                'chunk_id': k,
                'n_frames': len(chunk_idx_list),
                'abs_rel': c_abs_rel,
                'delta_1': c_delta_1,
                'global_abs_rel': c_g_abs_rel,
                'global_delta_1': c_g_delta_1,
            })

    # Chunking quality metrics
    chunking_metrics = None
    if sim_matrix is not None and chunk_frame_indices is not None:
        chunking_metrics = compute_chunking_quality_metrics(sim_matrix, chunk_frame_indices)

    return {
        'abs_rel': mean_abs_rel,
        'delta_1': mean_delta_1,
        'median_scale_mean': mean_median_scale,
        'median_scale_std': std_median_scale,
        'global_abs_rel': global_abs_rel,
        'global_delta_1': global_delta_1,
        'global_scale': seq_global_scale,
        'n_frames': N,
        'n_valid_frames': len(valid_metrics),
        'total_valid_pixels': total_valid_pixels,
        'n_total_frames': len(all_frames),
        'num_chunks': num_chunks,
        'anchor': anchor,
        'anchors': anchors,
        'depth_chunk_scales': depth_chunk_scales,
        'timing': timing,
        'chunking_metrics': chunking_metrics,
        'per_chunk_eval': per_chunk_eval,
        'per_frame_abs_rel': [m['abs_rel'] if m else None for m in per_frame_metrics],
        'per_frame_delta_1': [m['delta_1'] if m else None for m in per_frame_metrics],
        'per_frame_global_abs_rel': [m['abs_rel'] if m else None for m in per_frame_global_metrics],
        'per_frame_global_delta_1': [m['delta_1'] if m else None for m in per_frame_global_metrics],
        'chunk_frame_indices': chunk_frame_indices,
        'image_paths': image_paths,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bonn RGBD depth eval — VGGT in-model FL chunked inference"
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to Bonn dataset root (contains rgbd_bonn_*/)")
    parser.add_argument("--n_frames", type=int, default=200,
                        help="Number of frames to uniformly sample per scene")
    parser.add_argument("--chunk_size", type=int, default=50,
                        help="Max frames per chunk (sampling_max_frames)")
    parser.add_argument("--sampling_method", type=str, default="random_ls_revsim",
                        choices=["origin", "random_ls_revsim"],
                        help="Chunk sampling method. "
                             "'origin' = single-batch, no chunking; "
                             "'random_ls_revsim' = diversity-aware local-search reverse-similarity split.")
    parser.add_argument("--local_search_iters", type=int, default=5,
                        help="2-opt local search iterations for random_ls_revsim (0=skip)")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Specific scenes to evaluate (default: all)")
    parser.add_argument("--min_depth", type=float, default=0.1,
                        help="Minimum valid GT depth in meters")
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="Maximum valid GT depth in meters")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Custom model checkpoint (default: facebook/VGGT-1B)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--dino_batch_size", type=int, default=256,
                        help="DINOv2 mini-batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results_chunked_depth_bonn")
    # Poseweight arguments
    parser.add_argument("--poseweight_mode", type=str, default=None,
                        choices=["pseudo", "gt"],
                        help="Pose-weighted re-chunking: pseudo (1st chunk inference) or gt (GT poses)")
    parser.add_argument("--combine_mode", type=str, default="E",
                        choices=["E"],
                        help="Poseweight combine mode")
    parser.add_argument("--gamma", type=float, default=0.001,
                        help="Pseudo-pose interpolation gamma")
    parser.add_argument("--tau", type=float, default=None,
                        help="Pose weight temperature (None=auto)")
    parser.add_argument("--epsilon", type=float, default=0.005,
                        help="Pose weight epsilon threshold (None=disabled)")
    parser.add_argument("--rechunk_remaining_only", action="store_true",
                        help="Freeze chunk0 and only rechunk remaining frames")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]
    device = "cuda"

    scenes = args.scenes if args.scenes else ALL_SCENES

    model = load_model(device, args.model_path, args.chunk_size)
    model.aggregator.dino_batch_size = args.dino_batch_size
    model.aggregator.sampling_method = args.sampling_method
    model.aggregator.sampling_seed = args.seed
    model.aggregator.sampling_local_search_iters = args.local_search_iters

    # Origin mode: disable chunking, run single-batch forward
    if args.sampling_method == "origin":
        model.aggregator.sampling_max_frames = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    global_timings = []
    chunk_assignments = []  # (scene_name, chunk_frame_indices, image_paths)

    GREEN, BLUE, BOLD, RED, RESET = "\033[92m", "\033[94m", "\033[1m", "\033[91m", "\033[0m"

    print(f"\n{'#'*60}")
    print(f"# Chunked Depth Eval (Bonn RGBD, sampling={args.sampling_method})")
    print(f"# n_frames={args.n_frames}, chunk_size={args.chunk_size}, "
          f"local_search_iters={args.local_search_iters}, dino_batch_size={args.dino_batch_size}")
    print(f"# depth range=[{args.min_depth}, {args.max_depth}]m")
    print(f"{'#'*60}\n")

    for scene_name in scenes:
        print(f"\n{'='*60}")
        print(f"Scene: {scene_name}")
        print(f"{'='*60}")

        scene_path = Path(args.dataset_dir) / scene_name
        if not scene_path.exists():
            print(f"  Scene not found: {scene_path}")
            continue

        result = evaluate_scene(
            scene_name, scene_path, model, device, dtype,
            args.n_frames, args.min_depth, args.max_depth,
            poseweight_mode=args.poseweight_mode,
            combine_mode=args.combine_mode,
            gamma=args.gamma, tau=args.tau, epsilon=args.epsilon,
            rechunk_remaining_only=args.rechunk_remaining_only,
        )

        if result is None:
            all_results[scene_name] = {'summary': None}
            continue

        chunk_info = f"K={result['num_chunks']}"
        if result.get('anchors') is not None and len(result['anchors']) > 1:
            chunk_info += f", anchors={result['anchors']}"
        elif result['anchor'] is not None:
            chunk_info += f", anchor={result['anchor']}"

        # Depth scale diagnostics
        scales = result.get('depth_chunk_scales')
        if scales and len(scales) > 1:
            scale_std = float(np.std(scales))
            scale_range = f"{min(scales):.4f}~{max(scales):.4f}"
            chunk_info += f", depth_scales={scale_range} (std={scale_std:.4f})"

        t = result['timing']
        if t is not None and 'first_chunk_inference' in t:
            # Poseweight timing
            timing_str = (f"total={t['total']:.2f}s  "
                          f"[dino={t['dino']:.2f}s, sim={t['sim_matrix']:.3f}s, "
                          f"init_chunk={t['initial_chunking']:.3f}s, "
                          f"1st_infer={t['first_chunk_inference']:.2f}s, "
                          f"pseudo={t['pseudo_pose_computation']:.3f}s, "
                          f"pw_mat={t['pose_weight_computation']:.3f}s, "
                          f"rechunk={t['rechunking_total']:.3f}s, "
                          f"infer={t['all_chunks_inference']:.2f}s, "
                          f"align={t['alignment']:.3f}s]")
            vram_str = f"peak VRAM: {t['peak_vram_total_mb']:.0f}MB (dino={t['peak_vram_dino_mb']:.0f})" if 'peak_vram_total_mb' in t else ""
            global_timings.append(t)
        elif t is not None and 'dino' in t:
            depth_str = ""
            if 'depth_head_total' in t:
                depth_str = (f", depth_head={t['depth_head_total']:.2f}s"
                             f", depth_asm={t.get('depth_assembly', 0):.2f}s")
            timing_str = (f"total={t['total']:.2f}s  "
                          f"[dino={t['dino']:.2f}s, "
                          f"sampling={t['sampling_total']:.3f}s "
                          f"(sim={t['sampling_sim']:.3f}s + init={t.get('sampling_init', t['sampling_fl']):.3f}s + ls={t.get('sampling_ls', 0):.3f}s), "
                          f"transformer={t['transformer_total']:.2f}s, "
                          f"align={t['alignment']:.3f}s"
                          f"{depth_str}]")
            vram_str = (f"peak VRAM: {t['peak_vram_total_mb']:.0f}MB "
                        f"(dino={t['peak_vram_dino_mb']:.0f}, "
                        f"sampling={t['peak_vram_sampling_mb']:.0f}, "
                        f"transformer={t['peak_vram_transformer_mb']:.0f})")
            global_timings.append(t)
        elif t is not None:
            timing_str = f"total={t['total']:.2f}s (single-batch, no chunking)"
            vram_str = f"peak VRAM: {t['peak_vram_total_mb']:.0f}MB" if 'peak_vram_total_mb' in t else ""
            global_timings.append(t)
        else:
            timing_str = "total=N/A"
            vram_str = ""

        print(f"\n  {BOLD}{BLUE}{scene_name} (n={result['n_frames']}/{result['n_total_frames']}, "
              f"{chunk_info}):{RESET}")
        print(f"  {GREEN}abs_rel={result['abs_rel']:.4f}, delta<1.25={result['delta_1']:.4f}{RESET}  (per-image scale)")
        if result.get('global_abs_rel') is not None:
            print(f"  {GREEN}abs_rel={result['global_abs_rel']:.4f}, delta<1.25={result['global_delta_1']:.4f}{RESET}  (global scale={result['global_scale']:.4f})")
        print(f"  median_scale: mean={result['median_scale_mean']:.4f}, std={result['median_scale_std']:.4f}")
        print(f"  valid_frames={result['n_valid_frames']}/{result['n_frames']}, "
              f"valid_pixels={result['total_valid_pixels']}")
        print(f"  {timing_str}")
        if vram_str:
            print(f"  {vram_str}")

        # Per-chunk evaluation
        if result.get('per_chunk_eval'):
            print(f"\n  {BOLD}Per-chunk evaluation:{RESET}")
            for ce in result['per_chunk_eval']:
                if ce['abs_rel'] is not None:
                    g_str = ""
                    if ce.get('global_abs_rel') is not None:
                        g_str = f"  global: abs_rel={ce['global_abs_rel']:.4f}, d<1.25={ce['global_delta_1']:.4f}"
                    print(f"    Chunk {ce['chunk_id']} (n={ce['n_frames']:>3}): "
                          f"{GREEN}abs_rel={ce['abs_rel']:.4f}, d<1.25={ce['delta_1']:.4f}{RESET}{g_str}")
                else:
                    print(f"    Chunk {ce['chunk_id']} (n={ce['n_frames']:>3}): no valid frames")

        # Serialize timing
        timing_save = None
        if t is not None:
            timing_save = {k: v for k, v in t.items() if k != 'transformer_per_chunk'}
            timing_save['transformer_per_chunk'] = t.get('transformer_per_chunk', [])

        scene_summary = {
            'abs_rel': result['abs_rel'],
            'delta_1': result['delta_1'],
            'median_scale_mean': result['median_scale_mean'],
            'median_scale_std': result['median_scale_std'],
            'global_abs_rel': result.get('global_abs_rel'),
            'global_delta_1': result.get('global_delta_1'),
            'global_scale': result.get('global_scale'),
            'n_frames': result['n_frames'],
            'n_valid_frames': result['n_valid_frames'],
            'total_valid_pixels': result['total_valid_pixels'],
            'n_total_frames': result['n_total_frames'],
            'num_chunks': result['num_chunks'],
            'anchor': result['anchor'],
            'anchors': result.get('anchors'),
            'depth_chunk_scales': result.get('depth_chunk_scales'),
            'timing': timing_save,
            'chunking_metrics': result.get('chunking_metrics'),
            'per_chunk_eval': result.get('per_chunk_eval', []),
        }

        all_results[scene_name] = {'summary': scene_summary}

        if result.get('chunk_frame_indices') is not None:
            chunk_assignments.append((scene_name, result['chunk_frame_indices'], result['image_paths']))

        # Running mean
        evaluated = [s for s in all_results if all_results[s]['summary'] is not None]
        if len(evaluated) > 1:
            mean_abs_rel = np.mean([all_results[s]['summary']['abs_rel'] for s in evaluated])
            mean_delta_1 = np.mean([all_results[s]['summary']['delta_1'] for s in evaluated])
            print(f"  {BOLD}{BLUE}Mean so far ({len(evaluated)} scenes):{RESET} "
                  f"{RED}abs_rel={mean_abs_rel:.4f}, delta<1.25={mean_delta_1:.4f}{RESET}  (per-image)")
            eval_g = [s for s in evaluated if all_results[s]['summary'].get('global_abs_rel') is not None]
            if eval_g:
                mg_ar = np.mean([all_results[s]['summary']['global_abs_rel'] for s in eval_g])
                mg_d1 = np.mean([all_results[s]['summary']['global_delta_1'] for s in eval_g])
                print(f"  {' ' * len(f'Mean so far ({len(evaluated)} scenes):')}"
                      f"  {RED}abs_rel={mg_ar:.4f}, delta<1.25={mg_d1:.4f}{RESET}  (global)")

    # =========================================================================
    # Global summary
    # =========================================================================
    scene_results = {s: all_results[s]['summary']
                     for s in all_results if all_results[s]['summary'] is not None}

    if scene_results:
        mean_ar = float(np.mean([scene_results[s]['abs_rel'] for s in scene_results]))
        mean_d1 = float(np.mean([scene_results[s]['delta_1'] for s in scene_results]))

        # Global-scaled overall means
        scenes_with_global = [s for s in scene_results if scene_results[s].get('global_abs_rel') is not None]
        if scenes_with_global:
            mean_g_ar = float(np.mean([scene_results[s]['global_abs_rel'] for s in scenes_with_global]))
            mean_g_d1 = float(np.mean([scene_results[s]['global_delta_1'] for s in scenes_with_global]))
        else:
            mean_g_ar = None
            mean_g_d1 = None

        global_summary = {
            'abs_rel': mean_ar,
            'delta_1': mean_d1,
            'global_abs_rel': mean_g_ar,
            'global_delta_1': mean_g_d1,
            'num_scenes': len(scene_results),
        }

        print(f"\n{'#'*60}")
        print("OVERALL SUMMARY (per-image scale | global scale)")
        print(f"{'#'*60}")
        header = f"{'Scene':<45} {'abs_rel':<10} {'d<1.25':<10} {'g_abs_rel':<10} {'g_d<1.25':<10}"
        print(header)
        print("-" * len(header))

        for scene_name in ALL_SCENES:
            if scene_name not in scene_results:
                continue
            r = scene_results[scene_name]
            g_ar = f"{r['global_abs_rel']:.4f}" if r.get('global_abs_rel') is not None else "N/A"
            g_d1 = f"{r['global_delta_1']:.4f}" if r.get('global_delta_1') is not None else "N/A"
            print(f"{scene_name:<45} {r['abs_rel']:<10.4f} {r['delta_1']:<10.4f} {g_ar:<10} {g_d1:<10}")

        print("-" * len(header))
        g_ar_mean = f"{mean_g_ar:.4f}" if mean_g_ar is not None else "N/A"
        g_d1_mean = f"{mean_g_d1:.4f}" if mean_g_d1 is not None else "N/A"
        print(f"{'MEAN':<45} {mean_ar:<10.4f} {mean_d1:<10.4f} {g_ar_mean:<10} {g_d1_mean:<10}")

        # Scale consistency summary
        all_scales = []
        for s in scene_results:
            sc = scene_results[s].get('depth_chunk_scales')
            if sc and len(sc) > 1:
                all_scales.append(np.std(sc))
        if all_scales:
            print(f"\n  Depth scale consistency (std of per-chunk scales):")
            print(f"    mean_std = {np.mean(all_scales):.4f}, max_std = {np.max(all_scales):.4f}")

        # Global timing summary
        if global_timings:
            is_pw = 'first_chunk_inference' in global_timings[0]
            if is_pw:
                keys = ['total', 'dino', 'sim_matrix', 'initial_chunking',
                        'first_chunk_inference', 'pseudo_pose_computation',
                        'pose_weight_computation', 'rechunking_total',
                        'all_chunks_inference', 'alignment']
            else:
                keys = ['total', 'dino', 'sampling_sim', 'sampling_fl', 'sampling_ls', 'sampling_init', 'sampling_total',
                        'transformer_total', 'alignment']
            avg_timing = {}
            for k in keys:
                vals = [t[k] for t in global_timings if k in t]
                if vals:
                    avg_timing[k] = float(np.mean(vals))
            vram_keys = ['peak_vram_dino_mb', 'peak_vram_sampling_mb',
                         'peak_vram_transformer_mb', 'peak_vram_total_mb']
            for k in vram_keys:
                vals = [t[k] for t in global_timings if k in t]
                if vals:
                    avg_timing[k] = float(np.max(vals))
            global_summary['avg_timing'] = avg_timing

            print(f"\n  Avg timing ({len(global_timings)} scenes):")
            print(f"    total        = {avg_timing.get('total', 0):.3f}s")
            print(f"    dino         = {avg_timing.get('dino', 0):.3f}s")
            if is_pw:
                print(f"    sim_matrix   = {avg_timing.get('sim_matrix', 0):.3f}s")
                print(f"    init_chunk   = {avg_timing.get('initial_chunking', 0):.3f}s")
                print(f"    1st_infer    = {avg_timing.get('first_chunk_inference', 0):.3f}s")
                print(f"    pseudo_pose  = {avg_timing.get('pseudo_pose_computation', 0):.3f}s")
                print(f"    pw_matrix    = {avg_timing.get('pose_weight_computation', 0):.3f}s")
                print(f"    rechunk      = {avg_timing.get('rechunking_total', 0):.3f}s")
                print(f"    infer_all    = {avg_timing.get('all_chunks_inference', 0):.3f}s")
            else:
                print(f"    sampling     = {avg_timing.get('sampling_total', 0):.3f}s  "
                      f"(sim={avg_timing.get('sampling_sim', 0):.3f}s + init={avg_timing.get('sampling_init', 0):.3f}s + ls={avg_timing.get('sampling_ls', 0):.3f}s)")
                print(f"    transformer  = {avg_timing.get('transformer_total', 0):.3f}s")
            print(f"    alignment    = {avg_timing.get('alignment', 0):.3f}s")
            print(f"\n  Peak VRAM (max across all scenes):")
            print(f"    after dino        = {avg_timing.get('peak_vram_dino_mb', 0):.0f}MB")
            if not is_pw:
                print(f"    after sampling    = {avg_timing.get('peak_vram_sampling_mb', 0):.0f}MB")
                print(f"    after transformer = {avg_timing.get('peak_vram_transformer_mb', 0):.0f}MB")
            print(f"    total             = {avg_timing.get('peak_vram_total_mb', 0):.0f}MB")
        print(f"{'#'*60}")
    else:
        global_summary = None

    # Save results
    save_data = {
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
        'scenes': all_results,
        'global_summary': global_summary,
    }

    pair_suffix = f"_ls{args.local_search_iters}" if args.sampling_method == "random_ls_revsim" else ""
    anchor_suffix = "first_anchor"
    pw_suffix = ""
    if args.poseweight_mode is not None:
        pw_suffix = f"_pw{args.poseweight_mode}_cm{args.combine_mode}"
        if args.tau is not None:
            pw_suffix += f"_tau{args.tau}"
        if args.epsilon is not None:
            pw_suffix += f"_eps{args.epsilon}"
        if args.rechunk_remaining_only:
            pw_suffix += "_remonly"
        pw_suffix += f"_g{args.gamma}"
    name_core = (f"{args.sampling_method}_n{args.n_frames}_c{args.chunk_size}"
                 f"_l0.0{pair_suffix}{anchor_suffix}{pw_suffix}")
    if args.scenes:
        json_path = output_dir / f"depth_{args.scenes[0]}_{name_core}.json"
    else:
        json_path = output_dir / f"depth_{name_core}.json"
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Save chunk assignments as txt
    if chunk_assignments:
        txt_path = json_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            for scene_name, chunk_indices, image_paths in chunk_assignments:
                f.write(f"Scene: {scene_name}\n")
                for k, chunk in enumerate(chunk_indices):
                    sorted_chunk = sorted(chunk)
                    f.write(f"  Chunk {k} ({len(sorted_chunk)} frames):\n")
                    for idx in sorted_chunk:
                        f.write(f"    [{idx:4d}] {os.path.basename(image_paths[idx])}\n")
                f.write("\n")
        print(f"Chunk assignments saved to {txt_path}")


if __name__ == "__main__":
    main()
