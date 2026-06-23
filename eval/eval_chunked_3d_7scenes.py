"""
7-Scenes 3D Reconstruction Evaluation — Chunked Inference (VGGT)

Passes ALL n_frames to a single model(images) call. When n_frames > chunk_size,
VGGT internally:
  1. DINOv2 batch on all images (mini-batched)
  2. Diversity-aware split into K chunks (random + reverse-similarity 2-opt LS)
  3. K sequential transformer + camera_head + depth_head passes
  4. Depth scale alignment across chunks via shared anchor
  5. SE3 alignment across chunks
  → Returns pose_enc [1, S, 9], depth [1, S, H, W, 1], depth_conf [1, S, H, W]

3D metrics pipeline:
  6. Unproject predicted depth → predicted world point cloud
  7. Load GT depth (.depth.proj.png, uint16 mm → meters) → unproject in frame-0 camera coords
  8. Center crop 224×224, filter valid points
  9. Umeyama sim(3) alignment (pred → GT)
  10. ICP registration (Open3D, threshold=0.1m)
  11. Compute accuracy, completion, normal consistency (NC1, NC2)

GT pose format: per-frame .pose.txt
    4x4 camera-to-world (c2w) matrices already in OpenCV convention.
    No OpenGL→OpenCV conversion needed (unlike NRGBD).

GT depth: .depth.proj.png (NOT .depth.png)
    uint16 mm. Invalid where == 0 or >= 10000.

Usage:
    python eval_chunked_3d_7scenes.py \\
        --dataset_dir /path/to/7scenes \\
        --n_frames 500 --chunk_size 50 \\
        --sampling_method da_partitioning --rechunk_remaining_only \\
        --scenes chess fire
"""

import os, sys, json, time, random, logging, warnings
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
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
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.multi_anchor import (
    select_anchors,
    compute_chunk_alignment,
    apply_chunk_alignment,
    multi_anchor_depth_scale,
)

logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers")

torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False


# =============================================================================
# 7-Scenes constants
# =============================================================================
ALL_SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
# Native 640×480, fixed intrinsics
FX_NATIVE = FY_NATIVE = 525.0
CX_NATIVE, CY_NATIVE = 320.0, 240.0
# VGGT output resolution
VGGT_W, VGGT_H = 518, 392
MAX_PTS = 999_999
UMEYAMA_SAMPLE = 30_000
# Center crop (224×224) applied before flattening
CROP_T, CROP_B = 84, 308
CROP_L, CROP_R = 147, 371


# =============================================================================
# 3D Metric utilities
# =============================================================================

def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """Accuracy: mean/median distance from rec_points to nearest gt_points."""
    tree = KDTree(np.asarray(gt_points))
    distances, idx = tree.query(np.asarray(rec_points), workers=-1)
    acc = float(np.mean(distances))
    acc_med = float(np.median(distances))
    if gt_normals is not None and rec_normals is not None:
        nd = np.abs(np.sum(np.asarray(gt_normals)[idx] * np.asarray(rec_normals), axis=-1))
        return acc, acc_med, float(np.mean(nd)), float(np.median(nd))
    return acc, acc_med, None, None


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """Completeness: mean/median distance from gt_points to nearest rec_points."""
    tree = KDTree(np.asarray(rec_points))
    distances, idx = tree.query(np.asarray(gt_points), workers=-1)
    comp = float(np.mean(distances))
    comp_med = float(np.median(distances))
    if gt_normals is not None and rec_normals is not None:
        nd = np.abs(np.sum(np.asarray(gt_normals) * np.asarray(rec_normals)[idx], axis=-1))
        return comp, comp_med, float(np.mean(nd)), float(np.median(nd))
    return comp, comp_med, None, None


# =============================================================================
# Geometric utilities
# =============================================================================

def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Estimate sim(3) transform: dst ≈ s * R @ src + t."""
    assert src.shape == dst.shape and src.ndim == 2
    N = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    Sigma = dst_c.T @ src_c / N
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / N
        s = float((D * S.diagonal()).sum() / var_src)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def unproject_gt_depth(depth_m: np.ndarray, c2w: np.ndarray, K: np.ndarray):
    """Unproject depth map (meters) to world coordinates.

    depth_m: (H, W) float32 — depth in meters
    c2w:     (4, 4) float64 — OpenCV camera-to-world (or relative transform)
    K:       (3, 3) float32 — scaled intrinsics
    Returns: xyz_world (H, W, 3), valid_mask (H, W)
    """
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    valid = (depth_m > 0) & (depth_m < 10.0)
    z = depth_m
    x_cam = (u - K[0, 2]) * z / K[0, 0]
    y_cam = (v - K[1, 2]) * z / K[1, 1]
    xyz_cam = np.stack([x_cam, y_cam, z], axis=-1)
    R, t = c2w[:3, :3].astype(np.float32), c2w[:3, 3].astype(np.float32)
    xyz_world = xyz_cam @ R.T + t
    return xyz_world, valid


# =============================================================================
# 7-Scenes data loading
# =============================================================================

def parse_test_split(dataset_dir, scene):
    """Parse TestSplit.txt to get test sequence names."""
    split_path = os.path.join(dataset_dir, scene, 'TestSplit.txt')
    seqs = []
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num = ''.join(c for c in line if c.isdigit())
            seqs.append(f'seq-{int(num):02d}')
    return sorted(seqs)


def load_7scenes_pose(pose_path):
    """Load and validate a 7-Scenes per-frame pose file."""
    try:
        pose = np.loadtxt(pose_path)
        if pose.shape != (4, 4):
            return None
        if np.all(pose == 0) or np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
            return None
        return pose.astype(np.float64)
    except Exception:
        return None


def load_sequence_frames(scenes_dir, scene, seq):
    """Load all valid (color_path, depth_path, pose_path) triplets for a sequence.

    Uses .depth.proj.png (NOT .depth.png) for 7-Scenes projected depth.
    """
    seq_dir = Path(scenes_dir) / scene / seq
    color_files = sorted(seq_dir.glob('frame-*.color.png'))
    frames = []
    for cf in color_files:
        stem = cf.stem.replace('.color', '')
        dp = seq_dir / f'{stem}.depth.proj.png'
        pp = seq_dir / f'{stem}.pose.txt'
        if dp.exists() and pp.exists():
            frames.append((str(cf), str(dp), str(pp)))
    return frames


def sample_frames(frames, n_frames):
    """Uniformly sample up to n_frames from frames list."""
    actual_n = min(n_frames, len(frames))
    if actual_n == 0:
        return None
    if actual_n < n_frames:
        print(f"  Warning: only {len(frames)} frames available (requested {n_frames})")
    indices = np.linspace(0, len(frames) - 1, actual_n, dtype=int)
    indices = sorted(set(indices))
    return [frames[i] for i in indices]


def scaled_intrinsics():
    """Return 3x3 intrinsics scaled from 640x480 to 518x392."""
    sx, sy = VGGT_W / 640.0, VGGT_H / 480.0
    K = np.array([
        [FX_NATIVE * sx,            0,  CX_NATIVE * sx],
        [           0,   FY_NATIVE * sy,  CY_NATIVE * sy],
        [           0,             0,             1      ],
    ], dtype=np.float32)
    return K


# =============================================================================
# Model
# =============================================================================

def load_model(device, model_path=None, chunk_size=50, lambda_div=0.0):
    """Load VGGT model with depth_head enabled for 3D evaluation.

    Key differences from eval_chunked_pose_7scenes.py:
      - depth_head is KEPT (needed for 3D reconstruction)
      - point_head and track_head are disabled (not needed)
    """
    print(f"Loading VGGT model (chunk_size={chunk_size}, lambda_div={lambda_div}) ...")
    if model_path is None:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    else:
        model = VGGT()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Enable in-model chunked inference
    model.aggregator.sampling_max_frames = chunk_size
    model.aggregator.sampling_lambda_div = lambda_div

    # Disable unused heads for memory — depth_head is KEPT
    model.point_head = None
    model.track_head = None

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()
    return model


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, image_paths, device, dtype):
    """Run in-model chunked inference returning depth + pose for all frames.

    Returns:
        pose_enc:   [S, 9] tensor
        depth:      [1, S, H, W, 1] tensor (CPU float32)
        depth_conf: [1, S, H, W] tensor (CPU float32)
        img_shape:  (H, W) tuple
        num_chunks: int
        timing:     dict
    """
    images = load_and_preprocess_images(image_paths).to(device)
    img_shape = images.shape[-2:]

    torch.cuda.synchronize()
    t_total_start = time.time()

    with torch.no_grad():
        if dtype in (torch.bfloat16, torch.float16):
            images = images.to(dtype)
        with torch.amp.autocast('cuda', dtype=dtype, enabled=True):
            pred = model(images)
            pose_enc = pred['pose_enc'].clone()[0]    # [S, 9]
            depth = pred['depth'].float().clone()      # [1, S, H, W, 1]
            depth_conf = pred['depth_conf'].float().clone()  # [1, S, H, W]

    torch.cuda.synchronize()
    total_time = time.time() - t_total_start

    num_chunks = pred.get('num_chunks', 1)
    timing = pred.get('timing', None)
    if timing is not None:
        timing['total'] = total_time
    else:
        timing = {'total': total_time}
    timing['peak_vram_total_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del images, pred
    torch.cuda.empty_cache()
    return pose_enc, depth, depth_conf, img_shape, num_chunks, timing


# =============================================================================
# Pose-weight 3D inference helpers
# =============================================================================

def _poseweight_allchunk_inference_3d(model, images, rechunked, anchors, device, dtype,
                                      patch_tokens_cpu=None, align_mode="se3"):
    """Run inference on each rechunked chunk using pre-computed patch tokens, return aligned extrinsics + depth."""
    H, W = images.shape[-2:]
    K = len(rechunked)
    S = images.shape[1]

    torch.cuda.synchronize()
    t0 = time.time()

    chunk_se3s = []
    chunk_intrinsics = []
    chunk_depths = []
    chunk_depth_confs = []

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
            with torch.amp.autocast('cuda', enabled=False):
                pose_enc_list = model.camera_head(output_list)
                pe = pose_enc_list[-1][0]
            chunk_imgs = images[:, chunk_indices]
            depth_k, depth_conf_k = model.depth_head(
                output_list, images=chunk_imgs, patch_start_idx=patch_start_idx)

        pe_tensor = pe.unsqueeze(0).float()
        with torch.amp.autocast('cuda', dtype=torch.float64, enabled=True):
            ext, intr = pose_encoding_to_extri_intri(pe_tensor.to(device), (H, W))
        ext = ext[0]
        intr = intr[0]

        add_row = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float64).expand(S_k, 1, 4)
        se3 = torch.cat((ext.double(), add_row), dim=1)

        chunk_se3s.append(se3)
        chunk_intrinsics.append(intr)
        chunk_depths.append(depth_k.float()[0])
        chunk_depth_confs.append(depth_conf_k.float()[0])

        del output_list, pose_enc_list, pe, chunk_imgs, depth_k, depth_conf_k
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t_infer = time.time() - t0

    # Multi-anchor SE(3)/Sim(3) pose alignment + depth scale alignment
    t0 = time.time()
    anchor_set = set(anchors)

    all_extrinsic = torch.zeros(S, 3, 4, device=device, dtype=torch.float64)
    all_intrinsic = torch.zeros(S, 3, 3, device=device, dtype=torch.float64)

    H_d, W_d = chunk_depths[0].shape[1], chunk_depths[0].shape[2]
    all_depth = torch.zeros(S, H_d, W_d, 1, device=device)
    all_depth_conf = torch.zeros(S, H_d, W_d, device=device)

    ref_positions = [rechunked[0].index(a) for a in anchors]
    ref_anchor_se3_list = [chunk_se3s[0][p] for p in ref_positions]
    ref_anchor_depths = [chunk_depths[0][p] for p in ref_positions]

    for k in range(K):
        indices_k = rechunked[k]
        depth_k = chunk_depths[k]
        conf_k = chunk_depth_confs[k]

        if k == 0:
            T_rigid = None
            pose_scale = None
            depth_scale = 1.0
        else:
            src_positions = [indices_k.index(a) for a in anchors]
            src_anchor_se3_list = [chunk_se3s[k][p] for p in src_positions]
            T_rigid, pose_scale = compute_chunk_alignment(
                src_anchor_se3_list, ref_anchor_se3_list, align_mode)
            src_anchor_depths = [depth_k[p] for p in src_positions]
            depth_scale = multi_anchor_depth_scale(ref_anchor_depths, src_anchor_depths)

        for pos, idx in enumerate(indices_k):
            if k == 0 or idx not in anchor_set:
                if k == 0:
                    all_extrinsic[idx] = chunk_se3s[k][pos, :3, :]
                else:
                    aligned = apply_chunk_alignment(T_rigid, pose_scale, chunk_se3s[k][pos])
                    all_extrinsic[idx] = aligned[:3, :]
                all_intrinsic[idx] = chunk_intrinsics[k][pos]
                all_depth[idx] = depth_k[pos] * depth_scale
                all_depth_conf[idx] = conf_k[pos]

    t_align = time.time() - t0

    return (all_extrinsic, all_intrinsic,
            all_depth.unsqueeze(0).cpu(), all_depth_conf.unsqueeze(0).cpu(),
            t_infer, t_align)


def _poseweight_remaining_inference_3d(model, images, rechunked, anchors,
                                        chunk0_se3, chunk0_intrinsic, chunk0_depth, chunk0_depth_conf,
                                        chunk0_indices, device, dtype,
                                        patch_tokens_cpu=None, align_mode="se3"):
    """Reuse chunk0 results, infer only K-1 remaining chunks for 3D evaluation."""
    H, W = images.shape[-2:]
    K = len(rechunked)
    S = images.shape[1]

    ref_anchor_se3 = chunk0_se3[0]

    torch.cuda.synchronize()
    t0 = time.time()

    chunk_se3s = [chunk0_se3]
    chunk_intrinsics = [chunk0_intrinsic]
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
            with torch.amp.autocast('cuda', enabled=False):
                pose_enc_list = model.camera_head(output_list)
                pe = pose_enc_list[-1][0]
            chunk_imgs = images[:, chunk_indices]
            depth_k, depth_conf_k = model.depth_head(
                output_list, images=chunk_imgs, patch_start_idx=patch_start_idx)

        pe_tensor = pe.unsqueeze(0).float()
        with torch.amp.autocast('cuda', dtype=torch.float64, enabled=True):
            ext, intr = pose_encoding_to_extri_intri(pe_tensor.to(device), (H, W))
        ext = ext[0]
        intr = intr[0]

        add_row = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float64).expand(S_k, 1, 4)
        se3 = torch.cat((ext.double(), add_row), dim=1)

        chunk_se3s.append(se3)
        chunk_intrinsics.append(intr)
        chunk_depths.append(depth_k.float()[0])
        chunk_depth_confs.append(depth_conf_k.float()[0])

        del output_list, pose_enc_list, pe, chunk_imgs, depth_k, depth_conf_k
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t_infer = time.time() - t0

    # Alignment (all on GPU)
    t0 = time.time()
    anchor_set = set(anchors)
    H_d, W_d = chunk_depths[0].shape[1], chunk_depths[0].shape[2]
    all_extrinsic = torch.zeros(S, 3, 4, device=device, dtype=torch.float64)
    all_intrinsic = torch.zeros(S, 3, 3, device=device, dtype=torch.float64)
    all_depth = torch.zeros(S, H_d, W_d, 1, device=device)
    all_depth_conf = torch.zeros(S, H_d, W_d, device=device)
    ref_anchor_depth = chunk_depths[0][0]

    for k in range(K):
        indices_k = rechunked[k]
        depth_k = chunk_depths[k]
        conf_k = chunk_depth_confs[k]
        if k == 0:
            T_align = None
            scale_k = 1.0
        else:
            anchor_pos_k = indices_k.index(anchors[0])
            anchor_se3_k = chunk_se3s[k][anchor_pos_k]
            T_align = ref_anchor_se3 @ torch.linalg.inv(anchor_se3_k)
            chunk_anchor_depth = depth_k[anchor_pos_k]
            valid = (ref_anchor_depth.squeeze(-1) > 1e-6) & (chunk_anchor_depth.squeeze(-1) > 1e-6)
            scale_k = float(torch.median(ref_anchor_depth[valid] / chunk_anchor_depth[valid])) if valid.sum() > 0 else 1.0

        for pos, idx in enumerate(indices_k):
            if k == 0 or idx not in anchor_set:
                if k == 0:
                    all_extrinsic[idx] = chunk_se3s[k][pos, :3, :]
                else:
                    aligned = T_align @ chunk_se3s[k][pos]
                    all_extrinsic[idx] = aligned[:3, :]
                all_intrinsic[idx] = chunk_intrinsics[k][pos]
                all_depth[idx] = depth_k[pos] * scale_k
                all_depth_conf[idx] = conf_k[pos]

    t_align = time.time() - t0
    return (all_extrinsic, all_intrinsic,
            all_depth.unsqueeze(0).cpu(), all_depth_conf.unsqueeze(0).cpu(),
            t_infer, t_align)


def run_inference_poseweight_3d(model, image_paths, device, dtype,
                                 gamma=1.0, tau=None,
                                 score_type="revsim", ls_iters=5,
                                 n_anchors=1, chunk_size=50,
                                 rechunk_remaining_only=False,
                                 epsilon=None,
                                 align_mode="se3", anchor_select="uniform",
                                 seed=42):
    """Two-phase pose-weighted chunked inference for 3D evaluation.

    Returns aligned extrinsics, intrinsics, depth, depth_conf instead of SE3.
    """
    timing = {}
    images = load_and_preprocess_images(image_paths).to(device)
    if images.dim() == 4:
        images = images.unsqueeze(0)
    S = images.shape[1]
    img_shape = images.shape[-2:]

    # ---- Phase 1a: DINOv2 on all images ----
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype, enabled=True):
            patch_tokens_cpu, pooled_tokens = model.aggregator.forward_dino(images)
    torch.cuda.synchronize()
    timing['dino'] = time.time() - t0

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
    initial_chunks = [list(ch) for ch in chunks]

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
        chunk0_depth = depth_out0.float()[0]
        chunk0_depth_conf = depth_conf_out0.float()[0]
        del output_list0, pose_enc_list0, chunk0_imgs, depth_out0, depth_conf_out0

    pe0_f = pose_enc_0.unsqueeze(0).float()
    with torch.amp.autocast('cuda', dtype=torch.float64, enabled=True):
        ext0_full, intr0_full = pose_encoding_to_extri_intri(pe0_f.to(device), (H, W))
    ext0_full = ext0_full[0]
    intr0_full = intr0_full[0]
    add_row0 = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float64).expand(S_k0, 1, 4)
    chunk0_se3 = torch.cat((ext0_full.double(), add_row0), dim=1)

    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    timing['first_chunk_inference'] = time.time() - t0

    # ---- Phase 2b: Pseudo-poses for all N frames ----
    t0 = time.time()
    ext_np = ext0_full.cpu().numpy()
    R_w2c = ext_np[:, :3, :3]
    t_w2c = ext_np[:, :3, 3]
    chunk0_positions = np.array([
        -R_w2c[i].T @ t_w2c[i] for i in range(len(t_w2c))
    ])
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
        remaining_chunks = [ch for ch in chunks[1:]]
        remaining_anchors = anchors_init
        rechunked_rest, anchors_new, t_ls_rechunk = rechunk_with_pose_weights(
            remaining_chunks, remaining_anchors, sim_matrix, W_pose,
            score_type=score_type, alpha=0.0,
            n_anchors=n_anchors, local_search_iters=ls_iters,
            epsilon=epsilon,
            anchors_override=precomputed_anchors,
        )
        rechunked = [list(chunk0_indices)] + rechunked_rest
        anchors_new = anchors_init
    else:
        rechunked, anchors_new, t_ls_rechunk = rechunk_with_pose_weights(
            chunks, anchors_init, sim_matrix, W_pose,
            score_type=score_type, alpha=0.0,
            n_anchors=n_anchors, local_search_iters=ls_iters,
            epsilon=epsilon,
            anchors_override=precomputed_anchors,
        )
    timing['rechunking_total'] = time.time() - t0
    timing['rechunking_ls'] = t_ls_rechunk
    rechunked_chunks = [list(ch) for ch in rechunked]

    # ---- Phase 4: Inference + alignment (3D variant) ----
    if rechunk_remaining_only:
        (all_extrinsic, all_intrinsic, depth, depth_conf,
         t_infer, t_align) = _poseweight_remaining_inference_3d(
            model, images, rechunked, anchors_new,
            chunk0_se3, intr0_full, chunk0_depth, chunk0_depth_conf,
            chunk0_indices, device, dtype,
            patch_tokens_cpu=patch_tokens_cpu,
            align_mode=align_mode,
        )
    else:
        (all_extrinsic, all_intrinsic, depth, depth_conf,
         t_infer, t_align) = _poseweight_allchunk_inference_3d(
            model, images, rechunked, anchors_new, device, dtype,
            patch_tokens_cpu=patch_tokens_cpu,
            align_mode=align_mode,
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
    return (all_extrinsic, all_intrinsic, depth, depth_conf,
            img_shape, K, timing, initial_chunks, rechunked_chunks,
            sim_matrix, tau_used)


# =============================================================================
# Per-sequence evaluation
# =============================================================================

def process_sequence(model, frame_triplets, device, dtype, conf_thresh, rng,
                     gamma=1.0, tau=None, epsilon=None,
                     rechunk_remaining_only=False,
                     export_ply=False, ply_prefix=None, seed=42):
    """Process a single 7-Scenes sequence: chunked inference + 3D metric evaluation.

    Args:
        frame_triplets: list of (color_path, depth_path, pose_path)
        export_ply: if True, save GT and predicted point clouds as PLY files
        ply_prefix: path prefix for PLY files (e.g. '/out/chess_seq01')
    Returns:
        dict with 3D metrics, or None on failure
    """
    S = len(frame_triplets)
    if S == 0:
        return None

    color_paths = [t[0] for t in frame_triplets]
    depth_paths = [t[1] for t in frame_triplets]
    pose_paths  = [t[2] for t in frame_triplets]

    if model.aggregator.sampling_method == "da_partitioning":
        chunk_size = model.aggregator.sampling_max_frames
        if chunk_size <= 0:
            chunk_size = 50
        n_anchors = getattr(model.aggregator, 'sampling_n_anchors', 1)
        ls_iters = model.aggregator.sampling_local_search_iters
        align_mode = getattr(model.aggregator, 'sampling_align_mode', 'se3')
        anchor_select = getattr(model.aggregator, 'sampling_anchor_select', 'uniform')

        (all_extrinsic, all_intrinsic, depth_pred, depth_conf,
         img_shape, num_chunks, timing, initial_chunks, rechunked_chunks,
         sim_matrix, tau_used) = run_inference_poseweight_3d(
            model, color_paths, device, dtype,
            gamma=gamma, tau=tau,
            score_type="revsim", ls_iters=ls_iters,
            n_anchors=n_anchors, chunk_size=chunk_size,
            rechunk_remaining_only=rechunk_remaining_only,
            epsilon=epsilon,
            align_mode=align_mode, anchor_select=anchor_select,
            seed=seed)

        extrinsic = all_extrinsic.unsqueeze(0).float()
        intrinsic = all_intrinsic.unsqueeze(0).float()

        pred_world_pts = unproject_depth_map_to_point_map(
            depth_pred.squeeze(0),
            extrinsic.squeeze(0).cpu(),
            intrinsic.squeeze(0).cpu(),
        )
        conf_np = depth_conf.squeeze(0).cpu().numpy()

        del all_extrinsic, all_intrinsic, depth_pred, depth_conf, extrinsic, intrinsic
        torch.cuda.empty_cache()
    else:
        # ---- VGGT inference (in-model chunking handles everything) ----
        pose_enc, depth_pred, depth_conf, img_shape, num_chunks, timing = \
            run_inference(model, color_paths, device, dtype)

        # ---- Unproject predicted depth -> world points ----
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc.unsqueeze(0).float().to(device), img_shape
        )

        pred_world_pts = unproject_depth_map_to_point_map(
            depth_pred.squeeze(0),          # [S, H, W, 1]
            extrinsic.squeeze(0).cpu(),     # [S, 3, 4]
            intrinsic.squeeze(0).cpu(),     # [S, 3, 3]
        )  # (S, H, W, 3) numpy

        conf_np = depth_conf.squeeze(0).cpu().numpy()  # (S, H, W)

        del pose_enc, depth_pred, depth_conf, extrinsic, intrinsic
        torch.cuda.empty_cache()

    # ---- Build paired point clouds ----
    K_scaled = scaled_intrinsics()
    pred_pts_list = []
    gt_pts_list   = []
    colors_list   = []  # RGB colors for PLY export

    # Load frame-0 pose for reference coordinate system
    c2w_0 = load_7scenes_pose(pose_paths[0])
    if c2w_0 is None:
        print("    Frame-0 pose invalid, skipping sequence.")
        return None
    w2c_0 = np.linalg.inv(c2w_0)

    for i in range(S):
        # Load GT depth (.depth.proj.png)
        depth_raw = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)  # (480,640) uint16
        if depth_raw is None:
            continue

        depth_resized = cv2.resize(depth_raw, (VGGT_W, VGGT_H),
                                   interpolation=cv2.INTER_NEAREST)
        depth_m = depth_resized.astype(np.float32) / 1000.0
        # 7Scenes invalid depth: 0 or >= 10000 mm (10m)
        depth_m[(depth_resized == 0) | (depth_resized >= 10000)] = 0.0

        # Load GT pose (per-frame .pose.txt, already OpenCV c2w)
        c2w_i = load_7scenes_pose(pose_paths[i])
        if c2w_i is None:
            continue

        # Unproject GT to frame-0 camera space
        c2w_rel = w2c_0 @ c2w_i
        gt_xyz, valid_gt = unproject_gt_depth(depth_m, c2w_rel, K_scaled)

        # Center crop 224×224
        gt_xyz_crop   = gt_xyz  [CROP_T:CROP_B, CROP_L:CROP_R]
        valid_gt_crop = valid_gt[CROP_T:CROP_B, CROP_L:CROP_R]
        pred_xyz_crop = pred_world_pts[i][CROP_T:CROP_B, CROP_L:CROP_R]
        conf_crop     = conf_np[i]       [CROP_T:CROP_B, CROP_L:CROP_R]

        # Flatten
        gt_pts_flat   = gt_xyz_crop.reshape(-1, 3)
        pred_pts_flat = pred_xyz_crop.reshape(-1, 3)
        valid_gt_flat = valid_gt_crop.reshape(-1)
        conf_flat     = conf_crop.reshape(-1)

        # Filter: intersection of GT valid and pred valid
        mask_a = valid_gt_flat & np.isfinite(gt_pts_flat).all(axis=1)
        mask_b = np.isfinite(pred_pts_flat).all(axis=1)
        if conf_thresh > 0:
            mask_b &= (conf_flat > conf_thresh)

        mask_final = mask_a & mask_b
        if np.sum(mask_final) > 0:
            gt_pts_list.append(gt_pts_flat[mask_final])
            pred_pts_list.append(pred_pts_flat[mask_final])
            if ply_prefix is not None:
                color_img = cv2.imread(color_paths[i])
                if color_img is not None:
                    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                    color_resized = cv2.resize(color_img, (VGGT_W, VGGT_H))
                    color_crop = color_resized[CROP_T:CROP_B, CROP_L:CROP_R]
                    colors_list.append(color_crop.reshape(-1, 3)[mask_final])
                else:
                    colors_list.append(np.full((mask_final.sum(), 3), 128, dtype=np.uint8))

    if not pred_pts_list or not gt_pts_list:
        return None

    pred_pts = np.concatenate(pred_pts_list, axis=0).astype(np.float32)
    gt_pts   = np.concatenate(gt_pts_list,   axis=0).astype(np.float32)

    if len(pred_pts) < 100:
        print(f"    Too few valid points: {len(pred_pts)}")
        return None

    # ---- Umeyama alignment (pred → GT) ----
    n_points = len(pred_pts)
    if n_points > UMEYAMA_SAMPLE:
        idx = rng.choice(n_points, UMEYAMA_SAMPLE, replace=False)
        src_sub = pred_pts[idx]
        dst_sub = gt_pts[idx]
    else:
        src_sub = pred_pts
        dst_sub = gt_pts

    try:
        s_um, R_um, t_um = umeyama_alignment(src_sub, dst_sub, with_scale=True)
        pred_pts = (s_um * (R_um @ pred_pts.T)).T + t_um
    except Exception as e:
        print(f"    Umeyama failed: {e}")
        return None

    # ---- Subsample for ICP & Metrics ----
    if len(pred_pts) > MAX_PTS:
        idx_metrics = rng.choice(len(pred_pts), MAX_PTS, replace=False)
        pred_pts_metrics = pred_pts[idx_metrics].astype(np.float64)
        gt_pts_metrics   = gt_pts[idx_metrics].astype(np.float64)
    else:
        pred_pts_metrics = pred_pts.astype(np.float64)
        gt_pts_metrics   = gt_pts.astype(np.float64)

    # ---- Open3D: ICP + normals ----
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_pts_metrics)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts_metrics)

    try:
        reg = o3d.pipelines.registration.registration_icp(
            pcd_pred, pcd_gt,
            0.1,  # 10cm threshold
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        pcd_pred.transform(reg.transformation)
    except Exception as e:
        print(f"    ICP failed: {e}")

    pcd_pred.estimate_normals()
    pcd_gt.estimate_normals()

    pred_pts_final = np.asarray(pcd_pred.points)
    gt_pts_final   = np.asarray(pcd_gt.points)
    pred_normals   = np.asarray(pcd_pred.normals)
    gt_normals_arr = np.asarray(pcd_gt.normals)

    if len(pred_pts_final) < 10 or len(gt_pts_final) < 10:
        return None

    # ---- Metrics ----
    acc, acc_med, nc1, nc1_med = accuracy(
        gt_pts_final, pred_pts_final, gt_normals_arr, pred_normals)
    comp, comp_med, nc2, nc2_med = completion(
        gt_pts_final, pred_pts_final, gt_normals_arr, pred_normals)

    # ---- Auto PLY export ----
    if ply_prefix is not None:
        cd = (acc + comp) / 2.0
        pred_ply_path = f"{ply_prefix}_{cd:.4f}.ply"
        pcd_pred_export = o3d.geometry.PointCloud()
        pcd_pred_export.points = o3d.utility.Vector3dVector(pred_pts_final)
        if colors_list:
            all_colors = np.concatenate(colors_list, axis=0)
            if len(all_colors) > MAX_PTS:
                all_colors = all_colors[idx_metrics]
            pcd_pred_export.colors = o3d.utility.Vector3dVector(
                all_colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(pred_ply_path, pcd_pred_export)
        print(f"    Saved pred PLY → {pred_ply_path} ({len(pred_pts_final)} pts)")
        gt_ply_path = f"{ply_prefix}_{cd:.4f}_gt.ply"
        pcd_gt_export = o3d.geometry.PointCloud()
        pcd_gt_export.points = o3d.utility.Vector3dVector(gt_pts_final)
        if colors_list:
            pcd_gt_export.colors = o3d.utility.Vector3dVector(
                all_colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(gt_ply_path, pcd_gt_export)
        print(f"    Saved GT PLY  → {gt_ply_path} ({len(gt_pts_final)} pts)")

    return {
        'acc': acc, 'acc_med': acc_med,
        'comp': comp, 'comp_med': comp_med,
        'nc1': nc1, 'nc1_med': nc1_med,
        'nc2': nc2, 'nc2_med': nc2_med,
        'n_pts': int(len(pred_pts_final)),
        'n_frames': S,
        'num_chunks': num_chunks,
        'inference_time_s': round(timing['total'], 2),
        'timing': {k: v for k, v in timing.items()
                   if k != 'transformer_per_chunk'},
    }


# =============================================================================
# Aggregation
# =============================================================================

def _mean_metrics(results_list):
    keys = ['acc', 'acc_med', 'comp', 'comp_med',
            'nc1', 'nc1_med', 'nc2', 'nc2_med', 'inference_time_s']
    out = {}
    for k in keys:
        vals = [r[k] for r in results_list if r and r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="7-Scenes 3D Reconstruction Eval — VGGT in-model FL chunked inference"
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to 7-Scenes dataset root (contains chess/, fire/, ...)")
    parser.add_argument("--n_frames", type=int, default=200,
                        help="Number of frames to uniformly sample per sequence")
    parser.add_argument("--chunk_size", type=int, default=50,
                        help="Max frames per chunk (sampling_max_frames)")
    parser.add_argument("--sampling_method", type=str, default="da_partitioning",
                        choices=["da_partitioning", "random_partitioning", "origin"],
                        help="View partitioning method: "
                             "'da_partitioning' = diversity-aware partitioning (ours) — random "
                             "split refined by 2-opt local search and pose-weighted re-chunking; "
                             "'random_partitioning' = random partitioning without local search (baseline); "
                             "'origin' = no partitioning, single full-sequence pass (baseline).")
    parser.add_argument("--local_search_iters", type=int, default=5,
                        help="2-opt local search iterations for LS methods")
    parser.add_argument("--conf_thresh", type=float, default=0.0,
                        help="Depth confidence threshold")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Specific scenes to evaluate (default: all 7)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Custom model checkpoint (default: facebook/VGGT-1B)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--dino_batch_size", type=int, default=256,
                        help="DINOv2 mini-batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results_chunked_3d_7scenes")
    # Pose-weighted re-chunking args (used by sampling_method=da_partitioning)
    parser.add_argument("--gamma", type=float, default=0.001,
                        help="Softmax temperature for pseudo-pose soft assignment")
    parser.add_argument("--tau", type=float, default=None,
                        help="Distance decay for pose weights (None = auto from median dist)")
    parser.add_argument("--epsilon", type=float, default=0.005,
                        help="Fixed epsilon for combine mode E")
    parser.add_argument("--rechunk_remaining_only", action="store_true",
                        help="Freeze chunk0, re-chunk only remaining frames (reuse chunk0 inference)")
    parser.add_argument("--export_ply", action="store_true",
                        help="Export GT and predicted point clouds as PLY files")
    parser.add_argument("--ply_dir", type=str, default="./ply_output",
                        help="Directory to save PLY files")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]
    device = "cuda"

    scenes = args.scenes if args.scenes else ALL_SCENES
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.export_ply:
        ply_dir = Path(args.ply_dir)
        ply_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(device, args.model_path, args.chunk_size)
    model.aggregator.dino_batch_size = args.dino_batch_size
    # 'random_partitioning' maps to the aggregator's random-balanced split (no local search)
    model.aggregator.sampling_method = (
        "random_balanced" if args.sampling_method == "random_partitioning" else args.sampling_method)
    model.aggregator.sampling_local_search_iters = args.local_search_iters

    # Origin mode: disable chunking
    if args.sampling_method == "origin":
        model.aggregator.sampling_max_frames = 0

    print(f"\n{'#'*60}")
    print(f"# VGGT Chunked 3D Eval (7-Scenes, method={args.sampling_method})")
    print(f"# n_frames={args.n_frames}, chunk_size={args.chunk_size}")
    print(f"# dtype={args.dtype}, conf_thresh={args.conf_thresh}")
    if args.sampling_method == "da_partitioning":
        print(f"# pose-weighted re-chunking: "
              f"gamma={args.gamma}, tau={args.tau}, "
              f"epsilon={args.epsilon}, rechunk_remaining_only={args.rechunk_remaining_only}")
    print(f"{'#'*60}\n")

    scene_summaries = {}
    all_seq_results = []

    for scene in scenes:
        print(f'\n{"#"*60}')
        print(f'# Scene: {scene}')
        print(f'{"#"*60}')

        seqs = parse_test_split(str(dataset_dir), scene)
        seq_results = {}

        for seq in seqs:
            print(f'\n  [{scene}/{seq}]')
            frames = load_sequence_frames(str(dataset_dir), scene, seq)
            sampled = sample_frames(frames, args.n_frames)
            if sampled is None:
                print("    No frames found, skipping.")
                continue
            print(f'    Total={len(frames)}, Sampled={len(sampled)}')

            # Construct PLY prefix for auto-saving
            ply_prefix = None
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                ply_prefix = os.path.join(
                    args.output_dir,
                    f"{args.sampling_method}_{scene}_{seq}_{args.n_frames}_{args.chunk_size}")

            try:
                result = process_sequence(model, sampled, device, dtype,
                                          args.conf_thresh, rng,
                                          gamma=args.gamma,
                                          tau=args.tau,
                                          epsilon=args.epsilon,
                                          rechunk_remaining_only=args.rechunk_remaining_only,
                                          export_ply=False,
                                          ply_prefix=ply_prefix)
            except Exception as e:
                print(f'    ERROR: {e}')
                import traceback; traceback.print_exc()
                result = None

            if result is None:
                print('    Sequence failed.')
                continue

            seq_results[seq] = result
            all_seq_results.append(result)
            print(f'    acc={result["acc"]:.4f}  comp={result["comp"]:.4f}  '
                  f'nc1={result["nc1"]:.4f}  nc2={result["nc2"]:.4f}  '
                  f'K={result["num_chunks"]}  time={result["inference_time_s"]:.2f}s')

        if not seq_results:
            continue

        scene_mean = _mean_metrics(list(seq_results.values()))
        scene_summaries[scene] = {'sequences': seq_results, 'mean': scene_mean}
        print(f'\n  [{scene} MEAN] acc={scene_mean["acc"]:.4f}  comp={scene_mean["comp"]:.4f}  '
              f'nc1={scene_mean["nc1"]:.4f}  nc2={scene_mean["nc2"]:.4f}')

    overall = _mean_metrics(all_seq_results)
    print(f'\n{"="*60}')
    print(f'OVERALL MEAN ({len(all_seq_results)} sequences):')
    if overall.get("acc") is not None:
        print(f'  acc={overall["acc"]:.4f}   acc_med={overall["acc_med"]:.4f}')
        print(f'  comp={overall["comp"]:.4f}  comp_med={overall["comp_med"]:.4f}')
        print(f'  nc1={overall["nc1"]:.4f}   nc1_med={overall["nc1_med"]:.4f}')
        print(f'  nc2={overall["nc2"]:.4f}   nc2_med={overall["nc2_med"]:.4f}')
    print(f'{"="*60}')

    # Save results
    pair_suffix = f"_ls{args.local_search_iters}" if args.sampling_method == "da_partitioning" else ""
    anchor_suffix = "first_anchor"
    pw_suffix = ""
    if args.sampling_method == "da_partitioning":
        pw_suffix = f"_pw_g{args.gamma}"
        if args.epsilon is not None:
            pw_suffix += f"_eps{args.epsilon}"
        if args.rechunk_remaining_only:
            pw_suffix += "_ro"

    name_core = (f"{args.sampling_method}_n{args.n_frames}_c{args.chunk_size}"
                 f"_l0.0{pair_suffix}{anchor_suffix}{pw_suffix}")
    if args.scenes:
        json_name = f"chunked3d_{args.scenes[0]}_{name_core}.json"
    else:
        json_name = f"chunked3d_{name_core}.json"
    json_path = output_dir / json_name

    save_data = {
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
        'scene_summaries': scene_summaries,
        'overall_mean': overall,
    }
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f'\nResults saved → {json_path}')


if __name__ == "__main__":
    main()
