"""
7-Scenes Pose Estimation — Chunked Inference (VGGT)

Pipeline for sampling_method=da_partitioning:
  1. DINOv2 batch on all images (mini-batched)
  2. Diversity-aware split into K chunks (random + reverse-similarity 2-opt LS)
  3. Pose-weighted re-chunking from chunk-0 pseudo-poses
  4. K sequential transformer + camera_head passes
  5. Single-anchor SE3 alignment across chunks

Usage:
    python eval_chunked_pose_7scenes.py \
        --dataset_dir /path/to/7scenes \
        --n_frames 500 \
        --chunk_size 50 \
        --sampling_method da_partitioning \
        --rechunk_remaining_only \
        --scenes chess fire
"""

import os, sys, json, time, random, logging, warnings
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Allow running from the eval/ subdirectory: add repo root (parent of eval/) to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vggt.models.vggt import VGGT
from vggt.utils.rotation import mat_to_quat
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.models.vggt import step_sampling_split
import torch.nn.functional as F
from vggt.models.aggregator import (
    random_balanced_ls_revsim_split,
    compute_pseudo_poses, compute_pose_weight_matrix, rechunk_with_pose_weights,
)

logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False

SEVEN_SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']


# =============================================================================
# Metrics
# =============================================================================

def build_pair_index(N):
    i1, i2 = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    return i1, i2


def rotation_angle(rot_gt, rot_pred, eps=1e-15):
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    return err_q * 180 / np.pi


def translation_angle(tvec_gt, tvec_pred, eps=1e-15):
    t = tvec_pred / (torch.norm(tvec_pred, dim=1, keepdim=True) + eps)
    t_gt = tvec_gt / (torch.norm(tvec_gt, dim=1, keepdim=True) + eps)
    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = 1e6
    deg = err_t * 180.0 / np.pi
    return torch.min(deg, (180 - deg).abs())


def compute_relative_pose_errors(pred_se3, gt_se3, num_frames):
    i1, i2 = build_pair_index(num_frames)
    rel_gt = gt_se3[i1].bmm(closed_form_inverse_se3(gt_se3[i2]))
    rel_pred = pred_se3[i1].bmm(closed_form_inverse_se3(pred_se3[i2]))
    rra = rotation_angle(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    rta = translation_angle(rel_gt[:, :3, 3], rel_pred[:, :3, 3])
    return rra, rta


def calculate_auc(r_errors, t_errors, thresholds=[3, 5, 15, 30]):
    max_errors = np.maximum(r_errors, t_errors)
    auc = {}
    for th in thresholds:
        bins = np.arange(th + 1)
        hist, _ = np.histogram(max_errors, bins=bins)
        norm_hist = hist.astype(float) / float(len(max_errors))
        auc[th] = float(np.mean(np.cumsum(norm_hist)))
    return auc


def compute_ate_metrics(pred_se3_np, gt_c2w_np):
    """Compute ATE (Absolute Trajectory Error) after Umeyama alignment.

    Args:
        pred_se3_np: (N, 4, 4) numpy w2c SE3 matrices (predicted).
        gt_c2w_np: (N, 4, 4) numpy c2w matrices (ground truth).

    Returns:
        dict with ate_rmse, ate_mean, ate_median, ate_max, scale.
    """
    gt_centers = gt_c2w_np[:, :3, 3]
    R = pred_se3_np[:, :3, :3]
    t = pred_se3_np[:, :3, 3]
    pred_centers_raw = -np.einsum('nij,nj->ni', R.transpose(0, 2, 1), t)
    pred_centers, s_align, _, _ = umeyama_alignment(pred_centers_raw, gt_centers)
    errors = np.linalg.norm(pred_centers - gt_centers, axis=-1)
    return {
        'ate_rmse': float(np.sqrt(np.mean(errors ** 2))),
        'ate_mean': float(np.mean(errors)),
        'ate_median': float(np.median(errors)),
        'ate_max': float(np.max(errors)),
        'scale': float(s_align),
    }


def compute_rpe_metrics(pred_se3, gt_se3, image_paths=None):
    """Compute RPE (Relative Pose Error) for consecutive frame pairs.

    Args:
        pred_se3: (N, 4, 4) tensor w2c SE3 matrices (predicted).
        gt_se3: (N, 4, 4) tensor w2c SE3 matrices (ground truth).
        image_paths: optional list of image paths for sorting (ensures meaningful consecutive pairs).

    Returns:
        dict with rpe_trans_mean/median and rpe_rot_mean/median (degrees).
    """
    N = pred_se3.shape[0]
    if N < 2:
        return {'rpe_trans_mean': 0.0, 'rpe_trans_median': 0.0,
                'rpe_rot_mean': 0.0, 'rpe_rot_median': 0.0}

    # Sort by image path for meaningful consecutive pairs
    if image_paths is not None:
        order = np.argsort(image_paths)
        order_t = torch.tensor(order, dtype=torch.long)
        pred_se3 = pred_se3[order_t]
        gt_se3 = gt_se3[order_t]

    # Consecutive pairs
    gt_inv = torch.linalg.inv(gt_se3[1:])
    rel_gt = gt_se3[:-1] @ gt_inv
    pred_inv = torch.linalg.inv(pred_se3[1:])
    rel_pred = pred_se3[:-1] @ pred_inv

    # Relative pose error: E = rel_gt^{-1} @ rel_pred
    rel_err = torch.linalg.inv(rel_gt) @ rel_pred  # (N-1, 4, 4)

    # Translation error (Euclidean norm)
    trans_err = torch.norm(rel_err[:, :3, 3], dim=1)

    # Rotation error (angle from rotation matrix)
    R_err = rel_err[:, :3, :3]
    trace = R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2]
    cos_angle = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    rot_err = torch.acos(cos_angle) * 180.0 / np.pi

    return {
        'rpe_trans_mean': float(trans_err.mean()),
        'rpe_trans_median': float(trans_err.median()),
        'rpe_rot_mean': float(rot_err.mean()),
        'rpe_rot_median': float(rot_err.median()),
    }


def c2w_to_w2c(c2w):
    w2c = np.linalg.inv(c2w)
    return w2c[:3, :]


# =============================================================================
# Chunking quality metrics
# =============================================================================

def compute_chunking_quality_metrics(sim_matrix, chunks, gt_poses_c2w=None):
    """Compute chunking quality metrics from similarity matrix and chunks.

    Args:
        sim_matrix: (N, N) numpy cosine similarity matrix.
        chunks: list of K lists of frame indices.
        gt_poses_c2w: (N, 4, 4) numpy c2w matrices, or None to skip GT-dependent metrics.

    Returns:
        dict of metric name → value.
    """
    from scipy import stats

    N = sim_matrix.shape[0]
    K = len(chunks)
    sim = np.clip(sim_matrix, 0, None)

    # --- Coverage per chunk: f_cov(C_k) = sum_i max_{j in C_k} sim(i,j) ---
    per_chunk_cov = np.array([
        sim[:, chunk].max(axis=1).sum() if len(chunk) > 0 else 0.0
        for chunk in chunks
    ])

    # --- Coverage balance ---
    coverage_profiles = np.zeros((K, N), dtype=np.float32)
    for k, chunk in enumerate(chunks):
        if len(chunk) > 0:
            coverage_profiles[k] = sim[:, chunk].max(axis=1)
    per_frame_var = float(coverage_profiles.var(axis=0).mean())
    chunk_total_var = float(coverage_profiles.sum(axis=1).var())

    # --- Intra-chunk diversity: mean pairwise distance within each chunk ---
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

    metrics = {
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

    # --- GT-dependent metrics (spatial coverage + overlap correlation) ---
    if gt_poses_c2w is not None:
        # Spatial coverage: voxelized camera position occupancy ratio
        grid_resolution = 0.1
        all_positions = gt_poses_c2w[:, :3, 3]
        min_pos = all_positions.min(axis=0)
        voxel_ids_all = ((all_positions - min_pos) / grid_resolution).astype(int)
        total_voxels = len(set(map(tuple, voxel_ids_all)))

        spatial_covs = []
        for chunk in chunks:
            if len(chunk) == 0:
                spatial_covs.append(0.0)
                continue
            chunk_pos = all_positions[chunk]
            voxel_ids = ((chunk_pos - min_pos) / grid_resolution).astype(int)
            spatial_covs.append(len(set(map(tuple, voxel_ids))) / max(total_voxels, 1))
        spatial_cov = np.array(spatial_covs)

        metrics['spatial_coverage_mean'] = float(spatial_cov.mean())
        metrics['spatial_coverage_std'] = float(spatial_cov.std())

        # Overlap correlation: sim vs spatial proximity
        positions = gt_poses_c2w[:, :3, 3]
        directions = -gt_poses_c2w[:, :3, 2]

        diff = positions[:, None, :] - positions[None, :, :]
        trans_dist = np.linalg.norm(diff, axis=2)

        dirs_norm = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
        cos_ang = np.clip(dirs_norm @ dirs_norm.T, -1, 1)
        ang_dist = np.arccos(cos_ang)

        sigma_t = max(np.median(trans_dist[trans_dist > 0]), 1e-8)
        sigma_a = max(np.median(ang_dist[ang_dist > 0]), 1e-8)
        spatial_prox = np.exp(-trans_dist / sigma_t) * np.exp(-ang_dist / sigma_a)

        triu_idx = np.triu_indices(N, k=1)
        sim_flat = sim[triu_idx]
        prox_flat = spatial_prox[triu_idx]

        pearson_r, _ = stats.pearsonr(sim_flat, prox_flat)
        spearman_r, _ = stats.spearmanr(sim_flat, prox_flat)

        # Average Precision
        fov_rad = np.deg2rad(60.0)
        overlap_thresh = sigma_t * 0.5
        trans_flat = trans_dist[triu_idx]
        ang_flat = ang_dist[triu_idx]
        overlap_binary = ((trans_flat < overlap_thresh) & (ang_flat < fov_rad)).astype(np.float32)

        if 0 < overlap_binary.sum() < len(overlap_binary):
            sorted_idx = np.argsort(-sim_flat)
            sorted_labels = overlap_binary[sorted_idx]
            cumsum = np.cumsum(sorted_labels)
            precision_at_k = cumsum / np.arange(1, len(sorted_labels) + 1)
            avg_precision = float((precision_at_k * sorted_labels).sum() / sorted_labels.sum())
        else:
            avg_precision = float('nan')

        metrics['overlap_corr_pearson'] = float(pearson_r)
        metrics['overlap_corr_spearman'] = float(spearman_r)
        metrics['overlap_corr_ap'] = avg_precision

    return metrics


def umeyama_alignment(src, dst):
    """Umeyama alignment: find Sim(3) transform (s, R, t) mapping src → dst.

    Minimizes sum_i || dst_i - (s * R @ src_i + t) ||^2

    Args:
        src: (N, 3) source points (predicted).
        dst: (N, 3) destination points (ground truth).

    Returns:
        aligned_src: (N, 3) transformed source points.
        s, R, t: scale, rotation (3,3), translation (3,).
    """
    assert src.shape == dst.shape
    N = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src_sq = np.mean(np.sum(src_c ** 2, axis=1))
    cov = (dst_c.T @ src_c) / N

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma_src_sq
    t = mu_dst - s * R @ mu_src

    aligned_src = (s * (R @ src.T)).T + t
    return aligned_src, s, R, t


def _find_best_plane(centers, clip_k=5):
    """Find the two axes with the largest range -> best viewing plane."""
    axis_labels = ['X', 'Y', 'Z']
    ranges = []
    for a in range(3):
        vals = np.sort(centers[:, a])
        clipped = vals[clip_k:-clip_k] if len(vals) > 2 * clip_k else vals
        ranges.append(clipped[-1] - clipped[0])
    flat_axis = int(np.argmin(ranges))
    keep = [a for a in range(3) if a != flat_axis]
    print(f"           [vis] axis ranges X={ranges[0]:.2f} Y={ranges[1]:.2f} Z={ranges[2]:.2f}"
          f" -> drop {axis_labels[flat_axis]}, plot {axis_labels[keep[0]]}-{axis_labels[keep[1]]}")
    return keep[0], keep[1]


def _plot_trajectory_on_ax(ax, pred_se3, gt_c2w, title, axis_x=None, axis_y=None,
                            image_paths=None):
    """Plot a single predicted-vs-GT trajectory on a given axes.

    Applies Umeyama alignment, then draws GT (dashed gray) and
    predicted trajectory colored by per-frame position error.
    Frames are sorted by image path before plotting for clean trajectories.

    Args:
        ax: matplotlib Axes to draw on.
        pred_se3: (N, 4, 4) numpy w2c SE3 matrices (predicted).
        gt_c2w: (N, 4, 4) numpy c2w matrices (ground truth).
        title: subplot title string.
        axis_x, axis_y: world axes for 2D projection (None=auto detect).
        image_paths: optional list of image paths for sorting.

    Returns:
        dict with ATE metrics.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection

    gt_centers = gt_c2w[:, :3, 3]

    R = pred_se3[:, :3, :3]
    t = pred_se3[:, :3, 3]
    pred_centers_raw = -np.einsum('nij,nj->ni', R.transpose(0, 2, 1), t)

    pred_centers, s_align, _, _ = umeyama_alignment(pred_centers_raw, gt_centers)

    errors = np.linalg.norm(pred_centers - gt_centers, axis=-1)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))
    ate_median = float(np.median(errors))
    ate_max = float(np.max(errors))

    # Sort by image path for clean trajectory lines
    if image_paths is not None:
        order = np.argsort(image_paths)
        gt_centers = gt_centers[order]
        pred_centers = pred_centers[order]
        errors = errors[order]

    # Auto-detect best viewing plane
    if axis_x is None or axis_y is None:
        axis_x, axis_y = _find_best_plane(gt_centers)

    # GT trajectory
    ax.plot(gt_centers[:, axis_x], gt_centers[:, axis_y],
            '--', color='gray', linewidth=1.2, alpha=0.8, label='gt', zorder=1)

    # Predicted trajectory colored by error
    pts = np.column_stack([pred_centers[:, axis_x], pred_centers[:, axis_y]])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_errors = 0.5 * (errors[:-1] + errors[1:])

    cmap = cm.get_cmap('jet')
    norm = mcolors.Normalize(vmin=0, vmax=max(errors.max(), 1e-6))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.5, zorder=2)
    lc.set_array(seg_errors)
    ax.add_collection(lc)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    import matplotlib.pyplot as plt
    plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)

    axis_labels = ['X', 'Y', 'Z']
    ax.set_xlabel(axis_labels[axis_x], fontsize=8)
    ax.set_ylabel(axis_labels[axis_y], fontsize=8)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
    ax.set_facecolor('#d8d8d8')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title}\nATE RMSE={ate_rmse:.3f}  Mean={ate_mean:.3f}  Med={ate_median:.3f}",
                 fontsize=9, fontweight='bold')

    margin = 0.5
    all_pts = np.concatenate([gt_centers, pred_centers], axis=0)
    ax.set_xlim(all_pts[:, axis_x].min() - margin, all_pts[:, axis_x].max() + margin)
    ax.set_ylim(all_pts[:, axis_y].min() - margin, all_pts[:, axis_y].max() + margin)

    return {'ate_rmse': ate_rmse, 'ate_mean': ate_mean,
            'ate_median': ate_median, 'ate_max': ate_max,
            'scale': float(s_align)}


def save_single_trajectory(pred_se3, gt_c2w, scene, seq_name, vis_dir,
                            sampling_method, n_frames, chunk_size,
                            image_paths=None):
    """Save individual trajectory plot for a single sequence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ate = _plot_trajectory_on_ax(ax, pred_se3, gt_c2w,
                                  title=seq_name, image_paths=image_paths)
    sm = sampling_method or "unknown"
    fig.suptitle(f"{scene}/{seq_name}  {sm}  n{n_frames}_c{chunk_size}",
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(vis_dir, exist_ok=True)
    fname = f'{sm}_{scene}_{seq_name}_{n_frames}_{chunk_size}.png'
    save_path = os.path.join(vis_dir, fname)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"           [vis] saved {save_path}")


def visualize_scene_trajectories(seq_data, scene, vis_dir,
                                  n_frames=None, chunk_size=None,
                                  sampling_method=None):
    """Create a subplot grid of all sequences in a scene.

    Args:
        seq_data: list of (seq_name, pred_se3, gt_c2w[, image_paths]) tuples.
        scene: scene name string.
        vis_dir: output directory.
        n_frames, chunk_size, sampling_method: used in filename.

    Returns:
        dict of seq_name → ATE metrics.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    K = len(seq_data)
    if K == 0:
        return {}

    cols = min(4, K)
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    ate_results = {}
    for i, entry in enumerate(seq_data):
        seq_name, pred_se3, gt_c2w = entry[0], entry[1], entry[2]
        img_paths = entry[3] if len(entry) > 3 else None
        ate = _plot_trajectory_on_ax(axes[i], pred_se3, gt_c2w,
                                     title=seq_name, image_paths=img_paths)
        ate_results[seq_name] = ate
        print(f"    [ATE] {scene}/{seq_name}: "
              f"RMSE={ate['ate_rmse']:.4f}, Mean={ate['ate_mean']:.4f}, "
              f"Median={ate['ate_median']:.4f}, Max={ate['ate_max']:.4f}  "
              f"(scale={ate['scale']:.4f})")

    # Hide unused subplots
    for i in range(K, len(axes)):
        axes[i].set_visible(False)

    # Scene-level mean ATE in suptitle
    mean_rmse = np.mean([v['ate_rmse'] for v in ate_results.values()])
    sm = sampling_method or "unknown"
    nf = n_frames or 0
    cs = chunk_size or 0
    fig.suptitle(f"{scene}  {sm}_n{nf}_c{cs}  (mean ATE RMSE={mean_rmse:.4f})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(vis_dir, exist_ok=True)
    fname = f'traj_{scene}_{sm}_n{nf}_c{cs}.png'
    save_path = os.path.join(vis_dir, fname)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [vis] saved {save_path}")

    return ate_results


# =============================================================================
# Data loading
# =============================================================================

def parse_split(dataset_dir, scene, split='test'):
    """Parse a 7Scenes split file.

    split ∈ {'test', 'train', 'train_half'}:
      - 'test':       TestSplit.txt  (default; matches the standard benchmark)
      - 'train':      TrainSplit.txt
      - 'train_half': TrainSplit.txt, first ⌈N/2⌉ sequences (deterministic).
                      Used for hyperparameter sweeps so the test set stays
                      untouched — selecting hyperparameters on the test split
                      itself would be evaluation cheating.
    """
    if split == 'test':
        fname = 'TestSplit.txt'
    elif split in ('train', 'train_half'):
        fname = 'TrainSplit.txt'
    else:
        raise ValueError(f"Unknown split: {split}")
    split_path = os.path.join(dataset_dir, scene, fname)
    seqs = []
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num = ''.join(c for c in line if c.isdigit())
            seqs.append(f'seq-{int(num):02d}')
    seqs = sorted(seqs)
    if split == 'train_half':
        half_n = max(1, (len(seqs) + 1) // 2)
        seqs = seqs[:half_n]
    return seqs


def load_7scenes_pose(pose_path):
    try:
        pose = np.loadtxt(pose_path)
        if pose.shape != (4, 4):
            return None
        if np.all(pose == 0) or np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
            return None
        return pose
    except Exception:
        return None


def get_sequence_frames(seq_dir):
    frames = []
    color_files = sorted(Path(seq_dir).glob("frame-*.color.png"))
    for cf in color_files:
        basename = cf.name.replace(".color.png", "")
        pose_path = cf.with_name(f"{basename}.pose.txt")
        if pose_path.exists():
            frames.append((basename, str(cf), str(pose_path)))
    return frames


def sample_frames(frames, n_frames):
    if n_frames >= len(frames):
        return frames
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    indices = sorted(set(indices))
    return [frames[i] for i in indices]


# =============================================================================
# Model
# =============================================================================

def load_model(device, model_path=None, chunk_size=50, lambda_div=0.0):
    print(f"Loading VGGT model (chunk_size={chunk_size}, lambda_div={lambda_div}) ...")
    if model_path is None:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    else:
        model = VGGT()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # Enable in-model chunked inference
    model.aggregator.sampling_max_frames = chunk_size
    model.aggregator.sampling_lambda_div = lambda_div

    # Disable unused heads for memory
    model.depth_head = None
    model.point_head = None
    model.track_head = None

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()
    return model


# =============================================================================
# Anchor selection (for multi-anchor alignment)
# =============================================================================

def select_anchors(N, n_anchors, mode="uniform", sim_matrix=None, seed=42):
    """Pick `n_anchors` frame indices from [0, N) for cross-chunk alignment anchors.

    Modes:
        - "uniform": evenly-spaced indices over the input sequence. Always includes
            frame 0 (and frame N-1 when n_anchors >= 2). Equivalent to the legacy
            formula `[round(i * (N-1) / (n-1)) for i in range(n)]`.
        - "fps_sim": farthest-point sampling on the DINO similarity distance
            (1 - sim). Seeded at frame 0, then iteratively picks the frame whose
            min similarity to the already-selected set is smallest. Spreads anchors
            across appearance space.
        - "random": `n_anchors` distinct indices drawn uniformly at random from
            [0, N), seeded by `seed`. Returned sorted (so anchors[0] is the
            smallest selected index — used as the primary anchor at chunk
            position 0). For `n_anchors==1` returns `[0]` to match the
            single-anchor closed-form baseline.
        - "first": first `n_anchors` consecutive frames `[0, 1, ..., n-1]`.
            Anchors are temporally clustered at the start of the sequence,
            providing a tiny spatial baseline. Intended as a *non-diverse*
            control to show that anchor diversity matters for Sim(3) scale
            and rotation conditioning.

    Returned list preserves the convention that anchors[0] is the primary anchor
    (inserted at position 0 of every chunk by `_insert_anchors`). uniform,
    fps_sim, and first guarantee anchors[0] == 0; random does NOT (anchors[0]
    is whatever is the smallest of the n random draws).

    Args:
        N: total number of input frames.
        n_anchors: requested anchor count (clamped to [1, N]).
        mode: "uniform", "fps_sim", "random", or "first".
        sim_matrix: [N, N] numpy or torch similarity matrix. Required for fps_sim.
        seed: RNG seed for "random" mode (ignored otherwise).
    Returns:
        list of int frame indices (length <= n_anchors after dedup).
    """
    n_anchors = max(1, int(n_anchors))
    if n_anchors >= N:
        return list(range(N))

    if mode == "uniform":
        if n_anchors == 1:
            return [0]
        anchors = [round(i * (N - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        return list(dict.fromkeys(anchors))

    if mode == "fps_sim":
        if sim_matrix is None:
            raise ValueError("anchor_select='fps_sim' requires a similarity matrix.")
        if isinstance(sim_matrix, torch.Tensor):
            sim = sim_matrix.detach().float().cpu().numpy()
        else:
            sim = np.asarray(sim_matrix, dtype=np.float64)
        sim = np.clip(sim, 0.0, 1.0)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)

        selected = [0]  # seed at frame 0 so anchors[0] == 0 (matches uniform convention)
        min_dist = dist[0].astype(np.float64).copy()
        min_dist[0] = -np.inf
        while len(selected) < n_anchors:
            j = int(np.argmax(min_dist))
            if min_dist[j] <= -np.inf:
                break  # all candidates exhausted
            selected.append(j)
            min_dist = np.minimum(min_dist, dist[j])
            min_dist[selected] = -np.inf
        return selected

    if mode == "random":
        if n_anchors == 1:
            return [0]
        rng = np.random.default_rng(int(seed))
        selected = rng.choice(N, size=n_anchors, replace=False).tolist()
        selected.sort()
        return [int(x) for x in selected]

    if mode == "first":
        # First n consecutive frames from index 0: [0, 1, ..., n-1].
        # Anchors are temporally clustered → poor spatial baseline for Sim(3) scale fit
        # (used as a "non-diverse" ablation control).
        return list(range(min(n_anchors, N)))

    raise ValueError(f"Unknown anchor_select mode: {mode!r} (expected 'uniform', 'fps_sim', 'random', or 'first')")


def _compute_chunk_alignment(src_anchor_se3_list, dst_anchor_se3_list):
    """Single-anchor rigid SE3 alignment from a chunk's anchor to the reference anchor.

    With one anchor the transform is the exact closed-form T = dst @ inv(src)
    (rigid SE3, no scale).

    Returns:
        T_rigid: (4, 4) rigid transform tensor.
        scale: 0-d tensor (== 1; SE3 has no scale).

    See _apply_chunk_alignment for how to apply (T_rigid, scale) to a pose batch.
    """
    device = src_anchor_se3_list[0].device
    dtype = src_anchor_se3_list[0].dtype
    T_align = dst_anchor_se3_list[0] @ torch.linalg.inv(src_anchor_se3_list[0])
    scale = torch.ones((), device=device, dtype=dtype)
    return T_align, scale


def _apply_chunk_alignment(T_rigid, scale, se3_batch):
    """Apply (T_rigid, scale) alignment to a batch of SE3 matrices.

    Implements: out = T_rigid @ E_with_scaled_t, where E_with_scaled_t has its
    translation column multiplied by `scale`. Rotation stays rigid; scale is
    absorbed into the translation only. For scale=1 this matches the existing
    single-anchor behaviour exactly.

    Args:
        T_rigid: (4, 4) rigid transform.
        scale: 0-d tensor.
        se3_batch: (..., 4, 4) tensor.
    Returns:
        Aligned (..., 4, 4) tensor.
    """
    R = T_rigid[:3, :3]
    t = T_rigid[:3, 3]
    rot_in = se3_batch[..., :3, :3]
    trans_in = se3_batch[..., :3, 3]

    out = torch.zeros_like(se3_batch)
    out[..., :3, :3] = torch.matmul(R, rot_in)
    out[..., :3, 3] = torch.matmul(R, (scale * trans_in).unsqueeze(-1)).squeeze(-1) + t
    out[..., 3, 3] = 1.0
    return out


# =============================================================================
# Pose-weighted re-chunking inference
# =============================================================================

def _poseweight_remaining_inference(model, images, rechunked, anchors,
                                     chunk0_se3, chunk0_indices, device, dtype,
                                     patch_tokens_cpu=None):
    """Reuse chunk0 SE3, infer only remaining K-1 chunks, then SE3-align.

    Args:
        model: VGGT model instance.
        images: (1, S, 3, H, W) tensor on device.
        rechunked: list of K lists [chunk0_indices, ...rest].
        anchors: list of anchor frame indices.
        chunk0_se3: (S_k0, 4, 4) tensor, already-computed SE3 for chunk0.
        chunk0_indices: list of frame indices in chunk0.
        device: torch device.
        dtype: torch dtype.
        patch_tokens_cpu: pre-computed DINOv2 patch tokens on CPU.

    Returns:
        pred_se3: (N, 4, 4) tensor, aligned SE3 predictions.
        t_infer: float, inference time (K-1 chunks only).
        t_align: float, alignment time.
    """
    H, W = images.shape[-2:]
    K = len(rechunked)

    torch.cuda.synchronize()
    t0 = time.time()

    chunk_se3s = [chunk0_se3]  # reuse
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

        pe_tensor = pe.unsqueeze(0).float()
        with torch.amp.autocast('cuda', dtype=torch.float64, enabled=True):
            ext, _ = pose_encoding_to_extri_intri(pe_tensor.to(device), (H, W))
        ext = ext[0]

        add_row = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float64).expand(S_k, 1, 4)
        se3 = torch.cat((ext.double(), add_row), dim=1)
        chunk_se3s.append(se3)

        del output_list, pose_enc_list, pe
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t_infer = time.time() - t0

    # Single-anchor SE3 alignment via the shared anchor
    t0 = time.time()
    anchor_set = set(anchors)

    # Reference anchor poses come from chunk0_se3 (anchors are at known positions in chunk 0).
    ref_anchor_positions = [rechunked[0].index(a) for a in anchors]
    ref_anchor_se3_list = [chunk0_se3[p] for p in ref_anchor_positions]

    aligned_se3 = {}
    # chunk0: directly use (already in reference frame)
    for pos, idx in enumerate(rechunked[0]):
        aligned_se3[idx] = chunk_se3s[0][pos]

    # chunks 1..K-1: align to chunk0
    for k in range(1, K):
        src_positions = [rechunked[k].index(a) for a in anchors]
        src_anchor_se3_list = [chunk_se3s[k][p] for p in src_positions]
        T_rigid, scale = _compute_chunk_alignment(
            src_anchor_se3_list, ref_anchor_se3_list)
        for pos, idx in enumerate(rechunked[k]):
            if idx not in anchor_set:
                aligned_se3[idx] = _apply_chunk_alignment(
                    T_rigid, scale, chunk_se3s[k][pos])

    frame_indices_sorted = sorted(aligned_se3.keys())
    pred_se3 = torch.stack([aligned_se3[i] for i in frame_indices_sorted])
    t_align = time.time() - t0

    return pred_se3, t_infer, t_align


def _poseweight_allchunk_inference(model, images, rechunked, anchors, device, dtype,
                                    patch_tokens_cpu=None):
    """Run inference on each chunk using pre-computed patch tokens and SE3-align via shared anchor.

    Args:
        model: VGGT model instance.
        images: (1, S, 3, H, W) tensor on device.
        rechunked: list of K lists of frame indices (anchors at position 0).
        anchors: list of anchor frame indices.
        device: torch device.
        dtype: torch dtype.
        patch_tokens_cpu: pre-computed DINOv2 patch tokens on CPU.

    Returns:
        pred_se3: (N, 4, 4) tensor, aligned SE3 predictions (sorted by frame index).
        t_infer: float, total inference time.
        t_align: float, alignment time.
    """
    H, W = images.shape[-2:]
    K = len(rechunked)

    torch.cuda.synchronize()
    t0 = time.time()

    chunk_se3s = []
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

        pe_tensor = pe.unsqueeze(0).float()
        with torch.amp.autocast('cuda', dtype=torch.float64, enabled=True):
            ext, _ = pose_encoding_to_extri_intri(pe_tensor.to(device), (H, W))
        ext = ext[0]

        add_row = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float64).expand(S_k, 1, 4)
        se3 = torch.cat((ext.double(), add_row), dim=1)
        chunk_se3s.append(se3)

        del output_list, pose_enc_list, pe
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t_infer = time.time() - t0

    # Single-anchor SE3 alignment via the shared anchor
    t0 = time.time()
    anchor_set = set(anchors)
    ref_anchor_positions = [rechunked[0].index(a) for a in anchors]
    ref_anchor_se3_list = [chunk_se3s[0][p] for p in ref_anchor_positions]

    aligned_se3 = {}
    for k in range(K):
        if k == 0:
            for pos, idx in enumerate(rechunked[k]):
                aligned_se3[idx] = chunk_se3s[k][pos]
        else:
            src_positions = [rechunked[k].index(a) for a in anchors]
            src_anchor_se3_list = [chunk_se3s[k][p] for p in src_positions]
            T_rigid, scale = _compute_chunk_alignment(
                src_anchor_se3_list, ref_anchor_se3_list)
            for pos, idx in enumerate(rechunked[k]):
                if idx not in anchor_set:
                    aligned_se3[idx] = _apply_chunk_alignment(
                        T_rigid, scale, chunk_se3s[k][pos])

    frame_indices_sorted = sorted(aligned_se3.keys())
    pred_se3 = torch.stack([aligned_se3[i] for i in frame_indices_sorted])
    t_align = time.time() - t0

    return pred_se3, t_infer, t_align


def run_inference_poseweight(model, image_paths, device, dtype,
                              gamma=1.0, tau=None,
                              score_type="revsim", ls_iters=5,
                              n_anchors=1, chunk_size=50,
                              rechunk_remaining_only=False,
                              epsilon=None,
                              anchor_select="uniform",
                              seed=42):
    """Two-phase pose-weighted chunked inference (pseudo-pose version).

    Pipeline:
      Phase 1: DINO all -> sim matrix -> initial chunking (step_ls_revsim)
      Phase 2: 1st chunk inference -> extract positions -> pseudo-poses
      Phase 3: Compute W_pose -> combine with appearance -> re-chunk via 2-opt LS
      Phase 4: Remaining-chunk inference + SE3 alignment

    If rechunk_remaining_only=True:
      - chunk0 result is kept as-is (no re-assignment of chunk0 frames)
      - Only remaining N-|chunk0| frames are re-chunked into K-1 chunks
      - chunk0 SE3 from Phase 2 is reused; only K-1 chunks are inferred in Phase 4

    Returns:
        (pred_se3, img_shape, K, timing, initial_chunks, rechunked_chunks,
         sim_matrix, tau_used)
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

    # ---- Phase 1c: Anchor selection (uniform / fps_sim / random) ----
    precomputed_anchors = select_anchors(S, n_anchors, mode=anchor_select,
                                         sim_matrix=sim_matrix, seed=seed)
    timing['anchor_select'] = anchor_select

    # ---- Phase 1d: Initial chunking (random + revsim LS) ----
    t0 = time.time()
    chunks, anchors_init, t_ls_initial = random_balanced_ls_revsim_split(
        sim_matrix, chunk_size, n_anchors=n_anchors,
        local_search_iters=ls_iters, seed=seed,
        anchors_override=precomputed_anchors,
    )
    timing['initial_chunking'] = time.time() - t0
    timing['initial_ls'] = t_ls_initial
    initial_chunks = [list(ch) for ch in chunks]  # deep copy for vis

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
        del output_list0, pose_enc_list0

    # Decode chunk0 SE3 (needed for both pseudo-pose extraction and reuse)
    pe0_f = pose_enc_0.unsqueeze(0).float()
    with torch.amp.autocast('cuda', dtype=torch.float64, enabled=True):
        ext0_full, _ = pose_encoding_to_extri_intri(pe0_f.to(device), (H, W))
    ext0_full = ext0_full[0]
    add_row0 = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float64).expand(S_k0, 1, 4)
    chunk0_se3 = torch.cat((ext0_full.double(), add_row0), dim=1)

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
        chunk0_set = set(chunk0_indices)
        remaining_chunks = [ch for ch in chunks[1:]]  # K-1 chunks
        remaining_anchors = anchors_init  # anchor is shared

        rechunked_rest, anchors_new, t_ls_rechunk = rechunk_with_pose_weights(
            remaining_chunks, remaining_anchors, sim_matrix, W_pose,
            score_type=score_type, alpha=0.0,
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
            n_anchors=n_anchors, local_search_iters=ls_iters,
            epsilon=epsilon,
            anchors_override=precomputed_anchors,
        )
    timing['rechunking_total'] = time.time() - t0
    timing['rechunking_ls'] = t_ls_rechunk
    rechunked_chunks = [list(ch) for ch in rechunked]

    # ---- Phase 4 & 5: Inference + SE3 alignment ----
    if rechunk_remaining_only:
        # Reuse chunk0 SE3; only infer K-1 remaining chunks
        pred_se3, t_infer, t_align = _poseweight_remaining_inference(
            model, images, rechunked, anchors_new, chunk0_se3,
            chunk0_indices, device, dtype,
            patch_tokens_cpu=patch_tokens_cpu,
        )
    else:
        pred_se3, t_infer, t_align = _poseweight_allchunk_inference(
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
    return (pred_se3, img_shape, K, timing,
            initial_chunks, rechunked_chunks, sim_matrix, tau_used)


def run_inference(model, image_paths, device, dtype):
    images = load_and_preprocess_images(image_paths).to(device)
    img_shape = images.shape[-2:]

    torch.cuda.synchronize()
    t_total_start = time.time()

    with torch.no_grad():
        if dtype in (torch.bfloat16, torch.float16):
            images = images.to(dtype)
        with torch.amp.autocast('cuda', dtype=dtype, enabled=True):
            pred = model(images)
            pose_enc = pred['pose_enc'].clone()[0]  # [S, 9]

    torch.cuda.synchronize()
    total_time = time.time() - t_total_start

    # Log chunking info if available
    num_chunks = pred.get('num_chunks', 1)
    anchor = pred.get('anchor', None)
    anchors = pred.get('anchors', [anchor] if anchor is not None else None)
    timing = pred.get('timing', None)
    sim_matrix = pred.get('sim_matrix', None)
    chunk_frame_indices = pred.get('chunk_frame_indices', None)
    if timing is not None:
        timing['total'] = total_time
    else:
        timing = {'total': total_time}
    timing['peak_vram_total_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del images, pred
    torch.cuda.empty_cache()
    return pose_enc, img_shape, num_chunks, anchor, anchors, timing, sim_matrix, chunk_frame_indices


# =============================================================================
# Evaluation per sequence
# =============================================================================

def evaluate_sequence(seq_dir, model, device, dtype, n_frames, thresholds,
                      gamma=0.001, tau=None, epsilon=None,
                      rechunk_remaining_only=False, seed=42):
    all_frames = get_sequence_frames(seq_dir)
    if len(all_frames) < 2:
        print(f"  Skipping {seq_dir}: too few frames ({len(all_frames)})")
        return None

    selected = sample_frames(all_frames, n_frames)

    # Load GT poses and filter invalid
    valid = []
    gt_c2w = {}
    for i, (basename, color_path, pose_path) in enumerate(selected):
        pose = load_7scenes_pose(pose_path)
        if pose is not None:
            gt_c2w[i] = pose
            valid.append(i)

    if len(valid) < 2:
        print(f"  Skipping {seq_dir}: too few valid GT poses ({len(valid)})")
        return None

    image_paths = [selected[i][1] for i in valid]
    N = len(valid)

    # GT extrinsics (w2c 3x4) → 4x4
    gt_ext = np.stack([c2w_to_w2c(gt_c2w[i]) for i in valid], axis=0)
    gt_ext = torch.from_numpy(gt_ext).to(device, dtype=torch.float64)
    add_row = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float64).expand(N, 1, 4)
    gt_se3 = torch.cat((gt_ext, add_row), dim=1)

    # GT c2w matrix for poseweight GT mode
    gt_c2w_arr = np.stack([gt_c2w[i] for i in valid], axis=0)  # (N, 4, 4)

    # ------ Pose-weighted re-chunking (da_partitioning) ------
    if model.aggregator.sampling_method == "da_partitioning":
        chunk_size = model.aggregator.sampling_max_frames
        if chunk_size <= 0:
            chunk_size = 50
        n_anchors = getattr(model.aggregator, 'sampling_n_anchors', 1)
        ls_iters = model.aggregator.sampling_local_search_iters
        anchor_select = getattr(model.aggregator, 'sampling_anchor_select', 'uniform')

        (pred_se3, img_shape, num_chunks, timing,
         initial_chunks, rechunked_chunks, sim_matrix, tau_used) = \
            run_inference_poseweight(
                model, image_paths, device, dtype,
                gamma=gamma, tau=tau,
                score_type="revsim", ls_iters=ls_iters,
                n_anchors=n_anchors, chunk_size=chunk_size,
                rechunk_remaining_only=rechunk_remaining_only,
                epsilon=epsilon,
                anchor_select=anchor_select,
                seed=seed)

        rra, rta = compute_relative_pose_errors(pred_se3, gt_se3, N)
        rra_np = rra.cpu().numpy()
        rta_np = rta.cpu().numpy()
        auc = calculate_auc(rra_np, rta_np, thresholds)

        # Chunking quality metrics for both initial and rechunked
        chunking_metrics_initial = None
        chunking_metrics_rechunked = None
        if sim_matrix is not None:
            chunking_metrics_initial = compute_chunking_quality_metrics(
                sim_matrix, initial_chunks, gt_c2w_arr)
            chunking_metrics_rechunked = compute_chunking_quality_metrics(
                sim_matrix, rechunked_chunks, gt_c2w_arr)

        # Per-chunk evaluation
        per_chunk_eval = []
        if rechunked_chunks and len(rechunked_chunks) > 1:
            for k, chunk_idx_list in enumerate(rechunked_chunks):
                if len(chunk_idx_list) < 2:
                    per_chunk_eval.append({
                        'chunk_id': k, 'n_frames': len(chunk_idx_list),
                        'auc': None, 'mean_rra': None, 'mean_rta': None,
                    })
                    continue
                cidx = sorted(chunk_idx_list)
                c_pred = pred_se3[cidx]
                c_gt = gt_se3[cidx]
                c_rra, c_rta = compute_relative_pose_errors(c_pred, c_gt, len(cidx))
                c_rra_np = c_rra.cpu().numpy()
                c_rta_np = c_rta.cpu().numpy()
                c_auc = calculate_auc(c_rra_np, c_rta_np, thresholds)
                per_chunk_eval.append({
                    'chunk_id': k,
                    'n_frames': len(cidx),
                    'auc': c_auc,
                    'mean_rra': float(np.mean(c_rra_np)),
                    'mean_rta': float(np.mean(c_rta_np)),
                })

        _pred_se3_np = pred_se3.cpu().numpy()

        del pred_se3, gt_ext, add_row, gt_se3, rra, rta
        torch.cuda.empty_cache()

        return {
            'auc': auc,
            'mean_rra': float(np.mean(rra_np)),
            'mean_rta': float(np.mean(rta_np)),
            'n_frames': N,
            'n_total_frames': len(all_frames),
            'num_chunks': num_chunks,
            'anchor': 0,
            'anchors': [0],
            'timing': timing,
            'chunking_metrics': chunking_metrics_rechunked,
            'chunking_metrics_initial': chunking_metrics_initial,
            'rra': rra_np.tolist(),
            'rta': rta_np.tolist(),
            'per_chunk_eval': per_chunk_eval,
            'chunk_frame_indices': rechunked_chunks,
            'initial_chunk_frame_indices': initial_chunks,
            'tau_used': tau_used,
            'image_paths': image_paths,
            'pred_se3': _pred_se3_np,
            'gt_c2w': gt_c2w_arr,
        }

    # Normal inference — model internally handles chunking
    pose_enc, img_shape, num_chunks, anchor, anchors, timing, sim_matrix, chunk_frame_indices = \
        run_inference(model, image_paths, device, dtype)

    # Decode poses
    pe_tensor = pose_enc.unsqueeze(0).float()
    H, W = img_shape
    with torch.amp.autocast('cuda', dtype=torch.float64, enabled=True):
        ext, _ = pose_encoding_to_extri_intri(pe_tensor.to(device), (H, W))
        pred_ext = ext[0]

    pred_se3 = torch.cat((pred_ext.double(), add_row), dim=1)

    rra, rta = compute_relative_pose_errors(pred_se3, gt_se3, N)
    rra_np = rra.cpu().numpy()
    rta_np = rta.cpu().numpy()
    auc = calculate_auc(rra_np, rta_np, thresholds)

    # Per-chunk evaluation
    per_chunk_eval = []
    if chunk_frame_indices is not None and len(chunk_frame_indices) > 1:
        for k, chunk_idx_list in enumerate(chunk_frame_indices):
            if len(chunk_idx_list) < 2:
                per_chunk_eval.append({
                    'chunk_id': k, 'n_frames': len(chunk_idx_list),
                    'auc': None, 'mean_rra': None, 'mean_rta': None,
                })
                continue
            cidx = sorted(chunk_idx_list)
            c_pred = pred_se3[cidx]
            c_gt = gt_se3[cidx]
            c_rra, c_rta = compute_relative_pose_errors(c_pred, c_gt, len(cidx))
            c_rra_np = c_rra.cpu().numpy()
            c_rta_np = c_rta.cpu().numpy()
            c_auc = calculate_auc(c_rra_np, c_rta_np, thresholds)
            per_chunk_eval.append({
                'chunk_id': k,
                'n_frames': len(cidx),
                'auc': c_auc,
                'mean_rra': float(np.mean(c_rra_np)),
                'mean_rta': float(np.mean(c_rta_np)),
            })

    chunking_metrics = None
    if sim_matrix is not None and chunk_frame_indices is not None:
        gt_poses_c2w = np.stack([gt_c2w[i] for i in valid], axis=0)
        chunking_metrics = compute_chunking_quality_metrics(
            sim_matrix, chunk_frame_indices, gt_poses_c2w
        )

    # ATE / RPE metrics
    pred_se3_np = pred_se3.cpu().numpy()
    ate_metrics = compute_ate_metrics(pred_se3_np, gt_c2w_arr)
    rpe_metrics = compute_rpe_metrics(pred_se3, gt_se3, image_paths=image_paths)

    return {
        'auc': auc,
        'mean_rra': float(np.mean(rra_np)),
        'mean_rta': float(np.mean(rta_np)),
        'n_frames': N,
        'n_total_frames': len(all_frames),
        'num_chunks': num_chunks,
        'anchor': anchor,
        'anchors': anchors,
        'timing': timing,
        'chunking_metrics': chunking_metrics,
        'rra': rra_np.tolist(),
        'rta': rta_np.tolist(),
        'ate': ate_metrics,
        'rpe': rpe_metrics,
        'per_chunk_eval': per_chunk_eval,
        'chunk_frame_indices': chunk_frame_indices,
        'image_paths': image_paths,
        'pred_se3': pred_se3_np,
        'gt_c2w': gt_c2w_arr,
    }


# =============================================================================
# Main
# =============================================================================

def run_evaluation(model, args, dtype, device, scenes, thresholds):
    """Run full evaluation across all scenes/sequences.

    Returns:
        all_results: dict of scene → {sequences, summary}
        global_summary: dict with global AUC
    """
    mode_label = "baseline"

    all_results = {}
    global_rra = []
    global_rta = []
    global_timings = []

    for scene in scenes:
        print(f"\n{'='*60}")
        print(f"Scene: {scene}  [mode={mode_label}]")
        print(f"{'='*60}")

        test_seqs = parse_split(args.dataset_dir, scene, split='test')
        scene_rra = []
        scene_rta = []
        scene_results = {}
        scene_timings = []
        scene_pose_data = []  # for trajectory visualization

        for seq_name in test_seqs:
            seq_dir = os.path.join(args.dataset_dir, scene, seq_name)
            if not os.path.isdir(seq_dir):
                print(f"  {seq_name}: directory not found, skipping")
                continue

            print(f"  {seq_name}: ", end="", flush=True)
            result = evaluate_sequence(
                seq_dir, model, device, dtype, args.n_frames, thresholds,
                gamma=getattr(args, 'gamma', 0.001),
                tau=getattr(args, 'tau', None),
                epsilon=getattr(args, 'epsilon', None),
                rechunk_remaining_only=getattr(args, 'rechunk_remaining_only', False),
                seed=getattr(args, 'seed', 42),
            )

            if result is None:
                print("skipped")
                continue

            auc = result['auc']
            auc_str = ", ".join([f"AUC@{t}={auc[t]:.4f}" for t in thresholds])
            chunk_info = f"K={result['num_chunks']}"

            t = result['timing']
            if t is not None and 'first_chunk_inference' in t:
                # Pose-weight timing format
                timing_str = (
                    f"total={t['total']:.2f}s  "
                    f"[dino={t['dino']:.2f}s, sim={t['sim_matrix']:.3f}s, "
                    f"init_chunk={t['initial_chunking']:.3f}s (ls={t['initial_ls']:.3f}s), "
                    f"1st_infer={t['first_chunk_inference']:.2f}s, "
                    f"pseudo_pose={t['pseudo_pose_computation']:.3f}s, "
                    f"pose_wt={t['pose_weight_computation']:.3f}s, "
                    f"rechunk={t['rechunking_total']:.3f}s (ls={t['rechunking_ls']:.3f}s), "
                    f"all_infer={t['all_chunks_inference']:.2f}s, "
                    f"align={t['alignment']:.3f}s]"
                )
                if result.get('tau_used') is not None:
                    timing_str += f"  tau={result['tau_used']:.4f}"
                scene_timings.append(t)
            elif t is not None and 'dino' in t:
                timing_str = (f"total={t['total']:.2f}s  "
                              f"[dino={t['dino']:.2f}s, "
                              f"sampling={t['sampling_total']:.3f}s "
                              f"(sim={t['sampling_sim']:.3f}s + init={t.get('sampling_init', t['sampling_fl']):.3f}s + ls={t.get('sampling_ls', 0):.3f}s), "
                              f"transformer={t['transformer_total']:.2f}s, "
                              f"align={t['alignment']:.3f}s]")
                scene_timings.append(t)
            elif t is not None:
                timing_str = f"total={t['total']:.2f}s (single-batch, no chunking)"
                scene_timings.append(t)
            else:
                timing_str = "total=N/A"

            print(f"{auc_str}  (n={result['n_frames']}/{result['n_total_frames']}, "
                  f"{chunk_info})")
            print(f"           {timing_str}")

            # ATE / RPE metrics (shuffle modes)
            if result.get('ate'):
                ate = result['ate']
                print(f"           [ATE] RMSE={ate['ate_rmse']:.4f}, "
                      f"Mean={ate['ate_mean']:.4f}, Median={ate['ate_median']:.4f}, "
                      f"Max={ate['ate_max']:.4f}  (scale={ate['scale']:.4f})")
            if result.get('rpe'):
                rpe = result['rpe']
                print(f"           [RPE] Trans: Mean={rpe['rpe_trans_mean']:.4f}, "
                      f"Median={rpe['rpe_trans_median']:.4f}  |  "
                      f"Rot: Mean={rpe['rpe_rot_mean']:.2f}°, "
                      f"Median={rpe['rpe_rot_median']:.2f}°")

            # Per-chunk evaluation
            if result.get('per_chunk_eval'):
                for ce in result['per_chunk_eval']:
                    if ce['auc'] is not None:
                        ce_auc_str = ", ".join([f"AUC@{t}={ce['auc'][t]:.4f}" for t in thresholds])
                        print(f"           Chunk {ce['chunk_id']} (n={ce['n_frames']:>3}): "
                              f"{ce_auc_str}  "
                              f"(RRA={ce['mean_rra']:.2f}, RTA={ce['mean_rta']:.2f})")
                    else:
                        print(f"           Chunk {ce['chunk_id']} (n={ce['n_frames']:>3}): too few frames")

            # Serialize timing
            timing_save = None
            if t is not None:
                timing_save = {k: v for k, v in t.items() if k != 'transformer_per_chunk'}
                timing_save['transformer_per_chunk'] = t.get('transformer_per_chunk', [])

            scene_results[seq_name] = {
                'auc': auc,
                'mean_rra': result['mean_rra'],
                'mean_rta': result['mean_rta'],
                'n_frames': result['n_frames'],
                'n_total_frames': result['n_total_frames'],
                'num_chunks': result['num_chunks'],
                'anchor': result.get('anchor'),
                'anchors': result.get('anchors'),
                'ate': result.get('ate'),
                'rpe': result.get('rpe'),
                'tau_used': result.get('tau_used'),
                'timing': timing_save,
                'chunking_metrics': result.get('chunking_metrics'),
                'chunk_frame_indices': result.get('chunk_frame_indices'),
                'per_chunk_eval': result.get('per_chunk_eval', []),
            }
            scene_rra.append(np.array(result['rra']))
            scene_rta.append(np.array(result['rta']))

            if result.get('pred_se3') is not None and result.get('gt_c2w') is not None:
                img_paths = result.get('image_paths')
                scene_pose_data.append((seq_name, result['pred_se3'], result['gt_c2w'], img_paths))

                # Per-sequence trajectory visualization (always save)
                traj_dir = str(Path(args.output_dir) / 'trajectories')
                save_single_trajectory(
                    result['pred_se3'], result['gt_c2w'],
                    scene, seq_name, traj_dir,
                    sampling_method=args.sampling_method,
                    n_frames=args.n_frames,
                    chunk_size=args.chunk_size,
                    image_paths=img_paths,
                )

        # Scene-level trajectory visualization
        traj_dir = str(Path(args.output_dir) / 'trajectories')
        if scene_pose_data:
            visualize_scene_trajectories(
                scene_pose_data, scene, traj_dir,
                n_frames=args.n_frames,
                chunk_size=args.chunk_size,
                sampling_method=args.sampling_method,
            )

        if scene_rra:
            rra_all = np.concatenate(scene_rra)
            rta_all = np.concatenate(scene_rta)
            scene_auc = calculate_auc(rra_all, rta_all, thresholds)
            scene_summary = {
                'auc': scene_auc,
                'mean_rra': float(np.mean(rra_all)),
                'mean_rta': float(np.mean(rta_all)),
                'n_pairs': len(rra_all),
            }
            auc_str = ", ".join([f"AUC@{t}={scene_auc[t]:.4f}" for t in thresholds])
            print(f"  --- {scene} overall: {auc_str} "
                  f"(mean_rra={scene_summary['mean_rra']:.2f}, mean_rta={scene_summary['mean_rta']:.2f})")

            # Aggregate ATE/RPE across sequences
            seq_ate = [scene_results[s]['ate'] for s in scene_results if scene_results[s].get('ate')]
            seq_rpe = [scene_results[s]['rpe'] for s in scene_results if scene_results[s].get('rpe')]
            if seq_ate:
                scene_ate = {k: float(np.mean([a[k] for a in seq_ate])) for k in seq_ate[0]}
                scene_summary['ate'] = scene_ate
                print(f"           [ATE] RMSE={scene_ate['ate_rmse']:.4f}, "
                      f"Mean={scene_ate['ate_mean']:.4f}, Median={scene_ate['ate_median']:.4f}")
            if seq_rpe:
                scene_rpe = {k: float(np.mean([r[k] for r in seq_rpe])) for k in seq_rpe[0]}
                scene_summary['rpe'] = scene_rpe
                print(f"           [RPE] Trans: Mean={scene_rpe['rpe_trans_mean']:.4f}, "
                      f"Median={scene_rpe['rpe_trans_median']:.4f}  |  "
                      f"Rot: Mean={scene_rpe['rpe_rot_mean']:.2f}°, "
                      f"Median={scene_rpe['rpe_rot_median']:.2f}°")

            if scene_timings:
                # Avg timing across sequences (same format as nrgbd/bonn)
                timing_keys = ['total', 'dino', 'sampling_sim', 'sampling_fl', 'sampling_total',
                               'transformer_total', 'alignment',
                               # Pose-weight keys (will be skipped if not present)
                               'sim_matrix', 'initial_chunking', 'initial_ls',
                               'first_chunk_inference', 'pseudo_pose_computation',
                               'pose_weight_computation', 'rechunking_total', 'rechunking_ls',
                               'all_chunks_inference']
                avg_timing = {}
                for k in timing_keys:
                    vals = [st[k] for st in scene_timings if k in st]
                    if vals:
                        avg_timing[k] = float(np.mean(vals))
                vram_keys = ['peak_vram_dino_mb', 'peak_vram_sampling_mb',
                             'peak_vram_transformer_mb', 'peak_vram_total_mb']
                for k in vram_keys:
                    vals = [st[k] for st in scene_timings if k in st]
                    if vals:
                        avg_timing[k] = float(np.max(vals))
                # Also keep sampling_method if present
                for st in scene_timings:
                    if 'sampling_method' in st:
                        avg_timing['sampling_method'] = st['sampling_method']
                        break
                scene_summary['timing'] = avg_timing
                print(f"           avg total time: {avg_timing.get('total', 0):.2f}s")
                global_timings.extend(scene_timings)

            global_rra.append(rra_all)
            global_rta.append(rta_all)
        else:
            scene_summary = None

        all_results[scene] = {
            'sequences': scene_results,
            'summary': scene_summary,
        }

    # Global summary: mean of per-scene AUCs
    scene_aucs = {s: all_results[s]['summary']['auc']
                  for s in all_results if all_results[s]['summary'] is not None}
    if scene_aucs:
        global_auc = {}
        for t in thresholds:
            global_auc[t] = float(np.mean([scene_aucs[s][t] for s in scene_aucs]))
        global_summary = {
            'auc': global_auc,
            'num_scenes': len(scene_aucs),
            'per_scene': {s: scene_aucs[s] for s in scene_aucs},
        }
        auc_str = ", ".join([f"AUC@{t}={global_auc[t]:.4f}" for t in thresholds])
        print(f"\n  [{mode_label}] GLOBAL: {auc_str}  ({len(scene_aucs)} scenes)")

        # Global ATE/RPE (mean of per-scene means)
        scene_ates = {s: all_results[s]['summary'].get('ate')
                      for s in all_results if all_results[s]['summary'] is not None
                      and all_results[s]['summary'].get('ate') is not None}
        scene_rpes = {s: all_results[s]['summary'].get('rpe')
                      for s in all_results if all_results[s]['summary'] is not None
                      and all_results[s]['summary'].get('rpe') is not None}
        if scene_ates:
            global_ate = {k: float(np.mean([scene_ates[s][k] for s in scene_ates]))
                          for k in list(scene_ates.values())[0]}
            global_summary['ate'] = global_ate
            print(f"           [ATE] RMSE={global_ate['ate_rmse']:.4f}, "
                  f"Mean={global_ate['ate_mean']:.4f}, Median={global_ate['ate_median']:.4f}")
        if scene_rpes:
            global_rpe = {k: float(np.mean([scene_rpes[s][k] for s in scene_rpes]))
                          for k in list(scene_rpes.values())[0]}
            global_summary['rpe'] = global_rpe
            print(f"           [RPE] Trans: Mean={global_rpe['rpe_trans_mean']:.4f}, "
                  f"Median={global_rpe['rpe_trans_median']:.4f}  |  "
                  f"Rot: Mean={global_rpe['rpe_rot_mean']:.2f}°, "
                  f"Median={global_rpe['rpe_rot_median']:.2f}°")

        if global_timings:
            timing_keys = ['total', 'dino', 'sampling_sim', 'sampling_fl', 'sampling_ls', 'sampling_init', 'sampling_total',
                           'transformer_total', 'alignment',
                           # Pose-weight keys (will be skipped if not present)
                           'sim_matrix', 'initial_chunking', 'initial_ls',
                           'first_chunk_inference', 'pseudo_pose_computation',
                           'pose_weight_computation', 'rechunking_total', 'rechunking_ls',
                           'all_chunks_inference']
            avg_timing = {}
            for k in timing_keys:
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
    else:
        global_summary = None

    return all_results, global_summary


def main():
    parser = argparse.ArgumentParser(description="7-Scenes pose eval — VGGT chunked inference (DA-VGGT)")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--n_frames", type=int, default=200, help="Total frames to sample per sequence")
    parser.add_argument("--chunk_size", type=int, default=50, help="Max frames per chunk (sampling_max_frames)")
    parser.add_argument("--sampling_method", type=str, default="da_partitioning",
                        choices=["da_partitioning", "random_partitioning", "origin"],
                        help="View partitioning method: "
                             "'da_partitioning' = diversity-aware partitioning (ours) — random "
                             "split refined by 2-opt local search and pose-weighted re-chunking; "
                             "'random_partitioning' = random partitioning without local search (baseline); "
                             "'origin' = no partitioning, single full-sequence pass (baseline).")
    parser.add_argument("--local_search_iters", type=int, default=5,
                        help="2-opt local search iterations for da_partitioning (0=skip)")
    parser.add_argument("--scenes", nargs="+", default=None, help="Scenes to evaluate (default: all)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--dino_batch_size", type=int, default=256, help="DINOv2 mini-batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results_chunked_7scenes")
    # Pose-weighted re-chunking (used by sampling_method=da_partitioning)
    parser.add_argument("--gamma", type=float, default=0.001,
                        help="Softmax temperature for pseudo-pose soft assignment")
    parser.add_argument("--tau", type=float, default=None,
                        help="Distance decay for pose weights (None = auto from median dist)")
    parser.add_argument("--epsilon", type=float, default=0.005,
                        help="Fixed epsilon for combine mode E")
    parser.add_argument("--rechunk_remaining_only", action="store_true",
                        help="Freeze chunk0, re-chunk only remaining frames (reuse chunk0 inference)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]
    device = "cuda"
    thresholds = [3, 5, 15, 30]

    scenes = args.scenes if args.scenes else SEVEN_SCENES
    for s in scenes:
        assert s in SEVEN_SCENES, f"Unknown scene: {s}"

    model = load_model(device, args.model_path, args.chunk_size)
    model.aggregator.dino_batch_size = args.dino_batch_size
    # 'random_partitioning' maps to the aggregator's random-balanced split (no local search)
    model.aggregator.sampling_method = (
        "random_balanced" if args.sampling_method == "random_partitioning" else args.sampling_method)
    model.aggregator.sampling_local_search_iters = args.local_search_iters

    # Origin mode: disable chunking, run single-batch forward
    if args.sampling_method == "origin":
        model.aggregator.sampling_max_frames = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# VGGT Chunked Inference (sampling={args.sampling_method})")
    print(f"# n_frames={args.n_frames}, chunk_size={args.chunk_size}")
    if args.sampling_method == "da_partitioning":
        print(f"# pose-weighted re-chunking: "
              f"gamma={args.gamma}, tau={args.tau}, epsilon={args.epsilon}")
        if args.rechunk_remaining_only:
            print(f"# rechunk_remaining_only=True")
    print(f"{'#'*60}\n")

    results, summary = run_evaluation(model, args, dtype, device, scenes, thresholds)
    mode_names = ["baseline"]
    all_mode_results = {"baseline": results}
    all_mode_summaries = {"baseline": summary}

    # Save results
    save_data = {
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
        'modes': {},
    }
    for mode_name in mode_names:
        save_data['modes'][mode_name] = {
            'scenes': all_mode_results[mode_name],
            'global_summary': all_mode_summaries[mode_name],
        }

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
        json_path = output_dir / f"chunked_{args.scenes[0]}_{name_core}.json"
    else:
        json_path = output_dir / f"chunked_{name_core}.json"
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
