"""Multi-anchor alignment utilities for chunked inference evaluation scripts.

Shared helpers used by:
  - eval_chunked_pose_7scenes.py
  - eval_chunked_3d_scannet.py
  - eval_chunked_depth_bonn.py
  - (any other chunked eval script that aligns predictions across chunks)

Provides:
  * `select_anchors`     — pick n_anchors frame indices via uniform / fps_sim / random / first
  * `_compute_chunk_alignment` / `_apply_chunk_alignment` — multi-anchor SE(3)/Sim(3) pose alignment
  * `multi_anchor_depth_scale` — robust per-chunk depth scale from multiple anchors
"""
from __future__ import annotations

import numpy as np
import torch


# =============================================================================
# Anchor selection (input image list → list of frame indices)
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
            ablation control.

    Returned list preserves the convention that anchors[0] is the primary anchor
    (inserted at position 0 of every chunk by `_insert_anchors`).

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

        selected = [0]
        min_dist = dist[0].astype(np.float64).copy()
        min_dist[0] = -np.inf
        while len(selected) < n_anchors:
            j = int(np.argmax(min_dist))
            if min_dist[j] <= -np.inf:
                break
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
        return list(range(min(n_anchors, N)))

    raise ValueError(
        f"Unknown anchor_select mode: {mode!r} (expected 'uniform', 'fps_sim', 'random', or 'first')"
    )


# =============================================================================
# Multi-anchor SE(3) / Sim(3) pose alignment
# =============================================================================

def _rotation_procrustes(src_rot_cols, dst_rot_cols):
    """Best-fit rotation R such that R @ src ≈ dst (no centering, scale-invariant)."""
    Mmat = dst_rot_cols.T @ src_rot_cols
    U, _, Vt = torch.linalg.svd(Mmat)
    sign = torch.eye(3, device=src_rot_cols.device, dtype=src_rot_cols.dtype)
    if torch.det(U @ Vt) < 0:
        sign[2, 2] = -1.0
    return U @ sign @ Vt


def _multi_anchor_align(src_anchor_se3_list, dst_anchor_se3_list, mode):
    """Decoupled multi-anchor alignment of SE(3) correspondences.

    Models the chunk-k → reference transform as:
        c_dst = s * R * c_src + t          (positions: scale-bearing)
        R_dst = R * R_src                  (rotations: scale-invariant)

    R is fit from rotation-matrix columns via orthogonal Procrustes (using
    rotation info from every anchor). With R fixed, s (sim3 only) and t are
    fit by LS on the anchor camera centers. For mode="se3" the scale is
    forced to 1.

    Args:
        src_anchor_se3_list: list of (4, 4) tensors (chunk k's anchor poses).
        dst_anchor_se3_list: list of (4, 4) tensors (reference anchor poses).
        mode: "se3" or "sim3".
    Returns:
        R: (3, 3), t: (3,), s: 0-d tensor (==1 for se3 / degenerate).
    """
    src = torch.stack(src_anchor_se3_list).float()
    dst = torch.stack(dst_anchor_se3_list).float()

    src_cols = src[:, :3, :3].permute(0, 2, 1).reshape(-1, 3)
    dst_cols = dst[:, :3, :3].permute(0, 2, 1).reshape(-1, 3)
    R = _rotation_procrustes(src_cols, dst_cols)

    src_pos = src[:, :3, 3]
    dst_pos = dst[:, :3, 3]
    mu_src = src_pos.mean(dim=0)
    mu_dst = dst_pos.mean(dim=0)
    src_c = src_pos - mu_src
    dst_c = dst_pos - mu_dst

    if mode == "sim3":
        var_src = (src_c ** 2).sum()
        if var_src < 1e-12:
            s = src_pos.new_ones(())
        else:
            rotated_src_c = src_c @ R.T
            s = (dst_c * rotated_src_c).sum() / var_src
    elif mode == "se3":
        s = src_pos.new_ones(())
    else:
        raise ValueError(f"Unknown align_mode: {mode}")

    t = mu_dst - s * (R @ mu_src)
    return R, t, s


def compute_chunk_alignment(src_anchor_se3_list, dst_anchor_se3_list, mode):
    """Compute alignment from chunk k anchors to reference anchors.

    For len==1 returns the closed-form single-anchor SE(3) transform
    (exact, no scale). For len>=2 runs multi-anchor LS via
    `_multi_anchor_align` ("se3" or "sim3").

    Returns (T_rigid: (4, 4) tensor, scale: 0-d tensor).
    Apply via `apply_chunk_alignment`.
    """
    assert len(src_anchor_se3_list) == len(dst_anchor_se3_list) and len(src_anchor_se3_list) >= 1
    device = src_anchor_se3_list[0].device
    dtype = src_anchor_se3_list[0].dtype

    if len(src_anchor_se3_list) == 1:
        T_align = dst_anchor_se3_list[0] @ torch.linalg.inv(src_anchor_se3_list[0])
        scale = torch.ones((), device=device, dtype=dtype)
        return T_align, scale

    with torch.amp.autocast('cuda', enabled=False):
        R, t, s = _multi_anchor_align(src_anchor_se3_list, dst_anchor_se3_list, mode)

    T_rigid = torch.eye(4, device=device, dtype=dtype)
    T_rigid[:3, :3] = R.to(dtype)
    T_rigid[:3, 3] = t.to(dtype)
    return T_rigid, s.to(dtype)


def apply_chunk_alignment(T_rigid, scale, se3_batch):
    """Apply (T_rigid, scale) alignment to a batch of SE(3) matrices.

    Implements: out = T_rigid @ E_with_scaled_t, where E_with_scaled_t has
    its translation column multiplied by `scale`. Rotation stays rigid;
    scale is absorbed into the translation only. For scale=1 matches the
    single-anchor closed-form behaviour exactly.

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
# Multi-anchor depth scale alignment
# =============================================================================

def multi_anchor_depth_scale(ref_anchor_depths, src_anchor_depths, eps=1e-6):
    """Robust per-chunk depth scale from multiple anchor frames.

    For each anchor pair (ref_depth_i, src_depth_i), compute pixelwise ratios
    ref/src on valid pixels (both > eps). Concatenate all anchors' ratios and
    take the median.

    Returns a Python float scale (defaults to 1.0 if no valid pixels found).

    Args:
        ref_anchor_depths: list of M tensors of shape (H, W, 1) — anchors in reference chunk.
        src_anchor_depths: list of M tensors of shape (H, W, 1) — same anchors in chunk k.
        eps: minimum depth threshold for a pixel to be considered valid.
    """
    assert len(ref_anchor_depths) == len(src_anchor_depths)
    ratios_list = []
    for ref_d, src_d in zip(ref_anchor_depths, src_anchor_depths):
        valid = (ref_d.squeeze(-1) > eps) & (src_d.squeeze(-1) > eps)
        if valid.sum() == 0:
            continue
        ratios_list.append((ref_d[valid] / src_d[valid]).flatten())
    if not ratios_list:
        return 1.0
    all_ratios = torch.cat(ratios_list)
    return float(torch.median(all_ratios))
