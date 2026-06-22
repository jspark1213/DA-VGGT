# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def fl_greedy_sample(cls_tokens: torch.Tensor, B: int, S: int, max_frames: int) -> torch.Tensor:
    """Facility-location greedy frame sampling using DINOv2 CLS tokens.

    Selects the most representative subset of frames by greedily maximizing
    submodular coverage: at each step, pick the unselected frame whose
    addition maximally increases total coverage (sum of per-frame max-similarity
    to any selected frame).

    Args:
        cls_tokens: [B*S, C] L2-normalizable CLS features from DINOv2.
        B: batch size.
        S: original sequence length.
        max_frames: target number of frames to keep (S' <= max_frames).

    Returns:
        selected: [B, S'] int64 tensor of selected frame indices (sorted).
    """
    S_out = min(S, max_frames)
    device = cls_tokens.device

    # Reshape to [B, S, C] and compute cosine similarity per batch
    feats = cls_tokens.view(B, S, -1).float()
    feats = F.normalize(feats, p=2, dim=-1)
    sim = torch.bmm(feats, feats.transpose(1, 2))  # [B, S, S]
    sim = sim.clamp(min=0.0)  # only non-negative similarities

    # Greedy facility-location selection (per batch element)
    coverage = torch.zeros(B, S, device=device)   # max similarity to any selected
    selected_mask = torch.zeros(B, S, device=device, dtype=torch.bool)
    selected_list = []

    for _ in range(S_out):
        # Marginal gain: sum_j max(0, sim[i,j] - coverage[j]) for each candidate i
        # = sum_j max(sim[i,j], coverage[j]) - sum_j coverage[j]
        # Since coverage is fixed for this step, we maximize: sum_j max(sim[i,j], coverage[j])
        improvement = torch.max(sim, coverage.unsqueeze(1))  # [B, S, S]
        gains = improvement.sum(dim=-1)  # [B, S]
        gains[selected_mask] = -1.0  # already selected → invalid

        best = gains.argmax(dim=-1)  # [B]
        selected_list.append(best)
        selected_mask.scatter_(1, best.unsqueeze(1), True)

        # Update coverage: coverage[j] = max(coverage[j], sim[best, j])
        best_sim = sim[torch.arange(B, device=device), best]  # [B, S]
        coverage = torch.max(coverage, best_sim)

    selected = torch.stack(selected_list, dim=1)  # [B, S_out]
    selected = selected.sort(dim=1).values  # sort for temporal ordering
    return selected


def select_diverse_anchors(sim, chunk_cov, n_anchors):
    """Greedy diverse anchor selection: coverage quality x visual diversity.

    Selects n_anchors frames that are both well-represented across all chunks
    (high min-coverage) and visually diverse from each other.

    Args:
        sim: (N, N) numpy cosine similarity matrix.
        chunk_cov: (K, N) numpy per-frame coverage for each chunk.
        n_anchors: number of anchors to select.

    Returns:
        anchors: list of int, anchor frame indices. anchors[0] has highest coverage.
    """
    import numpy as np

    N = sim.shape[0]
    n_anchors = min(n_anchors, N)
    anchor_scores = chunk_cov.min(axis=0)  # (N,)
    anchors = []
    selected_mask = np.zeros(N, dtype=bool)

    for i in range(n_anchors):
        if i == 0:
            best = int(np.argmax(anchor_scores))
        else:
            # Penalize candidates similar to already-selected anchors
            sim_to_selected = sim[:, anchors].max(axis=1)  # (N,)
            diversity = 1.0 - sim_to_selected
            penalized = anchor_scores * diversity
            penalized[selected_mask] = -1.0
            best = int(np.argmax(penalized))
        anchors.append(best)
        selected_mask[best] = True

    return anchors


def _utility_2opt_local_search(chunks, U, local_search_iters):
    """2-opt local search: swap frames between chunks to maximise within-chunk pair utility.

    For each pair of chunks (k1, k2), exhaustively find the best swap F[i] <-> G[j]
    that maximises the net delta in total pair utility across both chunks.
    Repeats for `local_search_iters` outer iterations or until no improvement.

    Args:
        chunks: list of K lists of frame indices (modified in-place).
        U: [N, N] numpy float64 utility matrix (symmetric, zero diagonal).
        local_search_iters: number of outer iterations (0 = no-op).

    Returns:
        chunks: same list, modified in-place.
    """
    import numpy as np

    K = len(chunks)
    N = U.shape[0]
    if local_search_iters <= 0 or K <= 1:
        return chunks

    U64 = U.astype(np.float64) if U.dtype != np.float64 else U

    chunk_util = np.zeros((K, N), dtype=np.float64)
    for k in range(K):
        if len(chunks[k]) > 0:
            chunk_util[k] = U64[:, chunks[k]].sum(axis=1)

    for _iteration in range(local_search_iters):
        improved = False
        for k1 in range(K):
            for k2 in range(k1 + 1, K):
                while True:
                    F = np.array(chunks[k1], dtype=np.intp)
                    G = np.array(chunks[k2], dtype=np.intp)
                    if len(F) == 0 or len(G) == 0:
                        break
                    row = chunk_util[k2][F] - chunk_util[k1][F]
                    col = chunk_util[k1][G] - chunk_util[k2][G]
                    delta = row[:, None] + col[None, :] - 2.0 * U64[np.ix_(F, G)]
                    best_flat = int(np.argmax(delta))
                    bi, bj = divmod(best_flat, len(G))
                    if delta[bi, bj] <= 1e-8:
                        break
                    f_idx, g_idx = int(F[bi]), int(G[bj])
                    chunks[k1][bi] = g_idx
                    chunks[k2][bj] = f_idx
                    chunk_util[k1] += U64[:, g_idx] - U64[:, f_idx]
                    chunk_util[k2] += U64[:, f_idx] - U64[:, g_idx]
                    improved = True
        if not improved:
            break

    return chunks


def _insert_anchors(chunks, N, n_anchors, anchors_override=None):
    """Select anchors (or use the override) and insert into each chunk (primary at position 0)."""
    if anchors_override is not None:
        anchors = list(anchors_override)
        n_anchors = len(anchors)
    elif n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (N - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        anchors = list(dict.fromkeys(anchors))

    n_eff = max(1, len(anchors))
    anchor_set = set(anchors)
    for k in range(len(chunks)):
        non_anchor = [f for f in chunks[k] if f not in anchor_set]
        result = [anchors[0]]
        original_len = len(non_anchor)
        for a_idx, anc in enumerate(anchors[1:], 1):
            insert_pos = original_len * a_idx // n_eff + (a_idx - 1)
            non_anchor.insert(insert_pos, anc)
        result.extend(non_anchor)
        chunks[k] = result

    return chunks, anchors


def _compute_score_matrix(sim_matrix, score_type, alpha=0.7):
    """Compute a pairwise score matrix for 2-opt local search.

    Args:
        sim_matrix: [N, N] numpy similarity matrix (values in [0, 1]).
        score_type: one of 'utility', 'rawsim', 'revsim'.
            - 'utility': s^alpha * (1-s)^(1-alpha) — pair utility (peaks at mid-sim).
            - 'rawsim': s — maximise within-chunk similarity.
            - 'revsim': (1-s) — maximise within-chunk diversity.
        alpha: exponent for utility scoring (ignored for rawsim/revsim).

    Returns:
        U: [N, N] float64 numpy array, symmetric, zero diagonal.
    """
    import numpy as np
    sim = np.clip(sim_matrix, 0, None).astype(np.float64)

    if score_type == "rawsim":
        U = sim.copy()
    elif score_type == "revsim":
        U = 1.0 - sim
    elif score_type == "utility":
        sim_safe = np.clip(sim, 1e-8, 1.0 - 1e-8)
        U = np.power(sim_safe, alpha) * np.power(1.0 - sim_safe, 1.0 - alpha)
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    np.fill_diagonal(U, 0.0)
    return U


# ---------------------------------------------------------------------------
# Pose-weighted local search helpers
# ---------------------------------------------------------------------------

def compute_pseudo_poses(chunk0_positions, chunk0_indices, N, sim_matrix, gamma=1.0):
    """Compute pseudo-poses for all N frames via soft assignment from chunk 0.

    For frame t not in chunk 0:
        π_{t,a} = softmax_a( cos(f_t, f_a) / γ )
        p̂_t = Σ_{a ∈ chunk0} π_{t,a} · p_a

    For frame t in chunk 0: p̂_t = p_t (actual inferred position).

    Args:
        chunk0_positions: (S_k, 3) numpy array, 3D positions of chunk 0 frames.
        chunk0_indices: list of int, frame indices belonging to chunk 0.
        N: total number of frames.
        sim_matrix: (N, N) numpy cosine similarity matrix.
        gamma: softmax temperature (higher = more uniform, lower = sharper).

    Returns:
        pseudo_positions: (N, 3) numpy float64 array.
    """
    import numpy as np

    pseudo_positions = np.zeros((N, 3), dtype=np.float64)
    chunk0_arr = np.array(chunk0_indices, dtype=np.intp)
    chunk0_set = set(chunk0_indices)
    chunk0_pos = np.asarray(chunk0_positions, dtype=np.float64)  # (S_k, 3)

    # Set known positions for chunk 0 frames
    for local_idx, global_idx in enumerate(chunk0_indices):
        pseudo_positions[global_idx] = chunk0_pos[local_idx]

    # Vectorised soft assignment for all non-chunk0 frames
    non_chunk0 = np.array([i for i in range(N) if i not in chunk0_set], dtype=np.intp)
    if len(non_chunk0) > 0:
        sims = sim_matrix[non_chunk0][:, chunk0_arr]  # (M, S_k)
        sims_scaled = sims / gamma
        sims_scaled -= sims_scaled.max(axis=1, keepdims=True)  # numerical stability
        weights = np.exp(sims_scaled)
        weights /= weights.sum(axis=1, keepdims=True)  # (M, S_k)
        pseudo_positions[non_chunk0] = weights @ chunk0_pos  # (M, 3)

    return pseudo_positions


def compute_pose_weight_matrix(positions, tau=None):
    """Compute pairwise pose-proximity weight matrix.

    w^{pose}_{ij} = exp(-||p_i - p_j|| / τ)

    Args:
        positions: (N, 3) numpy array of 3D camera positions.
        tau: distance decay temperature.  If None, auto-set to median of
             all pairwise distances (scale-invariant default).

    Returns:
        W_pose: (N, N) numpy float64, symmetric, zero diagonal.
        tau_used: float.
    """
    import numpy as np

    pos = np.asarray(positions, dtype=np.float64)
    diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
    dists = np.linalg.norm(diff, axis=2)       # (N, N)

    if tau is None:
        nonzero = dists[np.triu_indices_from(dists, k=1)]
        tau = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
        tau = max(tau, 1e-8)

    W_pose = np.exp(-dists / tau).astype(np.float64)
    np.fill_diagonal(W_pose, 0.0)
    return W_pose, tau


# ----------------------------------------------------------------------------
# Rotation-aware extension (rebuttal ablation).
#
# Paper Eq.(5) propagates only translations. Reviewer asked whether including
# rotation cues changes results. The helpers below let `run_inference_poseweight`
# also propagate rotations from the seed chunk and build an SE(3) dispersion
# matrix d_ij = ||Δt|| + α·θ.
# ----------------------------------------------------------------------------

def _weighted_quaternion_mean(quats, weights):
    """Weighted quaternion mean via Markley's eigenvector method.

    For unit quaternions q_i with weights w_i, the mean is the eigenvector of
    the largest eigenvalue of M = Σ w_i q_i q_i^T (Markley 2007).
    Robust to antipodal duplicates and weighted assignments.

    Args:
        quats:   (S, 4) numpy array of unit quaternions (any convention; consistent).
        weights: (S,)   numpy array of non-negative weights, sum > 0.

    Returns:
        (4,) unit quaternion (same convention as input).
    """
    import numpy as np
    q = np.asarray(quats, dtype=np.float64)            # (S, 4)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    M = (w * q).T @ q                                  # (4, 4)
    eigvals, eigvecs = np.linalg.eigh(M)
    qm = eigvecs[:, -1]
    qm = qm / max(np.linalg.norm(qm), 1e-12)
    return qm


def compute_pseudo_rotations(chunk0_rotations, chunk0_indices, N, sim_matrix,
                              gamma=1.0):
    """Pseudo-rotations for all N frames via soft assignment from chunk 0.

    Mirrors `compute_pseudo_poses` but on SO(3): for frame t ∉ chunk 0,
        π_{t,a} = softmax_a(cos(f_t, f_a) / γ)
        R̂_t    = WeightedQuatMean({R_a}_{a∈chunk0}, π_{t,·})

    For t ∈ chunk 0: R̂_t = R_t (actual inferred rotation from VGGT).

    Args:
        chunk0_rotations: (S_k, 3, 3) numpy array of rotation matrices (w2c or c2w
            — caller's convention; output keeps the same convention).
        chunk0_indices:   list of int.
        N:                total frames.
        sim_matrix:       (N, N) numpy cosine similarity.
        gamma:            softmax temperature (matches translation propagation).

    Returns:
        pseudo_rotations: (N, 3, 3) numpy float64.
    """
    import numpy as np
    from vggt.utils.rotation import mat_to_quat, quat_to_mat
    import torch as _torch

    S_k = len(chunk0_indices)
    pseudo_rot = np.zeros((N, 3, 3), dtype=np.float64)

    chunk0_R = np.asarray(chunk0_rotations, dtype=np.float64)
    chunk0_R_t = _torch.from_numpy(chunk0_R)
    chunk0_quats = mat_to_quat(chunk0_R_t).cpu().numpy()  # (S_k, 4) ijkr

    chunk0_set = set(chunk0_indices)
    for local_idx, global_idx in enumerate(chunk0_indices):
        pseudo_rot[global_idx] = chunk0_R[local_idx]

    chunk0_arr = np.array(chunk0_indices, dtype=np.intp)
    non_chunk0 = np.array([i for i in range(N) if i not in chunk0_set], dtype=np.intp)
    if len(non_chunk0) == 0:
        return pseudo_rot

    sims = sim_matrix[non_chunk0][:, chunk0_arr] / gamma
    sims -= sims.max(axis=1, keepdims=True)
    weights = np.exp(sims)
    weights /= weights.sum(axis=1, keepdims=True)        # (M, S_k)

    # Per-frame Markley mean. M is small (≤ N ~ 500) so a Python loop is fine.
    pseudo_quats = np.zeros((len(non_chunk0), 4), dtype=np.float64)
    for j, w_row in enumerate(weights):
        pseudo_quats[j] = _weighted_quaternion_mean(chunk0_quats, w_row)
    pseudo_quats_t = _torch.from_numpy(pseudo_quats)
    pseudo_R = quat_to_mat(pseudo_quats_t).cpu().numpy()  # (M, 3, 3)
    pseudo_rot[non_chunk0] = pseudo_R

    return pseudo_rot


def _rotation_angles(R):
    """Pairwise geodesic angles on SO(3): θ_ij = arccos((tr(R_i^T R_j) - 1)/2).

    Args:
        R: (N, 3, 3) numpy rotation matrices.

    Returns:
        (N, N) numpy float64, in [0, π], zero diagonal.
    """
    import numpy as np
    Rt = np.transpose(R, (0, 2, 1))                       # (N, 3, 3)
    # trace(R_i^T @ R_j) = sum_{a,b} R_i[b,a] R_j[b,a] = (R_i * R_j).sum() over (a,b)
    # Pairwise: einsum('iba,jba->ij', R, R)  (since R_i^T[a,b] = R_i[b,a])
    tr_ij = np.einsum('iba,jba->ij', R, R)
    cos = np.clip((tr_ij - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos)


def compute_pseudo_focals(chunk0_focals, chunk0_indices, N, sim_matrix, gamma=1.0):
    """Pseudo-focal-lengths via the same softmax-on-similarity rule as Eq.(5).

    Args:
        chunk0_focals: (S_k,) numpy array — predicted focal length per chunk0 frame
            (typically mean of fx, fy from VGGT camera_head).
        chunk0_indices: list of int.
        N:              total frames.
        sim_matrix:     (N, N) cosine similarity.
        gamma:          softmax temperature (matches Eq.(5)).

    Returns:
        pseudo_focals: (N,) numpy float64.
    """
    import numpy as np

    pseudo_f = np.zeros(N, dtype=np.float64)
    chunk0_f = np.asarray(chunk0_focals, dtype=np.float64)
    chunk0_set = set(chunk0_indices)
    for li, gi in enumerate(chunk0_indices):
        pseudo_f[gi] = chunk0_f[li]

    chunk0_arr = np.array(chunk0_indices, dtype=np.intp)
    non_chunk0 = np.array([i for i in range(N) if i not in chunk0_set], dtype=np.intp)
    if len(non_chunk0) == 0:
        return pseudo_f

    sims = sim_matrix[non_chunk0][:, chunk0_arr] / gamma
    sims -= sims.max(axis=1, keepdims=True)
    weights = np.exp(sims)
    weights /= weights.sum(axis=1, keepdims=True)         # (M, S_k)
    pseudo_f[non_chunk0] = weights @ chunk0_f
    return pseudo_f


def compute_se3_pose_weight_matrix(positions, rotations=None, focals=None,
                                    tau=None, alpha_rot=1.0, beta_focal=0.0,
                                    mode="translation",
                                    fair_normalize=True):
    """SE(3) (+ optional intrinsics) pose-proximity weight matrix.

    Base distance by `mode`:
        'translation' : d_ij = ||Δt||              (paper baseline)
        'additive'    : d_ij = ||Δt|| + α·θ_ij
        'rotation'    : d_ij = α·θ_ij               (translation ignored)

    Optional intrinsics term: if focals is not None and beta_focal > 0,
        d_ij ← d_ij + β · |Δf_ij| / f̄,
    where f̄ is the mean of all pseudo-focals (dimensionless normalization).

    Fair scale normalization (fair_normalize=True, default for ablations):
        Each component is divided by its own median pairwise value before being
        summed, so α=β=1 corresponds to equal effective weight against the
        translation term. Without this, ||Δt|| (scene units) dominates and
        rotation/intrinsics terms contribute negligibly.

    W_pose_ij = exp(-d_ij / τ), zero diagonal. τ defaults to median pairwise d.

    Args:
        positions:  (N, 3) numpy — pseudo-translations.
        rotations:  (N, 3, 3) numpy — pseudo-rotations (required if mode≠'translation').
        focals:     (N,) numpy — pseudo-focals (optional intrinsics cue).
        tau:        distance decay; None ⇒ median(triu pairwise d).
        alpha_rot:  scalar weighting θ (radians).
        beta_focal: scalar weighting |Δf|/f̄ (dimensionless).
        mode:       'translation' | 'additive' | 'rotation'.
        fair_normalize: if True, normalize each component by its own median
            before summation (default; needed for fair ablation).

    Returns:
        W_pose: (N, N) numpy float64.
        tau_used: float.
    """
    import numpy as np

    def _triu_median(x):
        iu = np.triu_indices_from(x, k=1)
        v = x[iu]
        v = v[v > 0]
        return float(np.median(v)) if len(v) else 1.0

    def _triu_std(x):
        v = x[np.triu_indices_from(x, k=1)]
        v = v[v > 0]
        return float(np.std(v)) if len(v) else 1.0

    pos = np.asarray(positions, dtype=np.float64)
    if mode == "translation":
        diff = pos[:, None, :] - pos[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
    else:
        if rotations is None:
            raise ValueError(f"mode={mode!r} requires rotations to be provided")
        Rm = np.asarray(rotations, dtype=np.float64)
        theta = _rotation_angles(Rm)                      # (N, N) radians
        if mode == "additive":
            diff = pos[:, None, :] - pos[None, :, :]
            d_t = np.linalg.norm(diff, axis=2)
            if fair_normalize:
                med_t = max(_triu_median(d_t), 1e-8)
                med_th = max(_triu_median(theta), 1e-8)
                t_norm = d_t / med_t
                r_norm = theta / med_th
                # alpha_rot == None  -> adaptive std-matching: scale rotation so
                # its dispersion (std on the normalized term) equals translation's.
                if alpha_rot is None:
                    alpha_eff = _triu_std(t_norm) / max(_triu_std(r_norm), 1e-8)
                else:
                    alpha_eff = float(alpha_rot)
                print(f"    [se3_pose_weight additive] med_t={med_t:.6f} med_theta={med_th:.6f} "
                      f"std(t_norm)={_triu_std(t_norm):.6f} std(r_norm)={_triu_std(r_norm):.6f} "
                      f"alpha_eff={alpha_eff:.6f}")
                dists = t_norm + alpha_eff * r_norm
            else:
                dists = d_t + (alpha_rot if alpha_rot is not None else 1.0) * theta
        elif mode == "rotation":
            if fair_normalize:
                med_th = max(_triu_median(theta), 1e-8)
                # alpha_rot has no scale ambiguity when translation is absent;
                # use 1.0 (or user override) on the normalized term.
                alpha_eff = 1.0 if alpha_rot is None else float(alpha_rot)
                dists = alpha_eff * theta / med_th
            else:
                dists = (alpha_rot if alpha_rot is not None else 1.0) * theta
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Optional intrinsics term: scale-normalized focal-length difference.
    if focals is not None and (beta_focal is None or beta_focal > 0):
        f = np.asarray(focals, dtype=np.float64)
        f_mean = max(float(f.mean()), 1e-8)
        df = np.abs(f[:, None] - f[None, :]) / f_mean
        if fair_normalize:
            med_f = max(_triu_median(df), 1e-8)
            df_norm = df / med_f
            # beta_focal == None -> adaptive std-matching to translation term.
            if beta_focal is None:
                # reference std = std of pure-translation part of `dists` if
                # available, else of df_norm itself (1.0).
                if mode == "translation":
                    ref_std = _triu_std(dists)
                else:
                    # Use the translation component pre-mix; falls back to dists std.
                    ref_std = _triu_std(dists)
                beta_eff = ref_std / max(_triu_std(df_norm), 1e-8)
            else:
                beta_eff = float(beta_focal)
            dists = dists + beta_eff * df_norm
        else:
            beta_eff = float(beta_focal) if beta_focal is not None else 1.0
            dists = dists + beta_eff * df

    if tau is None:
        nonzero = dists[np.triu_indices_from(dists, k=1)]
        tau = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
        tau = max(tau, 1e-8)

    W_pose = np.exp(-dists / tau).astype(np.float64)
    np.fill_diagonal(W_pose, 0.0)
    return W_pose, tau


def _combine_score_matrices(U_appearance, W_pose, combine_mode="A", alpha_combine=0.5,
                            epsilon=None):
    """Combine appearance score matrix with pose-weight matrix.

    Args:
        U_appearance: (N, N) numpy float64, appearance score (e.g. revsim).
        W_pose: (N, N) numpy float64, pose proximity weights.
        combine_mode: "A" (linear), "B" (multiplicative), "C" (pose-only),
                      "D" (U + ε·W), "E" (U - ε·W).
        alpha_combine: mixing weight for mode A (weight on appearance side).
        epsilon: fixed epsilon for mode D/E. If None, use adaptive (0.01 × std(U)/std(W)).

    Returns:
        U_combined: (N, N) numpy float64.
    """
    import numpy as np

    if combine_mode == "A":
        U_combined = alpha_combine * U_appearance + (1.0 - alpha_combine) * W_pose
    elif combine_mode == "B":
        U_combined = U_appearance * W_pose
    elif combine_mode == "C":
        # Pose-only DISPERSION (paper Eq.7 with δ removed).
        # W_pose = exp(-||Δt||/τ) is a PROXIMITY matrix (close→large).
        # paper's ρ_ij = 1 - exp(-||Δt||/τ) is dispersion (far→large), which
        # is what the local search must MAXIMIZE for diversity.
        U_combined = 1.0 - W_pose
    elif combine_mode in ("D", "E"):
        std_app = max(np.std(U_appearance[U_appearance > 0]), 1e-12)
        std_pose = max(np.std(W_pose[W_pose > 0]), 1e-12)
        adaptive_eps = 0.01 * std_app / std_pose
        if epsilon is not None:
            eps = epsilon
        else:
            eps = adaptive_eps
        sign = "+" if combine_mode == "D" else "-"
        print(f"    [combine] std(U)={std_app:.6f} std(W)={std_pose:.6f} "
              f"std(U)/std(W)={std_app/std_pose:.6f} adaptive_eps={adaptive_eps:.8f} "
              f"used_eps={eps} mode={sign}")
        if combine_mode == "D":
            U_combined = U_appearance + eps * W_pose
        else:
            U_combined = U_appearance - eps * W_pose
    else:
        raise ValueError(f"Unknown combine_mode: {combine_mode}")

    np.fill_diagonal(U_combined, 0.0)
    return U_combined


def rechunk_with_pose_weights(chunks, anchors, sim_matrix, W_pose,
                               score_type="revsim", alpha=0.0,
                               combine_mode="A", alpha_combine=0.5,
                               n_anchors=1, local_search_iters=5,
                               epsilon=None, anchors_override=None):
    """Re-chunk using pose-weighted combined score for 2-opt local search.

    Pipeline:
      1. Strip anchors from chunks
      2. Compute appearance score matrix U from sim_matrix
      3. Combine U with W_pose via combine_mode
      4. Run 2-opt local search on combined score
      5. Re-insert anchors

    Returns:
        (chunks, anchors, ls_time)
    """
    import numpy as np
    import time as _time

    if local_search_iters <= 0 or len(chunks) <= 1:
        return chunks, anchors, 0.0

    sim = sim_matrix if isinstance(sim_matrix, np.ndarray) else sim_matrix
    N = sim.shape[0]

    anchor_set = set(anchors)
    stripped = [[f for f in ch if f not in anchor_set] for ch in chunks]

    U_appearance = _compute_score_matrix(sim, score_type, alpha)
    U_combined = _combine_score_matrices(U_appearance, W_pose, combine_mode, alpha_combine,
                                          epsilon=epsilon)

    t_ls_start = _time.time()
    _utility_2opt_local_search(stripped, U_combined, local_search_iters)
    ls_time = _time.time() - t_ls_start

    chunks, anchors = _insert_anchors(stripped, N, n_anchors, anchors_override=anchors_override)
    return chunks, anchors, ls_time


def _run_ls_pipeline(chunks, anchors, sim_matrix, score_type, alpha, n_anchors, local_search_iters,
                      anchors_override=None):
    """Shared pipeline: strip anchors → compute score → 2-opt → re-insert anchors.

    Args:
        chunks: list of K lists of frame indices (from Phase 1).
        anchors: list of anchor frame indices.
        sim_matrix: [N, N] numpy or torch similarity matrix.
        score_type: 'utility', 'rawsim', or 'revsim'.
        alpha: utility exponent (only used when score_type='utility').
        n_anchors: number of anchors.
        local_search_iters: 2-opt iterations.

    Returns:
        (chunks, anchors, ls_time) after local search and anchor re-insertion.
        ls_time is the wall-clock time spent on 2-opt local search (seconds).
    """
    import numpy as np
    import time as _time

    if local_search_iters <= 0 or len(chunks) <= 1:
        return chunks, anchors, 0.0

    sim = sim_matrix if isinstance(sim_matrix, np.ndarray) else sim_matrix.cpu().numpy()
    N = sim.shape[0]

    anchor_set = set(anchors)
    stripped = [[f for f in ch if f not in anchor_set] for ch in chunks]

    U = _compute_score_matrix(sim, score_type, alpha)
    t_ls_start = _time.time()
    _utility_2opt_local_search(stripped, U, local_search_iters)
    ls_time = _time.time() - t_ls_start

    chunks, anchors = _insert_anchors(stripped, N, n_anchors, anchors_override=anchors_override)
    return chunks, anchors, ls_time


def step_ls_split(S, chunk_size, sim_matrix, alpha=0.7, n_anchors=1, local_search_iters=5):
    """Step sampling + 2-opt utility local search.

    Phase 1: step_sampling_split for deterministic initial assignment.
    Phase 2: utility-based 2-opt local search (anchors excluded from swaps).
    Phase 3: anchor re-insertion.
    """
    from vggt.models.vggt import step_sampling_split
    chunks, anchors = step_sampling_split(S, chunk_size, n_anchors=n_anchors)
    return _run_ls_pipeline(chunks, anchors, sim_matrix, "utility", alpha, n_anchors, local_search_iters)


def step_ls_rawsim_split(S, chunk_size, sim_matrix, n_anchors=1, local_search_iters=5):
    """Step sampling + 2-opt local search with raw similarity scoring."""
    from vggt.models.vggt import step_sampling_split
    chunks, anchors = step_sampling_split(S, chunk_size, n_anchors=n_anchors)
    return _run_ls_pipeline(chunks, anchors, sim_matrix, "rawsim", 0.0, n_anchors, local_search_iters)


def step_ls_revsim_split(S, chunk_size, sim_matrix, n_anchors=1, local_search_iters=5):
    """Step sampling + 2-opt local search with reverse similarity (1-s) scoring."""
    from vggt.models.vggt import step_sampling_split
    chunks, anchors = step_sampling_split(S, chunk_size, n_anchors=n_anchors)
    return _run_ls_pipeline(chunks, anchors, sim_matrix, "revsim", 0.0, n_anchors, local_search_iters)


def random_balanced_ls_split(sim_matrix, chunk_size, alpha=0.7, n_anchors=1,
                             local_search_iters=5, seed=42):
    """Random balanced split + 2-opt utility local search.

    Phase 1: greedy_random_balanced_split for random initial assignment.
    Phase 2: utility-based 2-opt local search (anchors excluded from swaps).
    Phase 3: anchor re-insertion.
    """
    chunks, anchors = greedy_random_balanced_split(sim_matrix, chunk_size,
                                                   n_anchors=n_anchors, seed=seed)
    return _run_ls_pipeline(chunks, anchors, sim_matrix, "utility", alpha, n_anchors, local_search_iters)


def random_balanced_ls_rawsim_split(sim_matrix, chunk_size, n_anchors=1,
                                     local_search_iters=5, seed=42):
    """Random balanced split + 2-opt local search with raw similarity scoring."""
    chunks, anchors = greedy_random_balanced_split(sim_matrix, chunk_size,
                                                   n_anchors=n_anchors, seed=seed)
    return _run_ls_pipeline(chunks, anchors, sim_matrix, "rawsim", 0.0, n_anchors, local_search_iters)


def random_balanced_ls_revsim_split(sim_matrix, chunk_size, n_anchors=1,
                                     local_search_iters=5, seed=42, anchors_override=None):
    """Random balanced split + 2-opt local search with reverse similarity (1-s) scoring."""
    chunks, anchors = greedy_random_balanced_split(sim_matrix, chunk_size,
                                                   n_anchors=n_anchors, seed=seed,
                                                   anchors_override=anchors_override)
    return _run_ls_pipeline(chunks, anchors, sim_matrix, "revsim", 0.0, n_anchors, local_search_iters,
                             anchors_override=anchors_override)


def greedy_random_balanced_split(sim_matrix, chunk_size: int, n_anchors: int = 1, seed: int = 42,
                                  anchors_override=None):
    """Greedy random balanced partition: assign each frame to the smallest chunk.

    No feature information is used — this measures the value of balanced
    allocation alone, without any coverage-aware frame selection.

    Args:
        sim_matrix: [S, S] pairwise cosine similarity (unused, accepted for API compat).
        chunk_size: target number of frames per chunk.
        n_anchors: number of anchor frames.
        seed: random seed.

    Returns:
        chunks: list of K lists of frame indices (primary anchor at position 0).
        anchors: list of anchor frame indices.
    """
    import numpy as np

    if isinstance(sim_matrix, torch.Tensor):
        N = sim_matrix.shape[0]
    else:
        N = sim_matrix.shape[0]

    K = max(1, N // chunk_size)
    rng = np.random.RandomState(seed)

    chunks = [[] for _ in range(K)]
    chunk_counts = np.zeros(K, dtype=np.int32)

    for frame in rng.permutation(N):
        target_k = int(np.argmin(chunk_counts))
        chunks[target_k].append(int(frame))
        chunk_counts[target_k] += 1

    # Select anchors (override-aware)
    if anchors_override is not None:
        anchors = list(anchors_override)
    elif n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (N - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        anchors = list(dict.fromkeys(anchors))

    n_eff = max(1, len(anchors))
    anchor_set = set(anchors)
    for k in range(K):
        non_anchor = [f for f in chunks[k] if f not in anchor_set]
        result = [anchors[0]]
        original_len = len(non_anchor)
        for a_idx, anc in enumerate(anchors[1:], 1):
            insert_pos = original_len * a_idx // n_eff + (a_idx - 1)
            non_anchor.insert(insert_pos, anc)
        result.extend(non_anchor)
        chunks[k] = result

    return chunks, anchors


def fl_maxmin_split(sim_matrix, chunk_size: int, lambda_div: float = 0.0, n_anchors: int = 1):
    """Facility-location maxmin balanced partition with anchor selection (numpy).

    Matches the reference implementation in chunking_methods.py exactly:
    all computation is done with numpy on CPU for deterministic results.

    Args:
        sim_matrix: [S, S] pairwise cosine similarity (numpy array or torch tensor).
        chunk_size: target number of frames per chunk.
        lambda_div: diversity penalty weight (0 = pure FL maxmin, >0 = diversity).

    Returns:
        chunks: list of K lists of frame indices (primary anchor at position 0 in each).
        anchors: list of anchor frame indices (anchors[0] = primary with highest coverage).
    """
    import numpy as np

    # Convert to numpy if torch tensor
    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().numpy()

    N = sim_matrix.shape[0]
    K = max(1, N // chunk_size)
    sim = np.clip(sim_matrix, 0, None).astype(np.float32)
    # sim = (1 - sim) + 0.5

    chunks = [[] for _ in range(K)]
    coverage = np.zeros((K, N), dtype=np.float32)
    chunk_scores = np.zeros(K, dtype=np.float64)
    chunk_counts = np.zeros(K, dtype=np.int32)
    assigned = np.zeros(N, dtype=bool)

    total_steps = min(K * chunk_size, N)
    for step in range(total_steps):
        # 1. Pick the weakest non-full chunk
        eligible_mask = chunk_counts < chunk_size
        if not eligible_mask.any():
            break
        scores_masked = np.where(eligible_mask, chunk_scores, np.inf)
        target_k = int(np.argmin(scores_masked))

        # 2. Vectorized marginal gain computation
        current_cov = coverage[target_k]
        diff = sim - current_cov[np.newaxis, :]
        np.clip(diff, 0, None, out=diff)
        gains = diff.sum(axis=1)
        gains[assigned] = -1.0

        # 3. Diversity penalty: mean sim to existing chunk members
        if lambda_div > 0 and len(chunks[target_k]) > 0:
            penalty = sim[:, chunks[target_k]].mean(axis=1)
            mask = ~assigned & (gains > 0)
            gains[mask] *= (1.0 - lambda_div * penalty[mask])

        best_frame = int(np.argmax(gains))

        # 4. Update state
        chunks[target_k].append(best_frame)
        np.maximum(coverage[target_k], sim[best_frame], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())
        chunk_counts[target_k] += 1
        assigned[best_frame] = True

    # Assign remaining frames to weakest chunk
    unassigned = np.where(~assigned)[0]
    for r in unassigned:
        target_k = int(np.argmin(chunk_scores))
        chunks[target_k].append(int(r))
        np.maximum(coverage[target_k], sim[r], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())

    # Select anchors: uniform sampling from input sequence
    if n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (N - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        anchors = list(dict.fromkeys(anchors))  # remove duplicates, preserve order

    # Insert anchors into each chunk: primary at position 0, secondary uniform
    anchor_set = set(anchors)
    for k in range(K):
        non_anchor = [f for f in chunks[k] if f not in anchor_set]
        result = [anchors[0]]
        original_len = len(non_anchor)
        for a_idx, anc in enumerate(anchors[1:], 1):
            insert_pos = original_len * a_idx // n_anchors + (a_idx - 1)
            non_anchor.insert(insert_pos, anc)
        result.extend(non_anchor)
        chunks[k] = result

    return chunks, anchors


def fl_maxmin_ls_split(sim_matrix, chunk_size: int, alpha: float = 0.7,
                       lambda_qual: float = 0.0, n_anchors: int = 1,
                       local_search_iters: int = 5):
    """FL maxmin (raw-sim coverage) + quality term + utility-based 2-opt local search.

    Phase 1: greedy assignment using raw cosine similarity for coverage
             (global representativeness). When lambda_qual > 0, a quality term
             (pair utility to existing chunk members) is mixed in:
             gain = (1 - lambda_qual) * repr_gain + lambda_qual * qual_gain.
    Phase 2: 2-opt local search that swaps frames between chunks to maximise
             within-chunk pair utility u(s) = s^α(1-s)^(1-α) (intra-chunk quality).
    """
    import numpy as np

    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().numpy()

    N = sim_matrix.shape[0]
    K = max(1, N // chunk_size)
    sim = np.clip(sim_matrix, 0, None).astype(np.float32)

    # Precompute pair utility matrix (needed for quality term and Phase 2)
    eps = 1e-8
    sim_safe = np.clip(sim, eps, 1.0 - eps).astype(np.float64)
    U = np.power(sim_safe, alpha) * np.power(1.0 - sim_safe, 1.0 - alpha)
    np.fill_diagonal(U, 0.0)
    U = U.astype(np.float32)

    # === Phase 1: FL maxmin greedy (raw sim coverage + optional quality) ===
    chunks = [[] for _ in range(K)]
    coverage = np.zeros((K, N), dtype=np.float32)
    chunk_scores = np.zeros(K, dtype=np.float64)
    qual_accum = np.zeros((K, N), dtype=np.float64)
    chunk_counts = np.zeros(K, dtype=np.int32)
    assigned = np.zeros(N, dtype=bool)

    total_steps = min(K * chunk_size, N)
    for step in range(total_steps):
        eligible_mask = chunk_counts < chunk_size
        if not eligible_mask.any():
            break
        scores_masked = np.where(eligible_mask, chunk_scores, np.inf)
        target_k = int(np.argmin(scores_masked))

        # Raw-sim coverage marginal gain
        current_cov = coverage[target_k]
        diff = sim - current_cov[np.newaxis, :]
        np.clip(diff, 0, None, out=diff)
        repr_gains = diff.sum(axis=1).astype(np.float64)
        repr_gains[assigned] = -1.0

        # Quality gain (pair utility to existing members)
        if lambda_qual > 0 and len(chunks[target_k]) > 0:
            qual_gains = qual_accum[target_k].copy()
            qual_gains[assigned] = 0.0
            gains = (1.0 - lambda_qual) * repr_gains + lambda_qual * qual_gains
            gains[assigned] = -1.0
        else:
            gains = repr_gains

        best_frame = int(np.argmax(gains))

        chunks[target_k].append(best_frame)
        np.maximum(coverage[target_k], sim[best_frame], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())
        qual_accum[target_k] += U[best_frame]
        chunk_counts[target_k] += 1
        assigned[best_frame] = True

    unassigned = np.where(~assigned)[0]
    for r in unassigned:
        target_k = int(np.argmin(chunk_scores))
        chunks[target_k].append(int(r))
        np.maximum(coverage[target_k], sim[r], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())
        qual_accum[target_k] += U[r]

    # === Phase 2: 2-opt local search on pair utility ===
    import time as _time
    _t_ls_start = _time.time()
    if local_search_iters > 0 and K > 1:
        eps = 1e-8
        sim_safe = np.clip(sim, eps, 1.0 - eps).astype(np.float64)
        U = np.power(sim_safe, alpha) * np.power(1.0 - sim_safe, 1.0 - alpha)
        np.fill_diagonal(U, 0.0)

        chunk_util = np.zeros((K, N), dtype=np.float64)
        for k in range(K):
            if len(chunks[k]) > 0:
                chunk_util[k] = U[:, chunks[k]].sum(axis=1)

        for _iteration in range(local_search_iters):
            improved = False
            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    while True:
                        F = np.array(chunks[k1], dtype=np.intp)
                        G = np.array(chunks[k2], dtype=np.intp)
                        if len(F) == 0 or len(G) == 0:
                            break
                        row = chunk_util[k2][F] - chunk_util[k1][F]
                        col = chunk_util[k1][G] - chunk_util[k2][G]
                        delta = row[:, None] + col[None, :] - 2.0 * U[np.ix_(F, G)]
                        best_flat = int(np.argmax(delta))
                        bi, bj = divmod(best_flat, len(G))
                        if delta[bi, bj] <= 1e-8:
                            break
                        f_idx, g_idx = int(F[bi]), int(G[bj])
                        chunks[k1][bi] = g_idx
                        chunks[k2][bj] = f_idx
                        chunk_util[k1] += U[:, g_idx] - U[:, f_idx]
                        chunk_util[k2] += U[:, f_idx] - U[:, g_idx]
                        improved = True
            if not improved:
                break
    _ls_time = _time.time() - _t_ls_start

    # === Phase 3: Anchor selection & insertion ===
    if n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (N - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        anchors = list(dict.fromkeys(anchors))

    anchor_set = set(anchors)
    for k in range(K):
        non_anchor = [f for f in chunks[k] if f not in anchor_set]
        result = [anchors[0]]
        original_len = len(non_anchor)
        for a_idx, anc in enumerate(anchors[1:], 1):
            insert_pos = original_len * a_idx // n_anchors + (a_idx - 1)
            non_anchor.insert(insert_pos, anc)
        result.extend(non_anchor)
        chunks[k] = result

    return chunks, anchors, _ls_time


def fl_maxmin_ls_rawsim_split(sim_matrix, chunk_size: int,
                               lambda_qual: float = 0.0, n_anchors: int = 1,
                               local_search_iters: int = 5):
    """FL maxmin (raw-sim coverage) + 2-opt local search with raw similarity scoring.

    Phase 1: identical to fl_maxmin_ls_split (greedy FL maxmin).
    Phase 2: 2-opt local search maximising within-chunk raw similarity sum.
    """
    import numpy as np

    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().numpy()

    N = sim_matrix.shape[0]
    K = max(1, N // chunk_size)
    sim = np.clip(sim_matrix, 0, None).astype(np.float32)

    # === Phase 1: FL maxmin greedy (same as fl_maxmin_ls_split) ===
    chunks = [[] for _ in range(K)]
    coverage = np.zeros((K, N), dtype=np.float32)
    chunk_scores = np.zeros(K, dtype=np.float64)
    chunk_counts = np.zeros(K, dtype=np.int32)
    assigned = np.zeros(N, dtype=bool)

    total_steps = min(K * chunk_size, N)
    for step in range(total_steps):
        eligible_mask = chunk_counts < chunk_size
        if not eligible_mask.any():
            break
        scores_masked = np.where(eligible_mask, chunk_scores, np.inf)
        target_k = int(np.argmin(scores_masked))

        current_cov = coverage[target_k]
        diff = sim - current_cov[np.newaxis, :]
        np.clip(diff, 0, None, out=diff)
        gains = diff.sum(axis=1).astype(np.float64)
        gains[assigned] = -1.0
        best_frame = int(np.argmax(gains))

        chunks[target_k].append(best_frame)
        np.maximum(coverage[target_k], sim[best_frame], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())
        chunk_counts[target_k] += 1
        assigned[best_frame] = True

    unassigned = np.where(~assigned)[0]
    for r in unassigned:
        target_k = int(np.argmin(chunk_scores))
        chunks[target_k].append(int(r))
        np.maximum(coverage[target_k], sim[r], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())

    # === Phase 2: 2-opt local search with raw similarity ===
    import time as _time
    _t_ls_start = _time.time()
    if local_search_iters > 0 and K > 1:
        U = _compute_score_matrix(sim, "rawsim")
        _utility_2opt_local_search(chunks, U, local_search_iters)
    _ls_time = _time.time() - _t_ls_start

    # === Phase 3: Anchor selection & insertion ===
    if n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (N - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        anchors = list(dict.fromkeys(anchors))

    anchor_set = set(anchors)
    for k in range(K):
        non_anchor = [f for f in chunks[k] if f not in anchor_set]
        result = [anchors[0]]
        original_len = len(non_anchor)
        for a_idx, anc in enumerate(anchors[1:], 1):
            insert_pos = original_len * a_idx // n_anchors + (a_idx - 1)
            non_anchor.insert(insert_pos, anc)
        result.extend(non_anchor)
        chunks[k] = result

    return chunks, anchors, _ls_time


def fl_maxmin_ls_revsim_split(sim_matrix, chunk_size: int,
                               lambda_qual: float = 0.0, n_anchors: int = 1,
                               local_search_iters: int = 5):
    """FL maxmin (raw-sim coverage) + 2-opt local search with reverse similarity (1-s) scoring.

    Phase 1: identical to fl_maxmin_ls_split (greedy FL maxmin).
    Phase 2: 2-opt local search maximising within-chunk diversity (1 - similarity).
    """
    import numpy as np

    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().numpy()

    N = sim_matrix.shape[0]
    K = max(1, N // chunk_size)
    sim = np.clip(sim_matrix, 0, None).astype(np.float32)

    # === Phase 1: FL maxmin greedy (same as fl_maxmin_ls_split) ===
    chunks = [[] for _ in range(K)]
    coverage = np.zeros((K, N), dtype=np.float32)
    chunk_scores = np.zeros(K, dtype=np.float64)
    chunk_counts = np.zeros(K, dtype=np.int32)
    assigned = np.zeros(N, dtype=bool)

    total_steps = min(K * chunk_size, N)
    for step in range(total_steps):
        eligible_mask = chunk_counts < chunk_size
        if not eligible_mask.any():
            break
        scores_masked = np.where(eligible_mask, chunk_scores, np.inf)
        target_k = int(np.argmin(scores_masked))

        current_cov = coverage[target_k]
        diff = sim - current_cov[np.newaxis, :]
        np.clip(diff, 0, None, out=diff)
        gains = diff.sum(axis=1).astype(np.float64)
        gains[assigned] = -1.0
        best_frame = int(np.argmax(gains))

        chunks[target_k].append(best_frame)
        np.maximum(coverage[target_k], sim[best_frame], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())
        chunk_counts[target_k] += 1
        assigned[best_frame] = True

    unassigned = np.where(~assigned)[0]
    for r in unassigned:
        target_k = int(np.argmin(chunk_scores))
        chunks[target_k].append(int(r))
        np.maximum(coverage[target_k], sim[r], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())

    # === Phase 2: 2-opt local search with reverse similarity ===
    import time as _time
    _t_ls_start = _time.time()
    if local_search_iters > 0 and K > 1:
        U = _compute_score_matrix(sim, "revsim")
        _utility_2opt_local_search(chunks, U, local_search_iters)
    _ls_time = _time.time() - _t_ls_start

    # === Phase 3: Anchor selection & insertion ===
    if n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (N - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        anchors = list(dict.fromkeys(anchors))

    anchor_set = set(anchors)
    for k in range(K):
        non_anchor = [f for f in chunks[k] if f not in anchor_set]
        result = [anchors[0]]
        original_len = len(non_anchor)
        for a_idx, anc in enumerate(anchors[1:], 1):
            insert_pos = original_len * a_idx // n_anchors + (a_idx - 1)
            non_anchor.insert(insert_pos, anc)
        result.extend(non_anchor)
        chunks[k] = result

    return chunks, anchors, _ls_time


def fl_maxmin_band_split(sim_matrix, chunk_size: int, sim_lo: float = 0.5,
                         sim_hi: float = 0.9, n_anchors: int = 1,
                         local_search_iters: int = 5):
    """FL maxmin (raw-sim coverage) + hard-band 2-opt local search.

    Phase 1: identical to fl_maxmin_split (raw sim coverage).
    Phase 2: 2-opt local search using hard band utility:
             U[i,j] = 1 if sim_lo <= sim(i,j) <= sim_hi, else 0.
             Maximises the count of "useful pairs" within each chunk.

    This is the simple-threshold alternative to the smooth pair utility
    s^α(1-s)^(1-α). More interpretable: sim_lo = minimum overlap needed,
    sim_hi = maximum redundancy tolerated.
    """
    import numpy as np

    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().numpy()

    N = sim_matrix.shape[0]
    K = max(1, N // chunk_size)
    sim = np.clip(sim_matrix, 0, None).astype(np.float32)

    # === Phase 1: FL maxmin greedy (raw sim) ===
    chunks = [[] for _ in range(K)]
    coverage = np.zeros((K, N), dtype=np.float32)
    chunk_scores = np.zeros(K, dtype=np.float64)
    chunk_counts = np.zeros(K, dtype=np.int32)
    assigned = np.zeros(N, dtype=bool)

    total_steps = min(K * chunk_size, N)
    for step in range(total_steps):
        eligible_mask = chunk_counts < chunk_size
        if not eligible_mask.any():
            break
        scores_masked = np.where(eligible_mask, chunk_scores, np.inf)
        target_k = int(np.argmin(scores_masked))

        current_cov = coverage[target_k]
        diff = sim - current_cov[np.newaxis, :]
        np.clip(diff, 0, None, out=diff)
        gains = diff.sum(axis=1)
        gains[assigned] = -1.0

        best_frame = int(np.argmax(gains))

        chunks[target_k].append(best_frame)
        np.maximum(coverage[target_k], sim[best_frame], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())
        chunk_counts[target_k] += 1
        assigned[best_frame] = True

    unassigned = np.where(~assigned)[0]
    for r in unassigned:
        target_k = int(np.argmin(chunk_scores))
        chunks[target_k].append(int(r))
        np.maximum(coverage[target_k], sim[r], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())

    # === Phase 2: 2-opt local search on hard band ===
    import time as _time
    _t_ls_start = _time.time()
    if local_search_iters > 0 and K > 1:
        U = ((sim >= sim_lo) & (sim <= sim_hi)).astype(np.float64)
        np.fill_diagonal(U, 0.0)

        chunk_util = np.zeros((K, N), dtype=np.float64)
        for k in range(K):
            if len(chunks[k]) > 0:
                chunk_util[k] = U[:, chunks[k]].sum(axis=1)

        for _iteration in range(local_search_iters):
            improved = False
            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    while True:
                        F = np.array(chunks[k1], dtype=np.intp)
                        G = np.array(chunks[k2], dtype=np.intp)
                        if len(F) == 0 or len(G) == 0:
                            break
                        row = chunk_util[k2][F] - chunk_util[k1][F]
                        col = chunk_util[k1][G] - chunk_util[k2][G]
                        delta = row[:, None] + col[None, :] - 2.0 * U[np.ix_(F, G)]
                        best_flat = int(np.argmax(delta))
                        bi, bj = divmod(best_flat, len(G))
                        if delta[bi, bj] <= 1e-8:
                            break
                        f_idx, g_idx = int(F[bi]), int(G[bj])
                        chunks[k1][bi] = g_idx
                        chunks[k2][bj] = f_idx
                        chunk_util[k1] += U[:, g_idx] - U[:, f_idx]
                        chunk_util[k2] += U[:, f_idx] - U[:, g_idx]
                        improved = True
            if not improved:
                break
    _ls_time = _time.time() - _t_ls_start

    # === Phase 3: Anchor selection & insertion ===
    if n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (N - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        anchors = list(dict.fromkeys(anchors))

    anchor_set = set(anchors)
    for k in range(K):
        non_anchor = [f for f in chunks[k] if f not in anchor_set]
        result = [anchors[0]]
        original_len = len(non_anchor)
        for a_idx, anc in enumerate(anchors[1:], 1):
            insert_pos = original_len * a_idx // n_anchors + (a_idx - 1)
            non_anchor.insert(insert_pos, anc)
        result.extend(non_anchor)
        chunks[k] = result

    return chunks, anchors, _ls_time


def fl_covpair_split(sim_matrix, chunk_size: int, lambda_div: float = 0.0,
                     lambda_pair: float = 0.5, alpha: float = 0.5, n_anchors: int = 1):
    """Combined coverage + pair-utility balanced partition with anchor selection.

    Merges FL maxmin coverage gain with MaxPair's pair utility into a single
    objective. Coverage ensures each chunk represents the full sequence well;
    pair utility ensures frames within a chunk have good overlap for pose
    estimation (the "goldilocks zone" of similarity).

    Combined gain for adding frame f to chunk k:
        gain(f, k) = cov_gain(f, k) × (1 - λ_div × div_penalty(f, k))
                     + λ_pair × pair_gain(f, k)

    where:
        cov_gain  = Σ_j max(0, sim(f,j) - coverage_k(j))   [FL marginal coverage]
        div_penalty = mean sim to existing chunk members      [intra-chunk diversity]
        pair_gain = Σ_{m ∈ C_k} u(f,m)                      [pair utility to members]
        u(i,j) = sim(i,j)^α × (1-sim(i,j))^(1-α)           [peaks at sim = α]

    Chunk balance uses coverage sum (not combined score) for the weakest-chunk
    selection, consistent with FL maxmin.

    Args:
        sim_matrix: [S, S] pairwise cosine similarity (numpy or torch tensor).
        chunk_size: target number of frames per chunk.
        lambda_div: diversity penalty weight (0 = no penalty).
        lambda_pair: pair utility weight (0 = pure FL maxmin, higher = more pair utility).
        alpha: utility peak position (0.5 = balanced, 0.7 = prefer overlap).
        n_anchors: number of anchor frames for multi-anchor alignment.

    Returns:
        chunks: list of K lists of frame indices (primary anchor at position 0).
        anchors: list of anchor frame indices (anchors[0] = primary with highest coverage).
    """
    import numpy as np

    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().numpy()

    N = sim_matrix.shape[0]
    K = max(1, N // chunk_size)
    sim = np.clip(sim_matrix, 0, None).astype(np.float32)

    # Precompute pair utility matrix: u(i,j) = sim^α × (1-sim)^(1-α)
    eps = 1e-8
    sim_safe = np.clip(sim, eps, 1.0 - eps)
    utility = np.power(sim_safe, alpha) * np.power(1.0 - sim_safe, 1.0 - alpha)
    np.fill_diagonal(utility, 0.0)
    utility = utility.astype(np.float64)

    # State
    chunks = [[] for _ in range(K)]
    coverage = np.zeros((K, N), dtype=np.float32)
    chunk_scores = np.zeros(K, dtype=np.float64)     # coverage-based for balance
    chunk_gain_pair = np.zeros((K, N), dtype=np.float64)  # pair utility accumulator
    chunk_counts = np.zeros(K, dtype=np.int32)
    assigned = np.zeros(N, dtype=bool)

    total_steps = min(K * chunk_size, N)
    for step in range(total_steps):
        # 1. Pick the weakest non-full chunk (by coverage, same as FL maxmin)
        eligible_mask = chunk_counts < chunk_size
        if not eligible_mask.any():
            break
        scores_masked = np.where(eligible_mask, chunk_scores, np.inf)
        target_k = int(np.argmin(scores_masked))

        # 2. Coverage marginal gain
        current_cov = coverage[target_k]
        diff = sim - current_cov[np.newaxis, :]
        np.clip(diff, 0, None, out=diff)
        cov_gains = diff.sum(axis=1).astype(np.float64)
        cov_gains[assigned] = -1.0

        # 3. Diversity penalty on coverage gains
        if lambda_div > 0 and len(chunks[target_k]) > 0:
            penalty = sim[:, chunks[target_k]].mean(axis=1)
            mask = ~assigned & (cov_gains > 0)
            cov_gains[mask] *= (1.0 - lambda_div * penalty[mask])

        # 4. Pair utility gain
        if lambda_pair > 0 and len(chunks[target_k]) > 0:
            pair_gains = chunk_gain_pair[target_k].copy()
        else:
            pair_gains = np.zeros(N, dtype=np.float64)
        pair_gains[assigned] = 0.0

        # 5. Combined gain
        gains = cov_gains + lambda_pair * pair_gains
        gains[assigned] = -1.0

        best_frame = int(np.argmax(gains))

        # 6. Update state
        chunks[target_k].append(best_frame)
        np.maximum(coverage[target_k], sim[best_frame], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())
        chunk_gain_pair[target_k] += utility[best_frame]  # incremental pair utility
        chunk_counts[target_k] += 1
        assigned[best_frame] = True

    # Assign remaining frames to weakest chunk
    unassigned = np.where(~assigned)[0]
    for r in unassigned:
        target_k = int(np.argmin(chunk_scores))
        chunks[target_k].append(int(r))
        np.maximum(coverage[target_k], sim[r], out=coverage[target_k])
        chunk_scores[target_k] = float(coverage[target_k].sum())
        chunk_gain_pair[target_k] += utility[r]

    # Select anchors: uniform sampling from input sequence
    if n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (N - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        anchors = list(dict.fromkeys(anchors))  # remove duplicates, preserve order

    # Insert anchors into each chunk: primary at position 0, secondary uniform
    anchor_set = set(anchors)
    for k in range(K):
        non_anchor = [f for f in chunks[k] if f not in anchor_set]
        result = [anchors[0]]
        original_len = len(non_anchor)
        for a_idx, anc in enumerate(anchors[1:], 1):
            insert_pos = original_len * a_idx // n_anchors + (a_idx - 1)
            non_anchor.insert(insert_pos, anc)
        result.extend(non_anchor)
        chunks[k] = result

    return chunks, anchors


def fl_dual_split(sim_matrix, chunk_size: int, alpha: float = 0.7,
                  lambda_qual: float = 0.3, n_anchors: int = 1,
                  local_search_iters: int = 5):
    """Dual-objective FL partition: utility-based coverage + within-chunk quality.

    Replaces raw cosine similarity with pair utility u(s) = s^α(1-s)^(1-α)
    for BOTH the coverage (representation) and quality objectives.
    This rewards the "goldilocks zone" of similarity (good for both feature
    matching and triangulation) rather than near-identical frames.

    Phase 1: Greedy FL with utility-based coverage + quality gain.
    Phase 2: 2-opt local search to improve within-chunk pair quality.

    Args:
        sim_matrix: [S, S] pairwise cosine similarity (numpy or torch tensor).
        chunk_size: target number of frames per chunk.
        alpha: pair utility peak position (utility peaks at sim=alpha).
        lambda_qual: quality weight (0=pure utility coverage, 1=pure quality).
        n_anchors: number of anchor frames.
        local_search_iters: number of 2-opt local search iterations (0=skip).

    Returns:
        chunks: list of K lists of frame indices (primary anchor at position 0).
        anchors: list of anchor frame indices.
    """
    import numpy as np

    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().numpy()

    N = sim_matrix.shape[0]
    K = max(1, N // chunk_size)
    sim = np.clip(sim_matrix, 0, None).astype(np.float32)

    # Precompute pair utility matrix: u(s) = s^α × (1-s)^(1-α)
    eps = 1e-8
    sim_safe = np.clip(sim, eps, 1.0 - eps).astype(np.float64)
    U = np.power(sim_safe, alpha) * np.power(1.0 - sim_safe, 1.0 - alpha)
    np.fill_diagonal(U, 0.0)
    U = U.astype(np.float32)

    # --- Phase 1: Greedy FL with utility coverage + quality ---
    chunks = [[] for _ in range(K)]
    coverage_u = np.zeros((K, N), dtype=np.float32)   # max utility to any member
    chunk_scores = np.zeros(K, dtype=np.float64)       # utility coverage sum (for balance)
    qual_accum = np.zeros((K, N), dtype=np.float64)    # Σ_{m∈Ck} U(f,m) per candidate f
    chunk_counts = np.zeros(K, dtype=np.int32)
    assigned = np.zeros(N, dtype=bool)

    total_steps = min(K * chunk_size, N)
    for step in range(total_steps):
        # 1. Pick the weakest non-full chunk (by utility coverage, for balance)
        eligible_mask = chunk_counts < chunk_size
        if not eligible_mask.any():
            break
        scores_masked = np.where(eligible_mask, chunk_scores, np.inf)
        target_k = int(np.argmin(scores_masked))

        # 2. Utility coverage marginal gain (Repr component)
        current_cov_u = coverage_u[target_k]
        diff_u = U - current_cov_u[np.newaxis, :]
        np.clip(diff_u, 0, None, out=diff_u)
        repr_gains = diff_u.sum(axis=1).astype(np.float64)
        repr_gains[assigned] = -1.0

        # 3. Quality gain (pair utility to existing members)
        if lambda_qual > 0 and len(chunks[target_k]) > 0:
            qual_gains = qual_accum[target_k].copy()
        else:
            qual_gains = np.zeros(N, dtype=np.float64)
        qual_gains[assigned] = 0.0

        # 4. Combined gain
        gains = (1.0 - lambda_qual) * repr_gains + lambda_qual * qual_gains
        gains[assigned] = -1.0

        best_frame = int(np.argmax(gains))

        # 5. Update state
        chunks[target_k].append(best_frame)
        np.maximum(coverage_u[target_k], U[best_frame], out=coverage_u[target_k])
        chunk_scores[target_k] = float(coverage_u[target_k].sum())
        qual_accum[target_k] += U[best_frame]
        chunk_counts[target_k] += 1
        assigned[best_frame] = True

    # Assign remaining frames to weakest chunk
    unassigned = np.where(~assigned)[0]
    for r in unassigned:
        target_k = int(np.argmin(chunk_scores))
        chunks[target_k].append(int(r))
        np.maximum(coverage_u[target_k], U[r], out=coverage_u[target_k])
        chunk_scores[target_k] = float(coverage_u[target_k].sum())
        qual_accum[target_k] += U[r]

    # --- Phase 2: 2-opt local search for within-chunk quality (vectorized) ---
    import time as _time
    _t_ls_start = _time.time()
    if local_search_iters > 0 and K > 1:
        # chunk_util[k][f] = Σ_{m∈Ck} U(f, m)
        chunk_util = np.zeros((K, N), dtype=np.float64)
        for k in range(K):
            if len(chunks[k]) > 0:
                chunk_util[k] = U[:, chunks[k]].sum(axis=1)

        U64 = U.astype(np.float64)  # avoid repeated casts
        for _iteration in range(local_search_iters):
            improved = False
            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    # Exhaust all beneficial swaps for this (k1, k2) pair
                    while True:
                        F = np.array(chunks[k1], dtype=np.intp)
                        G = np.array(chunks[k2], dtype=np.intp)
                        if len(F) == 0 or len(G) == 0:
                            break
                        # delta[i,j] = net quality gain from swapping F[i] <-> G[j]
                        # = (cu_k2[F[i]] - cu_k1[F[i]]) + (cu_k1[G[j]] - cu_k2[G[j]]) - 2*U[F[i],G[j]]
                        row = chunk_util[k2][F] - chunk_util[k1][F]
                        col = chunk_util[k1][G] - chunk_util[k2][G]
                        delta = row[:, None] + col[None, :] - 2.0 * U64[np.ix_(F, G)]
                        best_flat = int(np.argmax(delta))
                        bi, bj = divmod(best_flat, len(G))
                        if delta[bi, bj] <= 1e-8:
                            break
                        f_idx, g_idx = int(F[bi]), int(G[bj])
                        chunks[k1][bi] = g_idx
                        chunks[k2][bj] = f_idx
                        chunk_util[k1] += U64[:, g_idx] - U64[:, f_idx]
                        chunk_util[k2] += U64[:, f_idx] - U64[:, g_idx]
                        improved = True
            if not improved:
                break
    _ls_time = _time.time() - _t_ls_start

    # --- Phase 3: Anchor selection & insertion ---
    # Recompute coverage on sim (not U) for anchor selection compatibility
    chunk_cov = np.zeros((K, N), dtype=np.float32)
    for k, chunk in enumerate(chunks):
        if len(chunk) > 0:
            chunk_cov[k] = sim[:, chunk].max(axis=1)

    if n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (N - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        anchors = list(dict.fromkeys(anchors))

    # Insert anchors into each chunk: primary at position 0, secondary uniform
    anchor_set = set(anchors)
    for k in range(K):
        non_anchor = [f for f in chunks[k] if f not in anchor_set]
        result = [anchors[0]]
        original_len = len(non_anchor)
        for a_idx, anc in enumerate(anchors[1:], 1):
            insert_pos = original_len * a_idx // n_anchors + (a_idx - 1)
            non_anchor.insert(insert_pos, anc)
        result.extend(non_anchor)
        chunks[k] = result

    return chunks, anchors, _ls_time


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        sampling_max_frames=0,
        sampling_lambda_div=0.0,
    ):
        super().__init__()

        self.sampling_max_frames = sampling_max_frames
        self.sampling_lambda_div = sampling_lambda_div
        self.sampling_method = "fl_maxmin"  # "fl_maxmin", "step", "maxpair", or "covpair"
        self.sampling_alpha = 0.5  # pair utility peak (0.5=balanced, 0.7=prefer overlap)
        self.sampling_lambda_pair = 0.5  # pair utility weight for covpair method
        self.sampling_n_anchors = 1  # number of anchor frames for multi-anchor alignment
        self.sampling_lambda_qual = 0.3  # fl_dual: quality weight
        self.sampling_local_search_iters = 5  # fl_dual: 2-opt iterations
        self.sampling_sim_lo = 0.5  # hard band lower bound
        self.sampling_sim_hi = 0.9  # hard band upper bound
        self.dino_batch_size = 256

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int, Optional[torch.Tensor]]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int, Optional[torch.Tensor]):
                The list of outputs from the attention blocks,
                the patch_start_idx indicating where patch tokens begin,
                and selected_indices [B, S'] if sampling was applied (None otherwise).
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_embed_output = self.patch_embed(images)

        selected_indices = None

        if isinstance(patch_embed_output, dict):
            patch_tokens = patch_embed_output["x_norm_patchtokens"]

            # --- FL Sampling: subsample frames using DINOv2 CLS tokens ---
            if self.sampling_max_frames > 0 and S > self.sampling_max_frames:
                cls_tokens = patch_embed_output["x_norm_clstoken"]  # [B*S, C]
                selected_indices = fl_greedy_sample(cls_tokens, B, S, self.sampling_max_frames)  # [B, S']
                S_new = selected_indices.shape[1]

                # Gather selected frames' patch tokens
                P_patches = patch_tokens.shape[1]
                C_dim = patch_tokens.shape[2]
                patch_tokens = patch_tokens.view(B, S, P_patches, C_dim)
                idx_expand = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(B, S_new, P_patches, C_dim)
                patch_tokens = torch.gather(patch_tokens, 1, idx_expand)  # [B, S_new, P_patches, C_dim]
                patch_tokens = patch_tokens.reshape(B * S_new, P_patches, C_dim)

                S = S_new
        else:
            patch_tokens = patch_embed_output

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        block4DPT_idx = [4, 11, 17, 23]

        for block_num in range(self.aa_block_num):
            need_intermediates = block_num in block4DPT_idx
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, need_intermediates=need_intermediates
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, need_intermediates=need_intermediates
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if need_intermediates:
                concat_inter = torch.cat(
                    [frame_intermediates[0], global_intermediates[0]], dim=-1
                )
                output_list.append(concat_inter)
                del concat_inter, frame_intermediates, global_intermediates

        return output_list, self.patch_start_idx, selected_indices

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, need_intermediates=False):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = [] if need_intermediates else None

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            if need_intermediates:
                intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, need_intermediates=False):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = [] if need_intermediates else None

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            if need_intermediates:
                intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def forward_dino(self, images, dino_batch_size=None):
        """Run DINOv2 backbone only, with mini-batching for large frame counts.

        DINOv2 runs under the caller's autocast context (bfloat16) so that
        patch tokens match the precision the aggregator transformer was trained on.
        For FL similarity, mean-pooled features are cast to float32 separately.

        Args:
            images: [B, S, 3, H, W] in [0, 1] range.
            dino_batch_size: max images per DINOv2 forward pass.

        Returns:
            patch_tokens: [B*S, P, C] on CPU (native dtype from DINOv2).
            pooled_tokens: [B*S, C] float32 on CPU (for FL similarity computation).
        """
        if dino_batch_size is None:
            dino_batch_size = self.dino_batch_size

        B, S, C_in, H, W = images.shape
        images_norm = (images - self._resnet_mean) / self._resnet_std
        images_flat = images_norm.view(B * S, C_in, H, W)

        all_patch_tokens = []
        all_pooled_tokens = []

        for start in range(0, B * S, dino_batch_size):
            end = min(start + dino_batch_size, B * S)
            batch = images_flat[start:end]
            # DINOv2 runs under caller's autocast (bfloat16) — matching external pipeline
            output = self.patch_embed(batch)

            if isinstance(output, dict):
                pt = output["x_norm_patchtokens"]
            else:
                pt = output

            all_patch_tokens.append(pt.cpu())
            # FL similarity: mean pool in float32 for deterministic FL decisions
            all_pooled_tokens.append(pt.float().mean(dim=1).cpu())

        patch_tokens = torch.cat(all_patch_tokens, dim=0)  # [B*S, P, C] on CPU
        pooled_tokens = torch.cat(all_pooled_tokens, dim=0)  # [B*S, C] float32 on CPU

        return patch_tokens, pooled_tokens

    def forward_transformer(self, patch_tokens, B, S, H, W, device):
        """Run transformer blocks on a chunk's patch tokens.

        The anchor frame must be at position 0 in the input so it receives
        the "first frame" special token (camera_token[0], register_token[0]).

        Args:
            patch_tokens: [B*S, P_patches, C] on device.
            B: batch size.
            S: number of frames in this chunk.
            H, W: original image dimensions (for RoPE position computation).
            device: GPU device.

        Returns:
            output_list: list of [B, S, P_total, 2C] tensors.
            patch_start_idx: int.
        """
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=device
            )

        if self.patch_start_idx > 0 and pos is not None:
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2, device=device, dtype=pos.dtype
            )
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        block4DPT_idx = [4, 11, 17, 23]

        for block_num in range(self.aa_block_num):
            need_intermediates = block_num in block4DPT_idx
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, need_intermediates=need_intermediates
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, need_intermediates=need_intermediates
                    )

            if need_intermediates:
                concat_inter = torch.cat(
                    [frame_intermediates[0], global_intermediates[0]], dim=-1
                )
                output_list.append(concat_inter)
                del concat_inter, frame_intermediates, global_intermediates

        return output_list, self.patch_start_idx


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
