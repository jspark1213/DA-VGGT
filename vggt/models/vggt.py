# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import (
    Aggregator, fl_maxmin_split, fl_covpair_split, fl_dual_split,
    fl_maxmin_ls_split, fl_maxmin_ls_rawsim_split, fl_maxmin_ls_revsim_split,
    fl_maxmin_band_split, greedy_random_balanced_split, select_diverse_anchors,
    step_ls_split, step_ls_rawsim_split, step_ls_revsim_split,
    random_balanced_ls_split, random_balanced_ls_rawsim_split, random_balanced_ls_revsim_split,
)
from vggt.utils.rotation import quat_to_mat, mat_to_quat
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


def _pose_enc_to_se3(pose_enc):
    """Convert pose encoding [..., 9] to SE3 matrix [..., 4, 4].

    Args:
        pose_enc: tensor with last dim 9: [tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w].

    Returns:
        SE3 matrix with shape [..., 4, 4].
    """
    t = pose_enc[..., :3]
    quat = F.normalize(pose_enc[..., 3:7], dim=-1)
    R = quat_to_mat(quat)  # [..., 3, 3]
    shape = pose_enc.shape[:-1]
    T = torch.zeros(*shape, 4, 4, device=pose_enc.device, dtype=pose_enc.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def _se3_to_pose_enc(se3, focal):
    """Convert SE3 matrix [..., 4, 4] and focal [..., 2] to pose encoding [..., 9].

    Args:
        se3: SE3 matrix with shape [..., 4, 4].
        focal: focal values [fov_h, fov_w] with shape [..., 2].

    Returns:
        Pose encoding with shape [..., 9].
    """
    t = se3[..., :3, 3]
    R = se3[..., :3, :3]
    quat = mat_to_quat(R)  # [..., 4]
    return torch.cat([t, quat, focal], dim=-1)


def maxpair_split(sim_matrix, chunk_size, alpha=0.5, n_anchors=1):
    """MaxPair chunking: maximize intra-chunk pair utility for pose estimation.

    Targets the "goldilocks zone" of frame similarity — moderate overlap for
    feature matching + sufficient baseline for triangulation.

    Utility: u(i,j) = sim(i,j)^α × (1 - sim(i,j))^(1-α), peaks at sim = α.
    Uses FL maxmin structure (weakest-chunk-first greedy) with utility-based
    gains instead of coverage gains. Permutation-invariant: uses only the
    similarity matrix, no temporal ordering assumed.

    Args:
        sim_matrix: [S, S] numpy cosine similarity matrix.
        chunk_size: target frames per chunk.
        alpha: utility peak position (0.5 = balanced, 0.7 = prefer overlap).
        n_anchors: number of anchor frames for multi-anchor alignment.

    Returns:
        chunks: list of K lists of frame indices (primary anchor at position 0).
        anchors: list of anchor frame indices (anchors[0] = primary with highest coverage).
    """
    import numpy as np

    S = sim_matrix.shape[0]
    K = max(1, S // chunk_size)
    max_per_chunk = (S + K - 1) // K

    sim = np.clip(sim_matrix, 0, None).astype(np.float32)

    # Precompute utility matrix
    eps = 1e-8
    sim_safe = np.clip(sim, eps, 1.0 - eps)
    utility = np.power(sim_safe, alpha) * np.power(1.0 - sim_safe, 1.0 - alpha)
    np.fill_diagonal(utility, 0.0)
    utility = utility.astype(np.float64)

    # Fallback gains for cold start: total utility of each frame to all others
    total_utility = utility.sum(axis=1)  # [S]

    # State
    chunks = [[] for _ in range(K)]
    chunk_gain = np.zeros((K, S), dtype=np.float64)  # chunk_gain[k][f] = Σ_{m∈C_k} u(f,m)
    chunk_scores = np.zeros(K, dtype=np.float64)
    chunk_counts = np.zeros(K, dtype=np.int32)
    assigned = np.zeros(S, dtype=bool)

    for _ in range(S):
        # 1. Pick weakest non-full chunk
        eligible_mask = chunk_counts < max_per_chunk
        if not eligible_mask.any():
            break
        scores_masked = np.where(eligible_mask, chunk_scores, np.inf)
        target_k = int(np.argmin(scores_masked))

        # 2. Best unassigned frame for this chunk
        if chunk_counts[target_k] == 0:
            # Cold start: pick frame with highest total utility to all frames
            gains = total_utility.copy()
        else:
            gains = chunk_gain[target_k].copy()
        gains[assigned] = -1.0
        best_frame = int(np.argmax(gains))

        # 3. Update state
        chunks[target_k].append(best_frame)
        chunk_scores[target_k] += gains[best_frame]
        chunk_gain[target_k] += utility[best_frame]  # O(S) incremental update
        chunk_counts[target_k] += 1
        assigned[best_frame] = True

    # Assign any remaining frames to weakest chunk
    unassigned = np.where(~assigned)[0]
    for r in unassigned:
        target_k = int(np.argmin(chunk_scores))
        chunks[target_k].append(int(r))
        chunk_scores[target_k] += chunk_gain[target_k, r]
        chunk_gain[target_k] += utility[r]

    # Anchor selection: uniform sampling from input sequence
    if n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (S - 1) / (n_anchors - 1)) for i in range(n_anchors)]
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


def sequential_split(S, chunk_size):
    """Sequential chunking with 2-frame overlap.

    Chunk 0 gets frames [0, chunk_size).  Chunk k>0 gets the last 2 frames
    of the previous chunk (shared) prepended to its own range, giving
    chunk_size + 2 frames per chunk (except chunk 0).

    Example (S=10, chunk_size=5):
        Chunk 0: [0,1,2,3,4]
        Chunk 1: [3,4,5,6,7,8,9]

    Example (S=15, chunk_size=5):
        Chunk 0: [0,1,2,3,4]
        Chunk 1: [3,4,5,6,7,8,9]
        Chunk 2: [8,9,10,11,12,13,14]

    Args:
        S: total number of frames.
        chunk_size: frames per chunk (chunk 0 size; later chunks are chunk_size+2).

    Returns:
        chunks: list of K lists of frame indices.
    """
    K = (S + chunk_size - 1) // chunk_size
    chunks = []
    for k in range(K):
        start = k * chunk_size
        end = min((k + 1) * chunk_size, S)
        indices = list(range(start, end))
        if k > 0:
            indices = [start - 2, start - 1] + indices   # prepend 2 shared frames
        chunks.append(indices)
    return chunks


def step_sampling_split(S, chunk_size, n_anchors=1, anchors_override=None):
    """Deterministic step-based frame splitting (matching split_frames_with_shared_first).

    Splits S frames into K batches where each batch shares anchor frames.
    Remaining frames are distributed via strided interleaving.

    Args:
        S: total number of frames.
        chunk_size: target frames per chunk (excluding shared anchor frames).
        n_anchors: number of anchor frames (default 1, always includes frame 0).
        anchors_override: optional list of anchor indices to use instead of the
            default uniform-by-index selection. anchors_override[0] becomes the
            primary anchor (inserted at position 0 of every chunk).

    Returns:
        chunks: list of K lists of frame indices, each starting with anchor[0].
        anchors: list of anchor frame indices.
    """
    stride = max(1, S // chunk_size)
    num_batches = stride

    if anchors_override is not None:
        anchors = list(anchors_override)
    elif n_anchors <= 1:
        anchors = [0]
    else:
        anchors = [round(i * (S - 1) / (n_anchors - 1)) for i in range(n_anchors)]
        anchors = list(dict.fromkeys(anchors))  # remove duplicates, preserve order

    n_eff = max(1, len(anchors))
    anchor_set = set(anchors)

    chunks = []
    for batch_idx in range(num_batches):
        non_anchor = []
        frame_idx = 1 + batch_idx
        while frame_idx < S:
            if frame_idx not in anchor_set:
                non_anchor.append(frame_idx)
            frame_idx += stride

        result = [anchors[0]]
        original_len = len(non_anchor)
        for a_idx, anc in enumerate(anchors[1:], 1):
            insert_pos = original_len * a_idx // n_eff + (a_idx - 1)
            non_anchor.insert(insert_pos, anc)
        result.extend(non_anchor)
        chunks.append(result)

    return chunks, anchors


def _procrustes_se3(src_se3_list, dst_se3_list):
    """SVD-based Procrustes alignment from multiple SE3 correspondences.

    For each pair of SE3 matrices, extracts camera center and forward direction
    endpoint, then finds the best rigid transform T such that T @ src ≈ dst.

    Args:
        src_se3_list: list of (4, 4) tensors, source SE3 matrices.
        dst_se3_list: list of (4, 4) tensors, destination SE3 matrices.

    Returns:
        T_align: (4, 4) rigid transform matrix.
    """
    src_points = []
    dst_points = []
    for T_src, T_dst in zip(src_se3_list, dst_se3_list):
        # Camera center
        src_points.append(T_src[:3, 3])
        dst_points.append(T_dst[:3, 3])
        # Forward direction endpoint (camera center + z-axis)
        src_points.append(T_src[:3, 3] + T_src[:3, 2])
        dst_points.append(T_dst[:3, 3] + T_dst[:3, 2])

    # Disable autocast: SVD and matmuls here must stay float32
    with torch.amp.autocast('cuda', enabled=False):
        src_pts = torch.stack(src_points).float()  # [2M, 3]
        dst_pts = torch.stack(dst_points).float()  # [2M, 3]

        # Center the point clouds
        c_src = src_pts.mean(dim=0)
        c_dst = dst_pts.mean(dim=0)
        src_c = src_pts - c_src
        dst_c = dst_pts - c_dst

        # SVD
        H = src_c.T @ dst_c  # [3, 3]
        U, _, Vt = torch.linalg.svd(H)

        # Correct for reflection
        d = torch.det(Vt.T @ U.T)
        sign = torch.ones(3, device=src_pts.device, dtype=src_pts.dtype)
        sign[2] = 1.0 if d >= 0 else -1.0
        R = Vt.T @ torch.diag(sign) @ U.T

        t = c_dst - R @ c_src

        T = torch.eye(4, device=src_pts.device, dtype=src_pts.dtype)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 sampling_max_frames=0, sampling_lambda_div=0.0):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            sampling_max_frames=sampling_max_frames,
            sampling_lambda_div=sampling_lambda_div,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # Multi-chunk inference when S exceeds sampling_max_frames
        if (self.aggregator.sampling_max_frames > 0
                and images.shape[1] > self.aggregator.sampling_max_frames
                and self.camera_head is not None):
            return self._forward_chunked(images, query_points)

        aggregated_tokens_list, patch_start_idx, selected_indices = self.aggregator(images)

        # If sampling was applied, subsample images to match
        if selected_indices is not None:
            B_img, S_img = images.shape[:2]
            S_new = selected_indices.shape[1]
            idx = selected_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, S', 1, 1, 1]
            idx = idx.expand(B_img, S_new, *images.shape[2:])  # [B, S', 3, H, W]
            images = torch.gather(images, 1, idx)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        if selected_indices is not None:
            predictions["selected_indices"] = selected_indices

        return predictions

    def _forward_chunked(self, images, query_points=None):
        """Multi-chunk forward: DINOv2 all → FL maxmin split → K × (transformer + camera_head) → SE3 align.

        Processes all images through DINOv2 once (mini-batched), then splits
        into balanced chunks using facility-location maxmin. Each chunk is
        processed through the transformer and camera head sequentially, with
        VRAM cleanup after each chunk. Finally, all chunk predictions are
        aligned via SE3 using a shared anchor frame.

        The anchor frame receives the "first frame" special token in every chunk
        to minimize inter-chunk variance.

        Args:
            images: [B, S, 3, H, W] in [0, 1] (B must be 1).
            query_points: ignored in chunked mode.

        Returns:
            dict with:
                - pose_enc: [1, S, 9] aligned pose encodings for all S frames.
                - chunk_frame_indices: list of K lists of frame indices per chunk.
                - anchor: global anchor frame index.
                - num_chunks: number of chunks K.
                - timing: dict with per-phase timing in seconds.
                - images: original images (inference only).
        """
        import time as _time

        B, S, C_in, H, W = images.shape
        device = images.device
        chunk_size = self.aggregator.sampling_max_frames
        n_anchors = self.aggregator.sampling_n_anchors
        assert B == 1, "Chunked inference only supports batch size 1"

        timing = {}

        # Track peak VRAM
        torch.cuda.reset_peak_memory_stats(device)

        # 1. DINOv2 on all images (mini-batched, patch tokens stored on CPU)
        torch.cuda.synchronize()
        t0 = _time.time()
        patch_tokens_cpu, pooled_tokens = self.aggregator.forward_dino(images)
        torch.cuda.synchronize()
        timing['dino'] = _time.time() - t0
        timing['peak_vram_dino_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        # patch_tokens_cpu: [S, P, C] on CPU (native dtype, e.g. bfloat16)
        # pooled_tokens: [S, C] float32 on CPU (for FL similarity)

        # 2. Chunk assignment (FL maxmin or step sampling)
        import numpy as np
        sampling_method = self.aggregator.sampling_method

        # Methods that don't need the similarity matrix at all
        _no_sim_methods = {"sequential", "step", "random_balanced"}

        if sampling_method in _no_sim_methods:
            # Skip sim computation entirely — not needed for these methods
            del pooled_tokens
            sim = None
            t_sim = 0.0
        else:
            torch.cuda.synchronize()
            t0 = _time.time()
            feats = F.normalize(pooled_tokens, dim=-1)  # already float32 CPU
            sim = (feats @ feats.T).numpy()  # [S, S] numpy on CPU
            del pooled_tokens, feats
            t_sim = _time.time() - t0

        t0 = _time.time()
        t_ls = 0.0  # local search time (0 for methods without LS)
        if sampling_method == "sequential":
            chunks = sequential_split(S, chunk_size)
            anchors = [0]  # nominal; chained alignment uses shared frames
            t_fl = _time.time() - t0
        elif sampling_method == "step":
            chunks, anchors = step_sampling_split(S, chunk_size, n_anchors=n_anchors)
            t_fl = _time.time() - t0
        elif sampling_method == "maxpair":
            chunks, anchors = maxpair_split(sim, chunk_size, alpha=self.aggregator.sampling_alpha, n_anchors=n_anchors)
            t_fl = _time.time() - t0
        elif sampling_method == "covpair":
            chunks, anchors = fl_covpair_split(
                sim, chunk_size,
                lambda_div=self.aggregator.sampling_lambda_div,
                lambda_pair=self.aggregator.sampling_lambda_pair,
                alpha=self.aggregator.sampling_alpha,
                n_anchors=n_anchors,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "fl_dual":
            chunks, anchors, t_ls = fl_dual_split(
                sim, chunk_size,
                alpha=self.aggregator.sampling_alpha,
                lambda_qual=self.aggregator.sampling_lambda_qual,
                n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "random_balanced":
            chunks, anchors = greedy_random_balanced_split(
                np.empty((S, 0)),  # dummy — sim not needed, only .shape[0] read
                chunk_size, n_anchors=n_anchors,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "fl_maxmin_ls":
            chunks, anchors, t_ls = fl_maxmin_ls_split(
                sim, chunk_size,
                alpha=self.aggregator.sampling_alpha,
                lambda_qual=self.aggregator.sampling_lambda_qual,
                n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "fl_maxmin_band":
            chunks, anchors, t_ls = fl_maxmin_band_split(
                sim, chunk_size,
                sim_lo=self.aggregator.sampling_sim_lo,
                sim_hi=self.aggregator.sampling_sim_hi,
                n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "step_ls":
            chunks, anchors, t_ls = step_ls_split(
                S, chunk_size, sim,
                alpha=self.aggregator.sampling_alpha,
                n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "random_balanced_ls":
            chunks, anchors, t_ls = random_balanced_ls_split(
                sim, chunk_size,
                alpha=self.aggregator.sampling_alpha,
                n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "fl_maxmin_ls_rawsim":
            chunks, anchors, t_ls = fl_maxmin_ls_rawsim_split(
                sim, chunk_size, n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "fl_maxmin_ls_revsim":
            chunks, anchors, t_ls = fl_maxmin_ls_revsim_split(
                sim, chunk_size, n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "step_ls_rawsim":
            chunks, anchors, t_ls = step_ls_rawsim_split(
                S, chunk_size, sim, n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "step_ls_revsim":
            chunks, anchors, t_ls = step_ls_revsim_split(
                S, chunk_size, sim, n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "random_ls_rawsim":
            chunks, anchors, t_ls = random_balanced_ls_rawsim_split(
                sim, chunk_size, n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        elif sampling_method == "random_ls_revsim":
            chunks, anchors, t_ls = random_balanced_ls_revsim_split(
                sim, chunk_size, n_anchors=n_anchors,
                local_search_iters=self.aggregator.sampling_local_search_iters,
            )
            t_fl = _time.time() - t0
        else:  # fl_maxmin (default)
            chunks, anchors = fl_maxmin_split(sim, chunk_size, lambda_div=self.aggregator.sampling_lambda_div, n_anchors=n_anchors)
            t_fl = _time.time() - t0

        timing['sampling_method'] = sampling_method
        timing['sampling_sim'] = t_sim
        timing['sampling_fl'] = t_fl
        timing['sampling_ls'] = t_ls
        timing['sampling_init'] = t_fl - t_ls
        timing['sampling_total'] = t_sim + t_fl

        K = len(chunks)
        sim_matrix_np = sim  # keep for chunking quality metrics (None for no-sim methods)
        if sim is not None:
            del sim
        torch.cuda.empty_cache()
        timing['peak_vram_sampling_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # 3. Sequential transformer + camera_head (+ optional depth_head) per chunk
        chunk_pose_encs = []     # list of [1, S_k, 9] on CPU
        chunk_frame_indices = []  # list of lists of frame indices
        chunk_times = []
        run_depth = self.depth_head is not None
        chunk_depths = [] if run_depth else None      # list of [1, S_k, H', W', 1] on CPU
        chunk_depth_confs = [] if run_depth else None  # list of [1, S_k, H', W'] on CPU
        depth_head_total = 0.0

        torch.cuda.synchronize()
        t_chunks_start = _time.time()

        for k in range(K):
            torch.cuda.synchronize()
            t_chunk_k = _time.time()

            chunk_indices = chunks[k]  # anchor at position 0
            S_k = len(chunk_indices)

            # Gather this chunk's patch tokens and move to GPU
            idx = torch.tensor(chunk_indices, dtype=torch.long)
            pt_chunk = patch_tokens_cpu[idx].to(device)  # [S_k, P, C]

            # Run transformer (B=1, anchor at position 0 gets "first frame" token)
            output_list, patch_start_idx = self.aggregator.forward_transformer(
                pt_chunk, B=1, S=S_k, H=H, W=W, device=device
            )
            del pt_chunk

            # Run camera head (in float32 for precision)
            with torch.amp.autocast('cuda', enabled=False):
                pose_enc_list = self.camera_head(output_list)
                pose_enc = pose_enc_list[-1]  # [1, S_k, 9]

            chunk_pose_encs.append(pose_enc.cpu())
            chunk_frame_indices.append(chunk_indices)

            # Run depth head if enabled
            if run_depth:
                torch.cuda.synchronize()
                t_depth_k = _time.time()
                chunk_imgs = images[:, chunk_indices]  # [1, S_k, 3, H, W]
                depth_k, depth_conf_k = self.depth_head(
                    output_list, images=chunk_imgs, patch_start_idx=patch_start_idx
                )
                chunk_depths.append(depth_k.cpu())
                chunk_depth_confs.append(depth_conf_k.cpu())
                del chunk_imgs, depth_k, depth_conf_k
                torch.cuda.synchronize()
                depth_head_total += _time.time() - t_depth_k

            # VRAM cleanup after each chunk
            del output_list, pose_enc_list, pose_enc
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            chunk_times.append(_time.time() - t_chunk_k)

        torch.cuda.synchronize()
        timing['transformer_total'] = _time.time() - t_chunks_start
        timing['transformer_per_chunk'] = chunk_times
        timing['depth_head_total'] = depth_head_total
        timing['peak_vram_transformer_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        del patch_tokens_cpu

        # 4. SE3 alignment across chunks
        torch.cuda.synchronize()
        t0 = _time.time()

        all_pose_encs = torch.zeros(S, 9, device=device)
        anchor_set = set(anchors)

        if sampling_method == "sequential":
            # Chained Procrustes alignment via 2 shared frames between
            # consecutive chunks.  Chunk 0 is reference.  For chunk k>0,
            # positions 0-1 are the shared frames (last 2 of chunk k-1).
            pe_0 = chunk_pose_encs[0][0].to(device).float()
            T_0 = _pose_enc_to_se3(pe_0)
            focals_0 = pe_0[:, 7:9]
            all_pose_encs[chunk_frame_indices[0]] = _se3_to_pose_enc(T_0, focals_0)

            # Last 2 frames of chunk 0 in aligned coords
            prev_shared_se3 = [T_0[-2], T_0[-1]]  # list of 2 × [4, 4]

            for k in range(1, K):
                pe_k = chunk_pose_encs[k][0].to(device).float()
                indices_k = chunk_frame_indices[k]

                # Shared frames are at positions 0, 1 of this chunk
                T_shared_src = [_pose_enc_to_se3(pe_k[0]), _pose_enc_to_se3(pe_k[1])]
                T_align = _procrustes_se3(T_shared_src, prev_shared_se3)

                # Align non-shared frames (positions 2+)
                if len(pe_k) > 2:
                    T_rest = _pose_enc_to_se3(pe_k[2:])
                    T_aligned = T_align @ T_rest
                    focals_rest = pe_k[2:, 7:9]
                    all_pose_encs[indices_k[2:]] = _se3_to_pose_enc(T_aligned, focals_rest)

                # Update: last 2 frames of this chunk in aligned coords
                prev_shared_se3 = [
                    T_align @ _pose_enc_to_se3(pe_k[-2]),
                    T_align @ _pose_enc_to_se3(pe_k[-1]),
                ]

        elif n_anchors <= 1:
            # Single-anchor alignment (backward compatible) — batched
            ref_pose = chunk_pose_encs[0][0, 0].to(device).float()  # [9]
            T_ref = _pose_enc_to_se3(ref_pose)  # [4, 4]

            for k in range(K):
                pe_k = chunk_pose_encs[k][0].to(device).float()  # [S_k, 9]
                indices_k = chunk_frame_indices[k]

                if k == 0:
                    # Reference chunk: identity alignment, batch all frames
                    T_k = _pose_enc_to_se3(pe_k)        # [S_k, 4, 4]
                    focals_k = pe_k[:, 7:9]              # [S_k, 2]
                    all_pose_encs[indices_k] = _se3_to_pose_enc(T_k, focals_k)
                else:
                    # Align non-anchor frames (anchor at position 0 is skipped)
                    T_anchor_k = _pose_enc_to_se3(pe_k[0])  # [4, 4]
                    T_align = T_ref @ torch.linalg.inv(T_anchor_k)
                    pe_rest = pe_k[1:]                   # [S_k-1, 9]
                    T_rest = _pose_enc_to_se3(pe_rest)   # [S_k-1, 4, 4]
                    T_aligned = T_align @ T_rest         # [S_k-1, 4, 4]
                    focals_rest = pe_rest[:, 7:9]        # [S_k-1, 2]
                    all_pose_encs[indices_k[1:]] = _se3_to_pose_enc(T_aligned, focals_rest)
        else:
            # Multi-anchor Procrustes alignment — batched
            ref_anchor_se3 = None

            for k in range(K):
                pe_k = chunk_pose_encs[k][0].to(device).float()  # [S_k, 9]
                indices_k = chunk_frame_indices[k]

                # Batch-extract anchor SE3 matrices from this chunk
                anchor_positions = [indices_k.index(anc) for anc in anchors]
                anchor_se3_batch = _pose_enc_to_se3(pe_k[anchor_positions])  # [n_anchors, 4, 4]
                chunk_anchor_se3 = list(anchor_se3_batch.unbind(0))

                if k == 0:
                    # Reference chunk: identity alignment, store anchor poses
                    ref_anchor_se3 = chunk_anchor_se3
                    T_k = _pose_enc_to_se3(pe_k)        # [S_k, 4, 4]
                    focals_k = pe_k[:, 7:9]              # [S_k, 2]
                    all_pose_encs[indices_k] = _se3_to_pose_enc(T_k, focals_k)
                else:
                    # Procrustes alignment, batch non-anchor frames
                    T_align = _procrustes_se3(chunk_anchor_se3, ref_anchor_se3)
                    non_anchor = [j for j, idx in enumerate(indices_k) if idx not in anchor_set]
                    if non_anchor:
                        non_anchor_indices = [indices_k[j] for j in non_anchor]
                        pe_na = pe_k[non_anchor]             # [M, 9]
                        T_na = _pose_enc_to_se3(pe_na)       # [M, 4, 4]
                        T_aligned = T_align @ T_na           # [M, 4, 4]
                        focals_na = pe_na[:, 7:9]            # [M, 2]
                        all_pose_encs[non_anchor_indices] = _se3_to_pose_enc(T_aligned, focals_na)

        torch.cuda.synchronize()
        timing['alignment'] = _time.time() - t0
        timing['peak_vram_total_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        predictions = {
            "pose_enc": all_pose_encs.unsqueeze(0),  # [1, S, 9]
            "chunk_frame_indices": chunk_frame_indices,
            "anchor": anchors[0],  # backward compat
            "anchors": anchors,
            "num_chunks": K,
            "timing": timing,
            "sim_matrix": sim_matrix_np,  # [S, S] numpy float32 for quality metrics
        }

        # 5. Assemble per-frame depth maps with anchor-based scale alignment (on GPU)
        if run_depth:
            torch.cuda.synchronize()
            t_depth_asm = _time.time()

            d0 = chunk_depths[0]  # [1, S_k0, H_d, W_d, 1] float32 CPU
            H_d, W_d = d0.shape[2], d0.shape[3]

            all_depth = torch.zeros(S, H_d, W_d, 1, device=device)
            all_depth_conf = torch.zeros(S, H_d, W_d, device=device)
            depth_chunk_scales = []

            if sampling_method == "sequential":
                # Chained depth scale alignment via shared frames
                prev_shared_depth = None  # last frame's depth of previous chunk (scaled)
                cumulative_scale = 1.0

                for k in range(K):
                    depth_k = chunk_depths[k][0].to(device)
                    conf_k = chunk_depth_confs[k][0].to(device)
                    indices_k = chunk_frame_indices[k]

                    if k == 0:
                        scale_k = 1.0
                        prev_shared_depth = depth_k[-1]  # last frame depth
                    else:
                        # Shared frame is at position 0 of this chunk
                        chunk_shared_depth = depth_k[0]
                        valid = (prev_shared_depth > 1e-6) & (chunk_shared_depth > 1e-6)
                        if valid.sum() > 0:
                            local_scale = float(torch.median(
                                prev_shared_depth[valid] / chunk_shared_depth[valid]
                            ))
                        else:
                            local_scale = 1.0
                        cumulative_scale *= local_scale
                        scale_k = cumulative_scale
                        prev_shared_depth = depth_k[-1] * scale_k

                    depth_chunk_scales.append(scale_k)

                    if k == 0:
                        idx = torch.tensor(indices_k, dtype=torch.long, device=device)
                        all_depth[idx] = depth_k * scale_k
                        all_depth_conf[idx] = conf_k
                    else:
                        # Skip shared frame (position 0), already written by previous chunk
                        if len(indices_k) > 1:
                            pos = torch.arange(1, len(indices_k), dtype=torch.long, device=device)
                            idx = torch.tensor(indices_k[1:], dtype=torch.long, device=device)
                            all_depth[idx] = depth_k[pos] * scale_k
                            all_depth_conf[idx] = conf_k[pos]

                    del depth_k, conf_k
            else:
                primary_anchor = anchors[0]
                ref_anchor_depth = chunk_depths[0][0, 0].to(device)  # [H_d, W_d, 1]

                for k in range(K):
                    depth_k = chunk_depths[k][0].to(device)       # [S_k, H_d, W_d, 1] → GPU
                    conf_k = chunk_depth_confs[k][0].to(device)   # [S_k, H_d, W_d] → GPU
                    indices_k = chunk_frame_indices[k]

                    if k == 0:
                        scale_k = 1.0
                    else:
                        anchor_pos = indices_k.index(primary_anchor)
                        chunk_anchor_depth = depth_k[anchor_pos]
                        valid = (ref_anchor_depth > 1e-6) & (chunk_anchor_depth > 1e-6)
                        if valid.sum() > 0:
                            scale_k = float(torch.median(
                                ref_anchor_depth[valid] / chunk_anchor_depth[valid]
                            ))
                        else:
                            scale_k = 1.0

                    depth_chunk_scales.append(scale_k)

                    if k == 0:
                        idx = torch.tensor(indices_k, dtype=torch.long, device=device)
                        all_depth[idx] = depth_k * scale_k
                        all_depth_conf[idx] = conf_k
                    else:
                        mask = [j for j, fi in enumerate(indices_k) if fi not in anchor_set]
                        if mask:
                            pos = torch.tensor(mask, dtype=torch.long, device=device)
                            idx = torch.tensor([indices_k[j] for j in mask], dtype=torch.long, device=device)
                            all_depth[idx] = depth_k[pos] * scale_k
                            all_depth_conf[idx] = conf_k[pos]

                    del depth_k, conf_k

            predictions["depth"] = all_depth.unsqueeze(0).cpu()       # [1, S, H_d, W_d, 1]
            predictions["depth_conf"] = all_depth_conf.unsqueeze(0).cpu()  # [1, S, H_d, W_d]
            predictions["depth_chunk_scales"] = depth_chunk_scales

            del all_depth, all_depth_conf, chunk_depths, chunk_depth_confs
            torch.cuda.synchronize()
            timing['depth_assembly'] = _time.time() - t_depth_asm

        if not self.training:
            predictions["images"] = images

        return predictions

