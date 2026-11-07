"""Shared DA-VGGT inference, extracted from the released 7Scenes evaluator.

The partitioning, pseudo-pose propagation and anchor alignment are unchanged.
Image tensors may be supplied to use a dataset-specific evaluation preprocessor.
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.models.aggregator import (
    random_balanced_ls_revsim_split,
    compute_pseudo_poses, compute_pose_weight_matrix, rechunk_with_pose_weights,
)
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.multi_anchor import (
    select_anchors, compute_chunk_alignment, apply_chunk_alignment,
    multi_anchor_depth_scale,
)


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
        path = Path(model_path)
        if path.is_dir():
            model = VGGT.from_pretrained(str(path))
        else:
            model = VGGT()
            if path.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(str(path), device="cpu")
            else:
                state = torch.load(path, map_location="cpu", weights_only=True)
                if "model" in state:
                    state = state["model"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            model.load_state_dict(state)
            del state

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
    images = (image_paths if isinstance(image_paths, torch.Tensor)
              else load_and_preprocess_images(image_paths)).to(device)
    img_shape = images.shape[-2:]

    torch.cuda.synchronize()
    t_total_start = time.time()

    with torch.no_grad():
        if dtype in (torch.bfloat16, torch.float16):
            images = images.to(dtype)
        with torch.amp.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
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
            with torch.amp.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
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
        with torch.amp.autocast('cuda', enabled=False):
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
            with torch.amp.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
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
        with torch.amp.autocast('cuda', enabled=False):
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
    images = (image_paths if isinstance(image_paths, torch.Tensor)
              else load_and_preprocess_images(image_paths)).to(device)
    if images.dim() == 4:
        images = images.unsqueeze(0)
    S = images.shape[1]
    img_shape = images.shape[-2:]

    # ---- Phase 1a: DINOv2 on all images ----
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
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
        with torch.amp.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
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
    with torch.amp.autocast('cuda', enabled=False):
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
