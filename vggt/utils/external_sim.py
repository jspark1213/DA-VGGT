"""External similarity feature extractors for the δ-source ablation
(rebuttal: Reviewer point on DINOv2 vs SALAD/MegaLoc).

Provides drop-in replacements for the DINOv2 mean-pool similarity matrix used
in eval_chunked_pose_*.py Phase 1b. Each `extract_*_features` returns an
(N, D) float CPU tensor; downstream code L2-normalizes and computes the
cosine similarity matrix.

Models:
  - SALAD (Izquierdo & Civera, CVPR 2024): DINOv2-ViT-B/14 + OT-based aggregator,
    descriptor dim 8448, official eval resolution 322×322.
    Bypasses VPRModel(pl.LightningModule) by building backbone+aggregator
    directly via models.helper, so we avoid pytorch_lightning/metric_learning
    training-time deps at inference.
  - MegaLoc (Berton & Masone, CVPRW 2025): DINOv2-ViT-B/14 + OT aggregator,
    descriptor dim 8448, evaluated here at 322×322 to match SALAD.
"""

import sys
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SALAD_REPO = "/workspace/salad"
MEGALOC_REPO = "/workspace/megaloc"

DEFAULT_SALAD_CKPT = "/workspace/VGGT-Long/weights/dino_salad.ckpt"

# Matches VGGT-Long's loop-detection setting (configs/base_config.yaml: SALAD.batch_size=32).
# Used as the default for both SALAD and MegaLoc to keep the comparison fair.
DEFAULT_RETRIEVAL_BATCH_SIZE = 32


def _imagenet_transform(image_size: Tuple[int, int]):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _load_images(image_paths: Sequence[str], image_size: Tuple[int, int]) -> List[torch.Tensor]:
    transform = _imagenet_transform(image_size)
    return [transform(Image.open(p).convert("RGB")) for p in image_paths]


# ---------------------------------------------------------------------------
# SALAD
# ---------------------------------------------------------------------------

def _build_salad_inference_model() -> nn.Module:
    """Build SALAD backbone+aggregator without VPRModel/pl.LightningModule."""
    if SALAD_REPO not in sys.path:
        sys.path.insert(0, SALAD_REPO)
    from models.helper import get_backbone, get_aggregator

    class SaladInference(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = get_backbone(
                "dinov2_vitb14",
                {"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
            )
            self.aggregator = get_aggregator(
                "SALAD",
                {"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
            )

        def forward(self, x):
            return self.aggregator(self.backbone(x))

    return SaladInference()


@torch.no_grad()
def extract_salad_features(
    image_paths: Sequence[str],
    ckpt_path: str = DEFAULT_SALAD_CKPT,
    device: str = "cuda",
    image_size: Tuple[int, int] = (322, 322),
    batch_size: int = DEFAULT_RETRIEVAL_BATCH_SIZE,
    autocast_dtype: torch.dtype = torch.float16,
    warmup: bool = True,
    return_timings: bool = False,
):
    """Returns (N, 8448) float CPU tensor of SALAD descriptors.

    If return_timings=True, also returns dict with: model_load, image_io, warmup,
    forward (steady-state, GPU-only), total.
    """
    import time
    timings = {}

    t0 = time.time()
    model = _build_salad_inference_model()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
    model.load_state_dict(sd, strict=True)
    model = model.eval().to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    timings["model_load"] = time.time() - t0

    t0 = time.time()
    tensors = _load_images(image_paths, image_size)
    timings["image_io"] = time.time() - t0

    if warmup and device == "cuda":
        t0 = time.time()
        dummy = torch.stack(tensors[: min(batch_size, len(tensors))]).to(device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            _ = model(dummy)
        torch.cuda.synchronize()
        timings["warmup"] = time.time() - t0
        del dummy
    else:
        timings["warmup"] = 0.0

    t0 = time.time()
    feats = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
            f = model(batch)
        feats.append(f.float().cpu())
    if device == "cuda":
        torch.cuda.synchronize()
    timings["forward"] = time.time() - t0

    del model
    torch.cuda.empty_cache()
    feats = torch.cat(feats, dim=0)
    timings["total"] = sum(timings.values())
    if return_timings:
        return feats, timings
    return feats


# ---------------------------------------------------------------------------
# MegaLoc
# ---------------------------------------------------------------------------

def _build_megaloc_model() -> nn.Module:
    if MEGALOC_REPO not in sys.path:
        sys.path.insert(0, MEGALOC_REPO)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from megaloc_model import MegaLoc

    model = MegaLoc()
    weights_path = hf_hub_download(repo_id="gberton/MegaLoc", filename="model.safetensors")
    sd = load_file(weights_path)
    model.load_state_dict(sd, strict=True)
    return model


@torch.no_grad()
def extract_megaloc_features(
    image_paths: Sequence[str],
    device: str = "cuda",
    image_size: Tuple[int, int] = (322, 322),
    batch_size: int = DEFAULT_RETRIEVAL_BATCH_SIZE,
    autocast_dtype: torch.dtype = torch.float16,
    warmup: bool = True,
    return_timings: bool = False,
):
    """Returns (N, 8448) float CPU tensor of MegaLoc descriptors.

    If return_timings=True, also returns dict with: model_load, image_io, warmup,
    forward (steady-state, GPU-only), total.
    """
    import time
    timings = {}

    t0 = time.time()
    model = _build_megaloc_model().eval().to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    timings["model_load"] = time.time() - t0

    t0 = time.time()
    tensors = _load_images(image_paths, image_size)
    timings["image_io"] = time.time() - t0

    if warmup and device == "cuda":
        t0 = time.time()
        dummy = torch.stack(tensors[: min(batch_size, len(tensors))]).to(device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            _ = model(dummy)
        torch.cuda.synchronize()
        timings["warmup"] = time.time() - t0
        del dummy
    else:
        timings["warmup"] = 0.0

    t0 = time.time()
    feats = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
            f = model(batch)
        feats.append(f.float().cpu())
    if device == "cuda":
        torch.cuda.synchronize()
    timings["forward"] = time.time() - t0

    del model
    torch.cuda.empty_cache()
    feats = torch.cat(feats, dim=0)
    timings["total"] = sum(timings.values())
    if return_timings:
        return feats, timings
    return feats


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compute_external_sim_matrix(
    image_paths: Sequence[str],
    sim_source: str,
    device: str = "cuda",
    image_size: Tuple[int, int] = (322, 322),
    batch_size: int = DEFAULT_RETRIEVAL_BATCH_SIZE,
):
    """Compute (N, N) cosine similarity matrix using the requested feature source.

    Args:
        image_paths: list of image file paths (length N).
        sim_source: 'salad' or 'megaloc'. For 'dinov2_mean', the existing
                    in-script path is used (this function is not called).

    Returns:
        sim_matrix: (N, N) numpy float64 cosine similarity (post L2-norm).
        feat_dim:   descriptor dimensionality.
    """
    import numpy as np

    if sim_source == "salad":
        feats, timings = extract_salad_features(
            image_paths, device=device, image_size=image_size,
            batch_size=batch_size, return_timings=True,
        )
    elif sim_source == "megaloc":
        feats, timings = extract_megaloc_features(
            image_paths, device=device, image_size=image_size,
            batch_size=batch_size, return_timings=True,
        )
    else:
        raise ValueError(f"Unknown sim_source: {sim_source}")

    feats = F.normalize(feats, dim=-1)
    sim_matrix = (feats @ feats.T).numpy().astype(np.float64)
    return sim_matrix, feats.shape[1], timings
