"""RGB-D loading for the shared local VGGT/FastVGGT/sparse-vggt protocol."""
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np


SEVEN_SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
NRGBD_SCENES = (
    "breakfast_room", "complete_kitchen", "green_room", "grey_white_room",
    "kitchen", "morning_apartment", "staircase", "thin_geometry", "whiteroom",
)
IMAGE_SIZE = (518, 392)
SCANNET50_SCENE_LIST = Path(__file__).resolve().parent / "scannet50_scenes.txt"


@dataclass
class Frame:
    frame_id: int
    color: Path
    depth: Path | None
    c2w: np.ndarray


@dataclass
class Sequence:
    name: str
    scene: str
    path: Path
    frames: list[Frame]
    intrinsics: np.ndarray | None
    first_gt_pose: np.ndarray
    invalid_frames: int = 0


def read_pose(path):
    try:
        pose = np.loadtxt(path).astype(np.float64)
    except (OSError, ValueError):
        return None
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or abs(np.linalg.det(pose[:3, :3])) < 1e-8):
        return None
    return pose


def read_nrgbd_poses(path):
    """Retain frame indices when a pose is invalid; convert OpenGL c2w to OpenCV."""
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) % 4:
        raise ValueError(f"{path}: expected groups of four pose rows")
    poses = []
    for start in range(0, len(lines), 4):
        try:
            pose = np.array([list(map(float, line.split())) for line in lines[start:start + 4]])
            valid = (pose.shape == (4, 4) and np.isfinite(pose).all()
                     and np.allclose(pose[3], [0, 0, 0, 1])
                     and abs(np.linalg.det(pose[:3, :3])) > 1e-8)
        except ValueError:
            valid = False
        if valid:
            pose[:, 1:3] *= -1
            poses.append(pose)
        else:
            poses.append(None)
    return poses


def fixed_intrinsics(dataset):
    focal = 525.0 if dataset == "7scenes" else 554.2562584220408
    return np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1]], dtype=np.float32)


def discover_sequences(root, dataset, scenes=None, num_scenes=50, scene_list=None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if dataset == "7scenes":
        entries = []
        for scene in (scenes or SEVEN_SCENES):
            # A sequence can also be selected explicitly for a smoke run.
            if "/" in scene:
                entries.append((scene, root / scene))
                continue
            split = root / scene / "TestSplit.txt"
            for line in split.read_text().splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                number = re.search(r"\d+", line)
                if number is None:
                    raise ValueError(f"Invalid sequence in {split}: {line!r}")
                name = f"{scene}/seq-{int(number.group()):02d}"
                entries.append((name, root / name))
        return list(dict(entries).items())
    if dataset == "nrgbd":
        names = scenes or list(NRGBD_SCENES)
        return [(name, root / name) for name in names]

    # Use the published ScanNet-50 split unless a list or individual scenes are requested.
    manifest = Path(scene_list) if scene_list else SCANNET50_SCENE_LIST
    if scene_list or not scenes:
        names = [line.strip() for line in manifest.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        available = {}
        for name in names:
            path = root / name
            flat_path = root / Path(name).name
            if not path.is_dir() and flat_path.is_dir():
                path = flat_path
            available[name] = path
    else:
        available = {p.relative_to(root).as_posix(): p
                     for pattern in ("scene*", "train/scene*", "val/scene*")
                     for p in sorted(root.glob(pattern)) if p.is_dir()}
    if scenes:
        entries = []
        for scene in scenes:
            matches = [(name, path) for name, path in available.items()
                       if name == scene or path.name == Path(scene).name]
            if len(matches) != 1:
                raise ValueError(f"Scene {scene!r} has {len(matches)} matches under {root}")
            entries.append(matches[0])
        return entries
    entries = list(available.items())
    if len(entries) > num_scenes:
        # Uniform scene interval when a longer manifest is provided.
        entries = entries[::max(1, len(entries) // num_scenes)][:num_scenes]
    if not entries:
        raise ValueError(f"No ScanNet scenes found under {root}")
    return entries


def load_sequence(name, path, dataset):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Sequence directory does not exist: {path}")
    frames, invalid = [], 0
    first_gt_pose = None
    if dataset == "nrgbd":
        poses = read_nrgbd_poses(path / "poses.txt")
        candidates = [(i, path / "images" / f"img{i}.png",
                       path / "depth" / f"depth{i}.png", pose) for i, pose in enumerate(poses)]
        intrinsics = fixed_intrinsics(dataset)
    elif dataset == "7scenes":
        candidates = []
        for color in sorted(path.glob("frame-*.color.png")):
            stem = color.name.removesuffix(".color.png")
            candidates.append((int(stem.split("-")[-1]), color,
                               path / f"{stem}.depth.proj.png", read_pose(path / f"{stem}.pose.txt")))
        intrinsics = fixed_intrinsics(dataset)
    else:
        candidates = []
        pose_paths = sorted((path / "pose").glob("*.txt"), key=lambda p: int(p.stem))
        poses = {int(p.stem): read_pose(p) for p in pose_paths}
        first_gt_pose = next((p.copy() for p in poses.values() if p is not None), None)
        colors = sorted((p for p in (path / "color").iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")), key=lambda p: int(p.stem))
        for color in colors:
            candidates.append((int(color.stem), color,
                               path / "depth" / f"{color.stem}.png",
                               poses.get(int(color.stem))))
        intrinsics = np.loadtxt(path / "intrinsic" / "intrinsic_depth.txt")[:3, :3].astype(np.float32)
    for frame_id, color, depth, pose in candidates:
        if pose is None or not color.is_file() or (depth is not None and not depth.is_file()):
            invalid += 1
            continue
        frames.append(Frame(frame_id, color, depth, pose))
    if not frames:
        raise ValueError(f"{name}: no valid RGB/depth/pose frames (projected .depth.proj.png is required for 7Scenes)")
    if first_gt_pose is None:
        first_gt_pose = frames[0].c2w.copy()
    return Sequence(name, name.split("/")[0] if dataset == "7scenes" else name,
                    path, frames, intrinsics, first_gt_pose, invalid)


def select_frames(frames, n_frames):
    """Uniform sampling with both endpoints, as in the local eval_chunked scripts."""
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if not frames:
        return []
    indices = np.linspace(0, len(frames) - 1, min(n_frames, len(frames)), dtype=int)
    return [frames[i] for i in indices]


def read_depth(frame):
    depth = cv2.imread(str(frame.depth), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise ValueError(f"Cannot read depth map: {frame.depth}")
    depth = depth.astype(np.float32) / 1000.0
    depth[(~np.isfinite(depth)) | (depth <= 0) | (depth >= 10)] = 0
    return depth


def unproject(depth, c2w, K):
    y, x = np.indices(depth.shape, dtype=np.float32)
    points = np.stack(((x - K[0, 2]) * depth / K[0, 0],
                       (y - K[1, 2]) * depth / K[1, 1], depth), axis=-1)
    points = points @ c2w[:3, :3].astype(np.float32).T + c2w[:3, 3].astype(np.float32)
    return points, (depth > 0) & np.isfinite(points).all(axis=-1)


def ground_truth_maps(sequence, image_shape):
    height, width = image_shape
    w2c0 = np.linalg.inv(sequence.frames[0].c2w)
    points, masks = [], []
    for frame in sequence.frames:
        depth = read_depth(frame)
        native_h, native_w = depth.shape
        K = sequence.intrinsics.copy()
        K[0] *= width / native_w
        K[1] *= height / native_h
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        xyz, valid = unproject(depth, w2c0 @ frame.c2w, K)
        points.append(xyz)
        masks.append(valid)
    return np.stack(points), np.stack(masks)
