"""The reconstruction protocol shared by the local eval_chunked_3d scripts."""
import numpy as np
from scipy.spatial import cKDTree

MAX_POINTS = 999_999
ALIGNMENT_POINTS = 30_000
ICP_THRESHOLD = 0.1


def umeyama_alignment(source, target):
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Sim(3) requires at least three paired 3D points")
    src_center, dst_center = source.mean(0), target.mean(0)
    src, dst = source - src_center, target - dst_center
    variance = np.square(src).sum() / len(src)
    if variance < 1e-12:
        raise ValueError("Degenerate predicted point cloud: zero spatial variance")
    u, singular, vt = np.linalg.svd(dst.T @ src / len(src))
    signs = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        signs[-1] = -1
    rotation = (u * signs) @ vt
    scale = float(singular @ signs / variance)
    translation = dst_center - scale * rotation @ src_center
    return scale, rotation, translation


def point_cloud(points, colors=None):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud


def point_metrics(pred_cloud, gt_cloud):
    pred, gt = np.asarray(pred_cloud.points), np.asarray(gt_cloud.points)
    if min(len(pred), len(gt)) < 3:
        raise ValueError("At least three valid points are required in each cloud")
    pred_cloud.estimate_normals()
    gt_cloud.estimate_normals()
    pred_normals, gt_normals = np.asarray(pred_cloud.normals), np.asarray(gt_cloud.normals)
    pred_dist, pred_nn = cKDTree(gt).query(pred, workers=-1)
    gt_dist, gt_nn = cKDTree(pred).query(gt, workers=-1)
    nc1 = np.abs((gt_normals[pred_nn] * pred_normals).sum(-1))
    nc2 = np.abs((pred_normals[gt_nn] * gt_normals).sum(-1))
    acc, comp = float(pred_dist.mean()), float(gt_dist.mean())
    return {
        "acc": acc, "acc_med": float(np.median(pred_dist)),
        "comp": comp, "comp_med": float(np.median(gt_dist)),
        "nc1": float(nc1.mean()), "nc1_med": float(np.median(nc1)),
        "nc2": float(nc2.mean()), "nc2_med": float(np.median(nc2)),
        "nc": float((nc1.mean() + nc2.mean()) / 2),
        "cd": (acc + comp) / 2, "cd_mean": (acc + comp) / 2, "cd_sum": acc + comp,
        "n_pts": len(pred), "n_gt_pts": len(gt),
    }


def evaluate_rgbd(predicted, truth, valid, confidence, rng, conf_thresh=0.0, colors=None):
    """Evaluate paired RGB-D maps in the first selected camera coordinate system."""
    import open3d as o3d
    if predicted.shape != truth.shape or predicted.shape[:-1] != valid.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {predicted.shape}, {truth.shape}, {valid.shape}")
    valid = valid & np.isfinite(truth).all(-1) & np.isfinite(predicted).all(-1)
    if conf_thresh > 0:
        valid &= confidence > conf_thresh
    height, width = valid.shape[-2:]
    if min(height, width) < 224:
        raise ValueError("RGB-D evaluation requires at least a 224x224 image")
    crop = (slice(None), slice(height // 2 - 112, height // 2 + 112),
            slice(width // 2 - 112, width // 2 + 112))
    mask = valid[crop]
    pred_pts, gt_pts = predicted[crop][mask], truth[crop][mask]
    rgb = None if colors is None else colors[crop][mask]
    if len(pred_pts) < 100:
        raise ValueError(f"Too few valid reconstruction points: {len(pred_pts)}")

    idx = rng.choice(len(pred_pts), ALIGNMENT_POINTS, replace=False) if len(pred_pts) > ALIGNMENT_POINTS else slice(None)
    scale, rotation, translation = umeyama_alignment(pred_pts[idx], gt_pts[idx])
    pred_pts = scale * (pred_pts @ rotation.T) + translation
    alignment = {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist()}
    if len(pred_pts) > MAX_POINTS:
        idx = rng.choice(len(pred_pts), MAX_POINTS, replace=False)
        pred_pts, gt_pts = pred_pts[idx], gt_pts[idx]
        rgb = None if rgb is None else rgb[idx]
    pred_rgb = gt_rgb = rgb
    pred_cloud, gt_cloud = point_cloud(pred_pts, pred_rgb), point_cloud(gt_pts, gt_rgb)
    registration = o3d.pipelines.registration.registration_icp(
        pred_cloud, gt_cloud, ICP_THRESHOLD, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred_cloud.transform(registration.transformation)
    metrics = point_metrics(pred_cloud, gt_cloud)
    metrics["alignment"] = {**alignment, "icp": registration.transformation.tolist(),
                            "icp_fitness": float(registration.fitness)}
    return metrics, pred_cloud, gt_cloud
