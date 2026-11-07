"""Shared runner for ScanNet-50, 7Scenes and NRGBD reconstruction."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time


def build_parser(dataset):
    parser = argparse.ArgumentParser(description=f"DA-VGGT {dataset} 3D reconstruction evaluation")
    parser.add_argument("--dataset_dir", required=True, help="Dataset root; see README.md")
    parser.add_argument("--n_frames", type=int, default=500, help="Uniformly sample up to this many frames per sequence")
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--sampling_method", choices=("da_partitioning", "random_partitioning", "origin"), default="da_partitioning")
    parser.add_argument("--local_search_iters", type=int, default=5)
    parser.add_argument("--conf_thresh", type=float, default=0.0, help="Keep confidence > threshold; 0 disables filtering")
    parser.add_argument("--scenes", nargs="+", help="Scene names; 7Scenes also accepts chess/seq-03")
    parser.add_argument("--model_path", help="VGGT checkpoint (.pt/.pth/.safetensors) or local Hugging Face directory")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dino_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--tau", type=float, help="Pose-distance decay; omit to use the median pairwise distance")
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--rechunk_remaining_only", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse chunk 0 and repartition only the remaining frames (paper setting)")
    parser.add_argument("--output_dir", default=f"./results/3d_{dataset}")
    parser.add_argument("--export_ply", action="store_true", help="Also save aligned predicted and GT point clouds")
    parser.add_argument("--ply_dir", help="PLY destination (requires --export_ply; default: output_dir/ply)")
    parser.add_argument("--dry_run", action="store_true", help="Validate data and write selected-frame manifest without loading a model")
    if dataset == "scannet":
        parser.add_argument("--num_scenes", type=int, default=50)
        parser.add_argument("--scene_list", help="Scene manifest; default: bundled eval/scannet50_scenes.txt")
    return parser


def protocol_metadata():
    return {
        "name": "local_eval_chunked_3d",
        "reference": "DA-VGGT released 7Scenes evaluator; local VGGT/FastVGGT/sparse-vggt eval_chunked_3d_*",
        "frame_selection": "linspace endpoints, up to n_frames",
        "rgb_preprocessing": "VGGT load_and_preprocess_images(mode=crop), width 518, height rounded to multiple of 14",
        "depth_unit": "millimeters converted to meters",
        "valid_depth_m": "0 < depth < 10",
        "gt_coordinates": "first selected camera; NRGBD OpenGL converted to OpenCV",
        "evaluation_crop": [224, 224],
        "sim3_correspondences": 30000,
        "max_points": 999999,
        "point_sampling": "same random indices for prediction and GT",
        "icp_threshold_m": 0.1,
        "icp_estimator": "Open3D point-to-point, rigid, default convergence criteria",
        "normals": "Open3D estimate_normals defaults; absolute nearest-neighbor dot products",
        "distance_unit": "meters",
        "cd": "(acc + comp) / 2; cd_sum is also recorded",
        "nc": "(nc1 + nc2) / 2",
        "aggregation": "unweighted mean over sequences (7Scenes), scenes (ScanNet/NRGBD)",
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def mean_metrics(results):
    import numpy as np
    keys = ("acc", "acc_med", "comp", "comp_med", "cd", "cd_mean", "cd_sum",
            "nc1", "nc1_med", "nc2", "nc2_med", "nc", "inference_time_s", "peak_vram_gib")
    return {key: float(np.mean([r[key] for r in results])) if results else None for key in keys}


def infer(model, images, args):
    import torch
    from eval.recon_inference import run_inference, run_inference_poseweight_3d
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if args.sampling_method == "da_partitioning" and len(images) > args.chunk_size:
        (extrinsic, intrinsic, depth, confidence, shape, count, timing,
         initial, chunks, _, tau_used) = run_inference_poseweight_3d(
            model, images, device, dtype, gamma=args.gamma, tau=args.tau,
            ls_iters=args.local_search_iters, chunk_size=args.chunk_size,
            rechunk_remaining_only=args.rechunk_remaining_only,
            epsilon=args.epsilon, align_mode="se3", n_anchors=1,
            anchor_select="uniform", seed=args.seed,
        )
        partition = {"initial": initial, "final": chunks, "tau_used": float(tau_used)}
    else:
        pose, depth, confidence, shape, count, timing = run_inference(model, images, device, dtype)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose.unsqueeze(0).float().to(device), shape)
        extrinsic, intrinsic = extrinsic[0], intrinsic[0]
        partition = None
    torch.cuda.synchronize(device)
    timing["wall_time_s"] = time.perf_counter() - started
    timing["peak_vram_total_mb"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    points = unproject_depth_map_to_point_map(depth[0].float().cpu(), extrinsic.float().cpu(), intrinsic.float().cpu())
    conf = confidence[0].float().cpu().numpy()
    return points, conf, tuple(shape), int(count), timing, partition


def main(dataset, argv=None):
    parser = build_parser(dataset)
    args = parser.parse_args(argv)
    if args.n_frames <= 0 or args.dino_batch_size <= 0:
        parser.error("--n_frames and --dino_batch_size must be positive")
    if args.sampling_method != "origin" and args.chunk_size < 2:
        parser.error("--chunk_size must be >= 2 (one shared anchor and at least one other frame)")
    if args.local_search_iters < 0 or args.gamma <= 0 or (args.tau is not None and args.tau <= 0):
        parser.error("local_search_iters must be nonnegative; gamma and tau must be positive")
    if args.conf_thresh < 0 or args.epsilon < 0:
        parser.error("conf_thresh and epsilon must be nonnegative")
    if args.ply_dir and not args.export_ply:
        parser.error("--ply_dir requires --export_ply")
    if dataset == "scannet" and args.num_scenes <= 0:
        parser.error("--num_scenes must be positive")

    import numpy as np
    from eval.recon_data import discover_sequences, load_sequence, select_frames

    output = Path(args.output_dir)
    name = f"chunked_3d_{dataset}_{args.sampling_method}_n{args.n_frames}_c{args.chunk_size}"
    result_path = output / f"{name}.json"
    manifest_path = output / f"{name}_manifest.json"
    data = {"dataset": dataset, "args": vars(args), "protocol": protocol_metadata(),
            "timestamp": datetime.now(timezone.utc).isoformat(), "sequences": {},
            "scenes": {}, "failures": {}, "global_summary": mean_metrics([])}
    try:
        entries = discover_sequences(args.dataset_dir, dataset, args.scenes,
                                     getattr(args, "num_scenes", 50), getattr(args, "scene_list", None))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sequences, manifest = [], {}
    for sequence_name, path in entries:
        try:
            sequence = load_sequence(sequence_name, path, dataset)
            available = len(sequence.frames)
            sequence.frames = select_frames(sequence.frames, args.n_frames)
            manifest[sequence_name] = {
                "available_frames": available, "invalid_frames": sequence.invalid_frames,
                "n_frames": len(sequence.frames), "frame_ids": [f.frame_id for f in sequence.frames],
                "color_paths": [str(f.color) for f in sequence.frames],
                "depth_paths": [str(f.depth) for f in sequence.frames],
            }
            sequences.append(sequence)
            print(f"{sequence_name}: {len(sequence.frames)}/{available} frames, {sequence.invalid_frames} invalid/missing", flush=True)
            if len(sequence.frames) < args.n_frames:
                print(f"  WARNING: fewer than the requested {args.n_frames} frames", flush=True)
        except (OSError, ValueError) as error:
            data["failures"][sequence_name] = str(error)
            print(f"{sequence_name}: ERROR: {error}", flush=True)
    write_json(manifest_path, {"dataset": dataset, "args": vars(args), "protocol": data["protocol"],
                               "sequences": manifest, "failures": data["failures"]})
    print(f"Frame manifest: {manifest_path}", flush=True)
    if args.dry_run:
        return 1 if data["failures"] else 0
    if not sequences:
        write_json(result_path, data)
        return 1

    import torch
    from eval.recon_inference import load_model
    from eval.recon_data import ground_truth_maps
    from eval.recon_metrics import evaluate_rgbd
    from vggt.utils.load_fn import load_and_preprocess_images

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("Model inference requires a CUDA device; --dry_run only checks the dataset")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = load_model(device, args.model_path, args.chunk_size)
    model.aggregator.dino_batch_size = args.dino_batch_size
    model.aggregator.sampling_method = "random_balanced" if args.sampling_method == "random_partitioning" else args.sampling_method
    model.aggregator.sampling_local_search_iters = args.local_search_iters
    model.aggregator.sampling_align_mode = "se3"
    if args.sampling_method == "origin":
        model.aggregator.sampling_max_frames = 0
    data["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(device)}
    ply_dir = Path(args.ply_dir) if args.ply_dir else output / "ply"
    for sequence in sequences:
        print(f"Evaluating {sequence.name} ...", flush=True)
        try:
            images = load_and_preprocess_images([str(f.color) for f in sequence.frames])
            points, confidence, shape, count, timing, partition = infer(model, images, args)
            truth, valid = ground_truth_maps(sequence, shape)
            colors = images.permute(0, 2, 3, 1).numpy() if args.export_ply else None
            metrics, pred_cloud, gt_cloud = evaluate_rgbd(points, truth, valid, confidence, rng,
                                                         args.conf_thresh, colors)
            if args.export_ply:
                import open3d as o3d
                ply_dir.mkdir(parents=True, exist_ok=True)
                stem = sequence.name.replace("/", "_")
                for suffix, cloud in (("pred", pred_cloud), ("gt", gt_cloud)):
                    destination = ply_dir / f"{stem}_{suffix}.ply"
                    if not o3d.io.write_point_cloud(str(destination), cloud):
                        raise OSError(f"Failed to write {destination}")
            result = {**metrics, "n_frames": len(sequence.frames), "num_chunks": count,
                      "frame_ids": manifest[sequence.name]["frame_ids"],
                      "inference_time_s": float(timing["total"]),
                      "peak_vram_gib": timing["peak_vram_total_mb"] / 1024,
                      "timing": {k: v for k, v in timing.items() if k != "transformer_per_chunk"},
                      "partition": partition}
            data["sequences"][sequence.name] = result
            scene = data["scenes"].setdefault(sequence.scene, {"sequences": {}})
            scene["sequences"][sequence.name.split("/")[-1]] = result
            scene["mean"] = mean_metrics(list(scene["sequences"].values()))
            data["global_summary"] = mean_metrics(list(data["sequences"].values()))
            print(f"  acc={result['acc']:.5f} comp={result['comp']:.5f} CD={result['cd']:.5f} "
                  f"NC={result['nc']:.5f} time={result['inference_time_s']:.2f}s", flush=True)
            del images, points, confidence, truth, valid, colors, pred_cloud, gt_cloud
        except Exception as error:
            import traceback
            traceback.print_exc()
            data["failures"][sequence.name] = f"{type(error).__name__}: {error}"
        finally:
            torch.cuda.empty_cache()
        data["completed_sequences"] = len(data["sequences"])
        data["requested_sequences"] = len(entries)
        write_json(result_path, data)
    print(f"Results: {result_path} ({len(data['sequences'])} completed, {len(data['failures'])} failed)", flush=True)
    return 1 if data["failures"] else 0
