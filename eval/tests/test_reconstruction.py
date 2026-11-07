"""CPU checks for dataset geometry, selection, masking and reconstruction metrics."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from eval.recon_data import (
    SCANNET50_SCENE_LIST, Frame, Sequence, discover_sequences, fixed_intrinsics, ground_truth_maps,
    load_sequence, read_depth, read_nrgbd_poses, select_frames,
)
from eval.recon_eval import main
from eval.recon_metrics import evaluate_rgbd, umeyama_alignment


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, path, depth=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((480, 640), 2000, np.uint16) if depth else np.zeros((480, 640, 3), np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))

    def test_uniform_selection_includes_endpoints_without_duplicates(self):
        self.assertEqual(select_frames(list(range(10)), 4), [0, 3, 6, 9])
        self.assertEqual(select_frames(list(range(3)), 10), [0, 1, 2])
        self.assertEqual(select_frames([], 3), [])
        with self.assertRaises(ValueError):
            select_frames([1], 0)

    def test_7scenes_requires_projected_depth_and_uses_test_split(self):
        scene = self.root / "chess"
        sequence = scene / "seq-03"
        sequence.mkdir(parents=True)
        (scene / "TestSplit.txt").write_text("sequence3\n")
        for idx in range(3):
            prefix = sequence / f"frame-{idx:06d}"
            self.image(Path(f"{prefix}.color.png"))
            self.image(Path(f"{prefix}.depth.png"), depth=True)
            np.savetxt(f"{prefix}.pose.txt", np.eye(4))
        self.image(sequence / "frame-000002.depth.proj.png", depth=True)
        self.assertEqual(discover_sequences(self.root, "7scenes", ["chess"]), [("chess/seq-03", sequence)])
        loaded = load_sequence("chess/seq-03", sequence, "7scenes")
        self.assertEqual([f.frame_id for f in loaded.frames], [2])
        self.assertEqual(loaded.invalid_frames, 2)

    def test_nrgbd_coordinate_conversion_preserves_invalid_frame_indices(self):
        path = self.root / "poses.txt"
        poses = np.tile(np.eye(4), (3, 1, 1))
        poses[1, 0, 0] = np.nan
        poses[2, :3, 3] = [1, 2, 3]
        np.savetxt(path, poses.reshape(-1, 4))
        loaded = read_nrgbd_poses(path)
        self.assertIsNone(loaded[1])
        np.testing.assert_array_equal(loaded[0], np.diag([1, -1, -1, 1]))
        np.testing.assert_array_equal(loaded[2][:3, 3], [1, 2, 3])
        path.write_text("1 0 0 0\n")
        with self.assertRaises(ValueError):
            read_nrgbd_poses(path)

    def test_scannet_manifest_and_numeric_frame_order(self):
        sequence = self.root / "val" / "scene0001_00"
        (sequence / "pose").mkdir(parents=True)
        (sequence / "intrinsic").mkdir()
        np.savetxt(sequence / "intrinsic/intrinsic_depth.txt", np.eye(4))
        for idx in [10, 2, 0]:
            self.image(sequence / "color" / f"{idx}.jpg")
            self.image(sequence / "depth" / f"{idx}.png", depth=True)
            np.savetxt(sequence / "pose" / f"{idx}.txt", np.eye(4))
        (self.root / "test_seq.txt").write_text("val/scene0001_00\n")
        self.assertEqual(discover_sequences(self.root, "scannet", scene_list=self.root / "test_seq.txt"),
                         [("val/scene0001_00", sequence)])
        loaded = load_sequence("val/scene0001_00", sequence, "scannet")
        self.assertEqual([f.frame_id for f in loaded.frames], [0, 2, 10])

    def test_scannet_default_split_ignores_unrelated_dataset_manifest(self):
        names = SCANNET50_SCENE_LIST.read_text().splitlines()
        for name in names:
            (self.root / name).mkdir(parents=True)
        (self.root / "test_seq.txt").write_text("train/scene9999_00\n")
        entries = discover_sequences(self.root, "scannet")
        self.assertEqual(entries, [(name, self.root / name) for name in names])
        self.assertEqual(discover_sequences(self.root, "scannet", scenes=["scene0000_00"]),
                         [("train/scene0000_00", self.root / "train/scene0000_00")])

    def test_scannet_default_split_resolves_flat_export_and_keeps_missing_scene(self):
        names = SCANNET50_SCENE_LIST.read_text().splitlines()
        for name in names[:-1]:
            (self.root / Path(name).name).mkdir()
        entries = discover_sequences(self.root, "scannet")
        self.assertEqual(entries[:-1], [(name, self.root / Path(name).name) for name in names[:-1]])
        # Missing scenes must be reported during loading, not silently excluded from the mean.
        self.assertEqual(entries[-1], (names[-1], self.root / names[-1]))
        self.assertEqual(discover_sequences(self.root, "scannet", scenes=[names[0]])[0][1],
                         self.root / Path(names[0]).name)

    def test_depth_units_invalid_values_and_relative_pose(self):
        depth_path = self.root / "depth.png"
        raw = np.full((480, 640), 2000, np.uint16)
        raw[0, :4] = [0, 9999, 10000, 65535]
        cv2.imwrite(str(depth_path), raw)
        first, second = np.eye(4), np.eye(4)
        first[:3, 3] = [10, 20, 30]
        second[:3, 3] = [11, 20, 30]
        frames = [Frame(i, self.root / "unused.png", depth_path, pose) for i, pose in enumerate([first, second])]
        depth = read_depth(frames[0])
        np.testing.assert_allclose(depth[0, :4], [0, 9.999, 0, 0])
        sequence = Sequence("chess/seq-01", "chess", self.root, frames, fixed_intrinsics("7scenes"), first)
        xyz, valid = ground_truth_maps(sequence, (480, 640))
        np.testing.assert_allclose(xyz[0, 240, 320], [0, 0, 2])
        np.testing.assert_allclose(xyz[1, 240, 320], [1, 0, 2])
        self.assertFalse(valid[0, 0, 0])

    def test_failed_dry_run_records_failure_and_returns_nonzero(self):
        (self.root / "chess").mkdir()
        (self.root / "chess/TestSplit.txt").write_text("sequence3\n")
        output = self.root / "output"
        result = main("7scenes", ["--dataset_dir", str(self.root), "--scenes", "chess",
                                   "--output_dir", str(output), "--dry_run"])
        self.assertEqual(result, 1)
        self.assertEqual(len(list(output.glob("*_manifest.json"))), 1)


class MetricTests(unittest.TestCase):
    def test_sim3_recovers_known_transform(self):
        rng = np.random.default_rng(0)
        source = rng.normal(size=(300, 3))
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        target = 2.5 * source @ rotation.T + [3, 4, 5]
        scale, estimated_rotation, translation = umeyama_alignment(source, target)
        self.assertAlmostEqual(scale, 2.5)
        np.testing.assert_allclose(estimated_rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(translation, [3, 4, 5], atol=1e-12)
        with self.assertRaises(ValueError):
            umeyama_alignment(np.zeros((10, 3)), np.zeros((10, 3)))

    def test_crop_confidence_filter_and_alignment(self):
        y, x = np.indices((392, 518), dtype=np.float32)
        truth = np.stack((x / 200, y / 200, 2 + np.sin(x / 60) / 10 + np.cos(y / 90) / 10), axis=-1)[None]
        prediction = (truth - np.array([0.1, 0.2, 0.3], np.float32)) / 2
        # Large border errors must be excluded by the established central crop.
        prediction[:, :84] = 100
        prediction[:, 308:] = 100
        prediction[:, :, :147] = 100
        prediction[:, :, 371:] = 100
        confidence = np.full(truth.shape[:-1], 2.0, np.float32)
        prediction[0, 100, 200] = 100
        confidence[0, 100, 200] = 0
        valid = np.ones(truth.shape[:-1], bool)
        result, _, _ = evaluate_rgbd(prediction, truth, valid, confidence,
                                    np.random.default_rng(42), conf_thresh=1)
        self.assertEqual(result["n_pts"], 224 * 224 - 1)
        self.assertLess(result["cd"], 1e-5)
        self.assertAlmostEqual(result["nc"], 1.0, places=4)
        self.assertAlmostEqual(result["cd_sum"], 2 * result["cd"])
        with self.assertRaisesRegex(ValueError, "Too few"):
            evaluate_rgbd(prediction, truth, np.zeros_like(valid), confidence, np.random.default_rng(42))


if __name__ == "__main__":
    unittest.main()
