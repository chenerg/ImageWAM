import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "create_lerobot_v3_nonidle_parquets.py"
_SPEC = importlib.util.spec_from_file_location("create_lerobot_v3_nonidle_parquets", _SCRIPT_PATH)
create_nonidle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = create_nonidle
_SPEC.loader.exec_module(create_nonidle)


def _write_fake_lerobot_v3(root: Path, *, include_media: bool = True) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 10,
                "total_episodes": 2,
                "total_frames": 10,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text('{"action": {"mean": [0.5]}}', encoding="utf-8")
    (root / "meta" / "tasks.parquet").write_bytes(b"copied task metadata")
    (root / "README.txt").write_text("extra dataset file", encoding="utf-8")

    episode_meta_dir = root / "meta" / "episodes" / "chunk-000"
    episode_meta_dir.mkdir(parents=True)
    episode_table = pa.table(
        {
            "episode_index": [0, 1],
            "length": [6, 4],
            "dataset_from_index": [0, 6],
            "dataset_to_index": [6, 10],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 1],
        }
    )
    pq.write_table(episode_table, episode_meta_dir / "file-000.parquet")

    deltas = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "index": list(range(6)),
                "episode_index": [0] * 6,
                "frame_index": list(range(6)),
                "action": [[delta] for delta in deltas[:6]],
                "observation.state": [[0.0] for _ in deltas[:6]],
            }
        ),
        data_dir / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "index": list(range(6, 10)),
                "episode_index": [1] * 4,
                "frame_index": list(range(4)),
                "action": [[delta] for delta in deltas[6:]],
                "observation.state": [[0.0] for _ in deltas[6:]],
            }
        ),
        data_dir / "file-001.parquet",
    )

    if include_media:
        (root / "videos").mkdir()
        (root / "videos" / "video.txt").write_text("source video", encoding="utf-8")
        (root / "images").mkdir()
        (root / "images" / "image.txt").write_text("source image", encoding="utf-8")


class CreateLeRobotV3NonidleParquetsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_matching_data_and_meta_parquet_subsets(self):
        import pyarrow.parquet as pq

        dataset_root = self.tmp_path / "dataset"
        _write_fake_lerobot_v3(dataset_root)
        source_data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
        source_data_before = source_data_path.read_bytes()

        output_root = self.tmp_path / "dataset_nonidle"
        report = create_nonidle.create_nonidle_parquets(
            dataset_root,
            output_root,
            thresholds=create_nonidle.IdleThresholds(
                idle_l2_threshold=1e-3,
                idle_arm_l2_threshold=1e-3,
                idle_gripper_l2_threshold=1e-3,
                min_idle_len=2,
                min_non_idle_len=1,
            ),
            progress_every=0,
        )

        file0 = pq.read_table(output_root / "data" / "chunk-000" / "file-000.parquet").to_pylist()
        file1 = pq.read_table(output_root / "data" / "chunk-000" / "file-001.parquet").to_pylist()
        self.assertEqual([row["index"] for row in file0], [0, 1, 2, 3])
        self.assertEqual([row["frame_index"] for row in file0], [0, 3, 4, 5])
        self.assertEqual([row["index"] for row in file1], [4, 5])
        self.assertEqual([row["frame_index"] for row in file1], [0, 3])

        meta = pq.read_table(
            output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ).to_pylist()
        self.assertEqual([row["length"] for row in meta], [4, 2])
        self.assertEqual([row["dataset_from_index"] for row in meta], [0, 4])
        self.assertEqual([row["dataset_to_index"] for row in meta], [4, 6])

        self.assertEqual(report["summary"]["total_frames"], 10)
        self.assertEqual(report["summary"]["kept_frames"], 6)
        self.assertEqual(report["format"], "imagewam_lerobot_v3_nonidle_dataset_v2")
        self.assertEqual(report["source_dataset_dir"], str(dataset_root.resolve()))
        self.assertEqual(report["output_dataset_dir"], str(output_root.resolve()))
        self.assertTrue((output_root / "nonidle_parquet_report.json").exists())

        self.assertTrue((output_root / "videos").is_symlink())
        self.assertTrue((output_root / "images").is_symlink())
        self.assertEqual((output_root / "videos").readlink(), dataset_root.resolve() / "videos")
        self.assertEqual((output_root / "images").readlink(), dataset_root.resolve() / "images")
        self.assertEqual(
            (output_root / "meta" / "stats.json").read_bytes(),
            (dataset_root / "meta" / "stats.json").read_bytes(),
        )
        self.assertEqual((output_root / "meta" / "tasks.parquet").read_bytes(), b"copied task metadata")
        self.assertEqual((output_root / "README.txt").read_text(encoding="utf-8"), "extra dataset file")
        self.assertEqual(json.loads((output_root / "meta" / "info.json").read_text())["total_frames"], 6)

        # The source dataset is not modified.
        self.assertEqual(json.loads((dataset_root / "meta" / "info.json").read_text())["total_frames"], 10)
        self.assertEqual(source_data_path.read_bytes(), source_data_before)
        self.assertFalse((dataset_root / "nonidle_parquet_report.json").exists())

    def test_existing_output_is_rejected(self):
        dataset_root = self.tmp_path / "dataset"
        _write_fake_lerobot_v3(dataset_root)
        output_root = self.tmp_path / "dataset_nonidle"
        output_root.mkdir()

        thresholds = create_nonidle.IdleThresholds(1e-3, 1e-3, 1e-3, 2, 1)
        with self.assertRaises(FileExistsError):
            create_nonidle.create_nonidle_parquets(
                dataset_root,
                output_root,
                thresholds=thresholds,
                progress_every=0,
            )

    def test_identical_or_nested_paths_are_rejected(self):
        dataset_root = self.tmp_path / "dataset"
        _write_fake_lerobot_v3(dataset_root)
        thresholds = create_nonidle.IdleThresholds(1e-3, 1e-3, 1e-3, 2, 1)

        invalid_outputs = [
            dataset_root,
            dataset_root / "derived",
            self.tmp_path,
        ]
        for output_root in invalid_outputs:
            with self.subTest(output_root=output_root), self.assertRaises(ValueError):
                create_nonidle.create_nonidle_parquets(
                    dataset_root,
                    output_root,
                    thresholds=thresholds,
                    progress_every=0,
                )

    def test_missing_media_directories_do_not_create_broken_links(self):
        dataset_root = self.tmp_path / "dataset"
        _write_fake_lerobot_v3(dataset_root, include_media=False)
        output_root = self.tmp_path / "dataset_nonidle"

        create_nonidle.create_nonidle_parquets(
            dataset_root,
            output_root,
            thresholds=create_nonidle.IdleThresholds(1e-3, 1e-3, 1e-3, 2, 1),
            progress_every=0,
        )

        self.assertFalse((output_root / "videos").is_symlink())
        self.assertFalse((output_root / "images").is_symlink())
        self.assertTrue((output_root / "data" / "chunk-000" / "file-000.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
