import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "materialize_lerobot_v3_nonidle.py"
_SPEC = importlib.util.spec_from_file_location("materialize_lerobot_v3_nonidle", _SCRIPT_PATH)
materialize = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = materialize
_SPEC.loader.exec_module(materialize)


class _FakeMeta:
    def __init__(self, episodes, features):
        self.episodes = episodes
        self.features = features
        self.fps = 10
        self.robot_type = "fake_robot"
        self.video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
        self.image_keys = [key for key, ft in features.items() if ft["dtype"] == "image"]
        self.camera_keys = self.video_keys + self.image_keys
        self.chunks_size = 1000
        self.data_files_size_in_mb = 100
        self.video_files_size_in_mb = 100


class _FakeOutputDataset:
    instances = []

    def __init__(self, root, repo_id, features, fps, robot_type, use_videos, video_backend):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.repo_id = repo_id
        self.features = features
        self.fps = fps
        self.robot_type = robot_type
        self.use_videos = use_videos
        self.video_backend = video_backend
        self.current = []
        self.episodes = []
        self.finalized = False
        self.instances.append(self)

    def add_frame(self, frame):
        self.current.append(frame)

    def save_episode(self, parallel_encoding=True):
        episode_index = len(self.episodes)
        rows = []
        for frame_index, frame in enumerate(self.current):
            row = dict(frame)
            row["episode_index"] = episode_index
            row["frame_index"] = frame_index
            row["timestamp"] = frame_index / self.fps
            row["index"] = sum(len(ep) for ep in self.episodes) + frame_index
            rows.append(row)
        self.episodes.append(rows)
        self.current = []

    def finalize(self):
        self.finalized = True


class _FakeLeRobotDataset:
    registry = {}

    def __init__(
        self,
        repo_id,
        root,
        download_videos=True,
        video_backend=None,
        **_,
    ):
        self.repo_id = repo_id
        self.root = Path(root)
        self.download_videos = download_videos
        self.video_backend = video_backend
        cfg = self.registry[str(self.root)]
        self.rows = list(cfg["rows"])
        self.meta = _FakeMeta(cfg["episodes"], cfg["features"])

    @classmethod
    def create(
        cls,
        repo_id,
        fps,
        features,
        root,
        robot_type=None,
        use_videos=True,
        video_backend=None,
        **_,
    ):
        return _FakeOutputDataset(root, repo_id, features, fps, robot_type, use_videos, video_backend)

    def __getitem__(self, idx):
        row = dict(self.rows[idx])
        row["task"] = "task0"
        return row

    def get_raw_item(self, idx):
        return dict(self.rows[idx])


def _fake_lerobot_modules():
    _FakeLeRobotDataset.registry = {}
    _FakeOutputDataset.instances = []
    lerobot_mod = types.ModuleType("lerobot")
    datasets_mod = types.ModuleType("lerobot.datasets")
    dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_mod.LeRobotDataset = _FakeLeRobotDataset
    return {
        "lerobot": lerobot_mod,
        "lerobot.datasets": datasets_mod,
        "lerobot.datasets.lerobot_dataset": dataset_mod,
    }


def _register_fake_dataset(root):
    features = {
        "action": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.images.cam": {"dtype": "video", "shape": (3, 1, 1), "names": None},
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    # Episode 0 idle mask: [False, True, True, False, False, True].
    # The terminal idle frame is intentionally kept.
    deltas = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    rows = []
    for idx, delta in enumerate(deltas):
        episode_index = 0 if idx < 6 else 1
        episode_frame = idx if episode_index == 0 else idx - 6
        rows.append(
            {
                "action": np.array([delta], dtype=np.float32),
                "observation.state": np.array([0.0], dtype=np.float32),
                "observation.images.cam": np.full((3, 1, 1), idx, dtype=np.uint8),
                "episode_index": episode_index,
                "frame_index": episode_frame,
                "timestamp": episode_frame / 10.0,
                "index": idx,
                "task_index": 0,
            }
        )
    episodes = [
        {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 6},
        {"episode_index": 1, "dataset_from_index": 6, "dataset_to_index": 10},
    ]
    _FakeLeRobotDataset.registry[str(Path(root).resolve())] = {
        "features": features,
        "episodes": episodes,
        "rows": rows,
    }


class MaterializeLeRobotV3NonidleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.modules_patcher = patch.dict(sys.modules, _fake_lerobot_modules())
        self.modules_patcher.start()

    def tearDown(self):
        self.modules_patcher.stop()
        self.tmp.cleanup()

    def test_keep_ranges_remove_long_nonterminal_idle_and_keep_short_or_terminal_idle(self):
        idle = np.array([False, True, False, True, True, False, True, True])
        ranges = materialize._compute_keep_ranges(idle, min_idle_len=2, min_non_idle_len=1)
        self.assertEqual(ranges, [[0, 3], [5, 8]])

    def test_materialize_reindexes_output_and_keeps_video_with_data(self):
        input_root = self.tmp_path / "input"
        output_root = self.tmp_path / "output"
        _register_fake_dataset(input_root)

        report = materialize.materialize_nonidle_dataset(
            input_root,
            output_root,
            thresholds=materialize.IdleThresholds(
                idle_l2_threshold=1e-3,
                idle_arm_l2_threshold=1e-3,
                idle_gripper_l2_threshold=1e-3,
                min_idle_len=2,
                min_non_idle_len=1,
            ),
            video_backend="pyav",
            parallel_encoding=False,
            show_progress=False,
        )

        output = _FakeOutputDataset.instances[0]
        self.assertTrue(output.finalized)
        self.assertEqual(len(output.episodes), 2)
        self.assertEqual([len(ep) for ep in output.episodes], [4, 2])
        self.assertEqual([row["frame_index"] for row in output.episodes[0]], [0, 1, 2, 3])
        self.assertEqual([row["episode_index"] for row in output.episodes[1]], [1, 1])
        self.assertEqual([int(row["observation.images.cam"][0, 0, 0]) for row in output.episodes[0]], [0, 3, 4, 5])
        self.assertEqual([float(row["action"][0]) for row in output.episodes[1]], [1.0, 1.0])

        self.assertEqual(report["summary"]["total_frames"], 10)
        self.assertEqual(report["summary"]["kept_frames"], 6)
        self.assertEqual(report["episodes"][0]["keep_ranges"], [[0, 1], [3, 6]])
        report_path = output_root / "nonidle_materialize_report.json"
        self.assertTrue(report_path.exists())

    def test_existing_output_requires_overwrite(self):
        input_root = self.tmp_path / "input"
        output_root = self.tmp_path / "output"
        output_root.mkdir()
        (output_root / "old.txt").write_text("old", encoding="utf-8")
        _register_fake_dataset(input_root)

        thresholds = materialize.IdleThresholds(1e-3, 1e-3, 1e-3, 2, 1)
        with self.assertRaises(FileExistsError):
            materialize.materialize_nonidle_dataset(
                input_root,
                output_root,
                thresholds=thresholds,
                parallel_encoding=False,
                show_progress=False,
            )

        report = materialize.materialize_nonidle_dataset(
            input_root,
            output_root,
            thresholds=thresholds,
            overwrite=True,
            parallel_encoding=False,
            show_progress=False,
        )
        self.assertFalse((output_root / "old.txt").exists())
        self.assertEqual(report["summary"]["kept_frames"], 6)


if __name__ == "__main__":
    unittest.main()
