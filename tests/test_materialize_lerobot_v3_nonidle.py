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
        self.tasks = None  # populated by _FakeLeRobotDataset after construction

    def get_video_file_path(self, ep_index, vid_key):
        # Synthetic per-episode path; the fake decoder keys frames off this string.
        return Path(f"videos/ep{ep_index}_{vid_key}.mp4")


class _FakeOutputMeta:
    def __init__(self):
        self.chunk_settings = None

    def update_chunk_settings(self, **kwargs):
        self.chunk_settings = kwargs


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
        self.meta = _FakeOutputMeta()
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
        self.root = Path(root).resolve()
        self.download_videos = download_videos
        self.video_backend = video_backend
        cfg = self.registry[str(self.root)]
        self.rows = list(cfg["rows"])
        self.hf_dataset = _FakeHfDataset(self.rows)
        self.meta = _FakeMeta(cfg["episodes"], cfg["features"])
        self.tolerance_s = 1e-4
        self.meta.tasks = _FakeTasks(["task0"])

        # Register per-episode frame arrays so the batched decode path resolves
        # frames by the synthesized video path (root / meta.get_video_file_path).
        for ep in self.meta.episodes:
            ep_index = ep["episode_index"]
            for vid_key in self.meta.video_keys:
                rel = self.meta.get_video_file_path(ep_index, vid_key)
                video_path = str(self.root / rel)
                frames = [
                    self.rows[i][vid_key]
                    for i in range(ep["dataset_from_index"], ep["dataset_to_index"])
                ]
                _FakeVideoUtils._registry[video_path] = {"arrays": frames, "fps": self.meta.fps}

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
    ):
        return _FakeOutputDataset(root, repo_id, features, fps, robot_type, use_videos, video_backend)

    def __getitem__(self, idx):
        row = dict(self.rows[idx])
        row["task"] = "task0"
        return row

    def get_raw_item(self, idx):
        return dict(self.rows[idx])


class _FakeHfDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Mirror HuggingFace datasets: a slice returns a column-major dict.
            return {key: [row[key] for row in self.rows[idx]] for key in self.rows[0]}
        return dict(self.rows[idx])


class _FakeVideoUtils:
    """Stand-in for ``lerobot.datasets.video_utils``.

    ``decode_video_frames`` is keyed by the synthesized video path registered
    per episode, so the batched decode path in the materializer is exercised
    end-to-end instead of falling back to per-frame reads.
    """

    _registry: dict = {}

    @staticmethod
    def decode_video_frames(video_path, timestamps, tolerance_s, backend=None):
        entry = _FakeVideoUtils._registry[str(video_path)]
        arrays = entry["arrays"]
        fps = entry["fps"]
        out = [arrays[int(round(float(ts) * fps))] for ts in timestamps]
        return np.stack(out, axis=0)


class _FakeTasksIloc:
    def __init__(self, names):
        self._names = names

    def __getitem__(self, idx):
        return types.SimpleNamespace(name=self._names[idx])


class _FakeTasks:
    """Mimics the pandas DataFrame slice used by ``_task_string_at``."""

    def __init__(self, task_names):
        self.iloc = _FakeTasksIloc(list(task_names))


def _fake_lerobot_modules():
    _FakeLeRobotDataset.registry = {}
    _FakeVideoUtils._registry = {}
    _FakeOutputDataset.instances = []
    lerobot_mod = types.ModuleType("lerobot")
    datasets_mod = types.ModuleType("lerobot.datasets")
    dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_mod.LeRobotDataset = _FakeLeRobotDataset
    video_utils_mod = types.ModuleType("lerobot.datasets.video_utils")
    video_utils_mod.decode_video_frames = _FakeVideoUtils.decode_video_frames
    return {
        "lerobot": lerobot_mod,
        "lerobot.datasets": datasets_mod,
        "lerobot.datasets.lerobot_dataset": dataset_mod,
        "lerobot.datasets.video_utils": video_utils_mod,
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
        {
            "episode_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": 6,
            "videos/observation.images.cam/from_timestamp": 0.0,
        },
        {
            "episode_index": 1,
            "dataset_from_index": 6,
            "dataset_to_index": 10,
            "videos/observation.images.cam/from_timestamp": 0.0,
        },
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
        self.assertEqual(
            output.meta.chunk_settings,
            {
                "chunks_size": 1000,
                "data_files_size_in_mb": 100,
                "video_files_size_in_mb": 100,
            },
        )
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

    def test_action_state_filtering_uses_hf_dataset_when_get_raw_item_is_absent(self):
        input_root = self.tmp_path / "input"
        _register_fake_dataset(input_root)
        dataset = _FakeLeRobotDataset(repo_id="input", root=input_root)

        with patch.object(_FakeLeRobotDataset, "get_raw_item", None):
            action, state = materialize._load_action_state_for_episode(dataset, 0, 3)

        np.testing.assert_array_equal(action[:, 0], np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(state[:, 0], np.array([0.0, 0.0, 0.0]))

    def test_batch_decode_path_is_used_with_one_call_per_chunk(self):
        """The materializer must batch-decode (one decode_video_frames call per
        camera per chunk) rather than reading frames one at a time."""
        input_root = self.tmp_path / "input"
        output_root = self.tmp_path / "output"
        _register_fake_dataset(input_root)

        fake_vu = sys.modules["lerobot.datasets.video_utils"]
        real_decode = fake_vu.decode_video_frames
        batch_sizes: list[int] = []

        def spy(video_path, timestamps, tolerance_s, backend=None):
            batch_sizes.append(len(timestamps))
            return real_decode(video_path, timestamps, tolerance_s, backend)

        thresholds = materialize.IdleThresholds(1e-3, 1e-3, 1e-3, 2, 1)
        with patch.object(fake_vu, "decode_video_frames", spy):
            materialize.materialize_nonidle_dataset(
                input_root,
                output_root,
                thresholds=thresholds,
                parallel_encoding=False,
                decode_chunk_size=256,
                show_progress=False,
            )

        # Episode 0 keep_ranges [[0,1],[3,6]] -> chunk sizes 1, 3.
        # Episode 1 keep_ranges [[0,1],[3,4]] -> chunk sizes 1, 1.
        self.assertEqual(sorted(batch_sizes), [1, 1, 1, 3])
        self.assertEqual(len(batch_sizes), 4)  # one batched call per chunk, single camera

        output = _FakeOutputDataset.instances[0]
        self.assertEqual([len(ep) for ep in output.episodes], [4, 2])
        self.assertEqual(
            [int(row["observation.images.cam"][0, 0, 0]) for row in output.episodes[0]],
            [0, 3, 4, 5],
        )

    def test_falls_back_to_per_frame_when_batch_decode_fails(self):
        """A failing batch decode must fall back to per-frame reads and still
        produce a correct, fully-aligned output dataset."""
        input_root = self.tmp_path / "input"
        output_root = self.tmp_path / "output"
        _register_fake_dataset(input_root)

        fake_vu = sys.modules["lerobot.datasets.video_utils"]

        def boom(*args, **kwargs):
            raise RuntimeError("simulated batch decode failure")

        thresholds = materialize.IdleThresholds(1e-3, 1e-3, 1e-3, 2, 1)
        with patch.object(fake_vu, "decode_video_frames", boom):
            report = materialize.materialize_nonidle_dataset(
                input_root,
                output_root,
                thresholds=thresholds,
                parallel_encoding=False,
                show_progress=False,
            )

        self.assertEqual(report["summary"]["kept_frames"], 6)
        output = _FakeOutputDataset.instances[0]
        self.assertEqual([len(ep) for ep in output.episodes], [4, 2])
        self.assertEqual(
            [int(row["observation.images.cam"][0, 0, 0]) for row in output.episodes[0]],
            [0, 3, 4, 5],
        )
        self.assertEqual([float(row["action"][0]) for row in output.episodes[1]], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
