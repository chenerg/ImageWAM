import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from imagewam.datasets.lerobot.lerobot.lerobot_dataset_v3 import MultiLeRobotDatasetV3
else:
    MultiLeRobotDatasetV3 = None


class _Column:
    def __init__(self, values):
        self._values = list(values)

    def to_pylist(self):
        return list(self._values)

    def __iter__(self):
        return iter(self._values)


class _Episodes:
    def __init__(self, rows):
        self._rows = list(rows)
        self.column_names = list(rows[0]) if rows else []

    def __getitem__(self, key):
        return _Column([row[key] for row in self._rows])


class _Meta:
    def __init__(self, episode_rows, features, camera_keys=None):
        self.episodes = _Episodes(episode_rows)
        self.info = {"fps": 10, "video": bool(camera_keys)}
        self.features = features
        self.camera_keys = list(camera_keys or [])
        self.video_keys = list(camera_keys or [])
        self.depth_keys = []
        self.tasks = {0: "task0"}


class _FakeLeRobotDataset:
    registry = {}
    instances = []

    def __init__(
        self,
        repo_id,
        root,
        episodes=None,
        image_transforms=None,
        delta_timestamps=None,
        **_,
    ):
        self.repo_id = repo_id
        self.root = Path(root)
        self.episodes = episodes
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        cfg = self.registry[str(self.root)]
        self.meta = _Meta(cfg["episode_rows"], cfg["features"], cfg.get("camera_keys", []))
        selected = set(episodes) if episodes is not None else None
        self.rows = [
            row
            for row in cfg["frame_rows"]
            if selected is None or row["episode_index"] in selected
        ]
        self.full_get_calls = []
        self.raw_get_calls = []
        self.instances.append(self)

    @property
    def fps(self):
        return 10

    @property
    def num_episodes(self):
        return len(self.episodes) if self.episodes is not None else len(self.meta.episodes._rows)

    def __len__(self):
        return len(self.rows)

    def _sample(self, idx, include_visual):
        self._validate_idx(idx)
        row = self.rows[idx]
        frame = float(row["episode_frame"])
        item = {
            "index": torch.tensor(row["absolute_index"]),
            "episode_index": torch.tensor(row["episode_index"]),
            "frame_index": torch.tensor(row["episode_frame"]),
            "timestamp": torch.tensor(frame / 10.0),
            "task_index": torch.tensor(0),
            "action": torch.tensor(frame + 100.0),
            "observation.state": torch.tensor(frame),
            "task": "task0",
        }
        if "raw.left_pose" in self.meta.features:
            item["raw.left_pose"] = torch.full((7,), frame)
            item["raw.state_pose"] = torch.full((7,), frame + 10.0)
        if include_visual:
            for key in self.meta.camera_keys:
                item[key] = torch.full((3, 1, 1), frame)
        return item

    def _validate_idx(self, idx):
        if idx < 0 or idx >= len(self.rows):
            raise IndexError(f"local index {idx} out of range for fake dataset")

    def __getitem__(self, idx):
        self.full_get_calls.append(int(idx))
        return self._sample(idx, include_visual=True)

    def get_raw_item(self, idx):
        self.raw_get_calls.append(int(idx))
        return self._sample(idx, include_visual=False)


def _fake_lerobot_modules():
    _FakeLeRobotDataset.registry = {}
    _FakeLeRobotDataset.instances = []
    lerobot_mod = types.ModuleType("lerobot")
    datasets_mod = types.ModuleType("lerobot.datasets")
    dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_mod.LeRobotDataset = _FakeLeRobotDataset
    return {
        "lerobot": lerobot_mod,
        "lerobot.datasets": datasets_mod,
        "lerobot.datasets.lerobot_dataset": dataset_mod,
    }


def _register_root(root, episode_lengths, *, episode_ids=None, abs_starts=None, features=None, camera_keys=None):
    root = Path(root)
    episode_ids = episode_ids or list(range(len(episode_lengths)))
    abs_starts = abs_starts or [sum(episode_lengths[:i]) for i in range(len(episode_lengths))]
    features = features or {
        "action": {"dtype": "float32"},
        "observation.state": {"dtype": "float32"},
        "observation.images.cam": {"dtype": "video"},
    }
    episode_rows = []
    frame_rows = []
    for episode_id, length, abs_start in zip(episode_ids, episode_lengths, abs_starts, strict=True):
        episode_rows.append(
            {
                "episode_index": episode_id,
                "dataset_from_index": abs_start,
                "dataset_to_index": abs_start + length,
            }
        )
        for frame in range(length):
            frame_rows.append(
                {
                    "episode_index": episode_id,
                    "episode_frame": frame,
                    "absolute_index": abs_start + frame,
                }
            )
    _FakeLeRobotDataset.registry[str(root)] = {
        "episode_rows": episode_rows,
        "frame_rows": frame_rows,
        "features": features,
        "camera_keys": list(camera_keys or []),
    }


def _write_filter(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@unittest.skipIf(torch is None, "torch is required for LeRobot dataset tests")
class LeRobotV3StrictNonidleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.modules_patcher = patch.dict(sys.modules, _fake_lerobot_modules())
        self.modules_patcher.start()

    def tearDown(self):
        self.modules_patcher.stop()
        self.tmp.cleanup()

    def test_strict_nonidle_uses_filtered_timeline_for_delta_context(self):
        root = self.tmp_path / "root"
        filter_path = _write_filter(self.tmp_path / "nonidle.json", {"episodes": {"0": [[0, 2], [4, 6]]}})
        _register_root(root, [6], camera_keys=["observation.images.cam"])

        dataset = MultiLeRobotDatasetV3(
            dataset_dirs=[str(root)],
            delta_timestamps={
                "action": [-0.1, 0.0],
                "observation.state": [-0.1, 0.0],
                "observation.images.cam": [-0.1, 0.0],
            },
            nonidle_filter_path=filter_path,
        )

        child = _FakeLeRobotDataset.instances[0]
        self.assertIsNone(child.delta_timestamps)
        self.assertIsNone(child.image_transforms)
        self.assertEqual(len(dataset), 4)

        item = dataset[2]
        self.assertEqual(item["index"].item(), 4)
        self.assertEqual(item["action"].tolist(), [101.0, 104.0])
        self.assertEqual(item["observation.state"].tolist(), [1.0, 4.0])
        self.assertEqual(item["observation.images.cam"][:, 0, 0, 0].tolist(), [1.0, 4.0])
        self.assertEqual(item["action_is_pad"].tolist(), [False, False])

        first = dataset[0]
        self.assertEqual(first["action"].tolist(), [100.0, 100.0])
        self.assertEqual(first["action_is_pad"].tolist(), [True, False])

    def test_strict_nonidle_uses_child_relative_indices_with_selected_episodes(self):
        root = self.tmp_path / "root"
        filter_path = _write_filter(self.tmp_path / "nonidle.json", {"episodes": {"2": [[1, 3]]}})
        _register_root(root, [2, 3], episode_ids=[0, 2], abs_starts=[100, 200])

        dataset = MultiLeRobotDatasetV3(
            dataset_dirs=[str(root)],
            episodes={str(root): [2]},
            delta_timestamps={"action": [-0.1, 0.0]},
            nonidle_filter_path=filter_path,
        )

        item = dataset[0]
        self.assertEqual(item["index"].item(), 201)
        self.assertEqual(item["action"].tolist(), [101.0, 101.0])
        self.assertEqual(item["action_is_pad"].tolist(), [True, False])
        child = _FakeLeRobotDataset.instances[0]
        self.assertLess(max(child.full_get_calls + child.raw_get_calls), len(child))

    def test_strict_nonidle_resolves_multi_root_global_offsets(self):
        root_a = self.tmp_path / "a"
        root_b = self.tmp_path / "b"
        filter_path = _write_filter(self.tmp_path / "nonidle.json", {"0": [[1, 3]]})
        _register_root(root_a, [2])
        _register_root(root_b, [3])

        dataset = MultiLeRobotDatasetV3(
            dataset_dirs=[str(root_a), str(root_b)],
            delta_timestamps={"action": [0.0]},
            nonidle_filter_path=filter_path,
        )

        self.assertEqual(len(dataset), 3)
        item = dataset[1]
        self.assertEqual(item["dataset_index"].item(), 1)
        self.assertEqual(item["index"].item(), 1)
        self.assertEqual(item["action"].tolist(), [101.0])

    def test_strict_nonidle_skips_visual_decode_when_not_training(self):
        root = self.tmp_path / "root"
        filter_path = _write_filter(self.tmp_path / "nonidle.json", {"0": [[0, 2]]})
        _register_root(root, [2], camera_keys=["observation.images.cam"])

        dataset = MultiLeRobotDatasetV3(
            dataset_dirs=[str(root)],
            delta_timestamps={"action": [0.0], "observation.images.cam": [0.0]},
            nonidle_filter_path=filter_path,
        )
        dataset.set_during_training(False)

        item = dataset[0]
        child = _FakeLeRobotDataset.instances[0]
        self.assertEqual(child.full_get_calls, [])
        self.assertNotIn("observation.images.cam", item)
        self.assertEqual(item["action"].tolist(), [100.0])
        self.assertEqual(item["task"], "task0")

    def test_strict_nonidle_stacks_raw_keys_before_hetero_bridge(self):
        root = self.tmp_path / "embodiment"
        filter_path = _write_filter(self.tmp_path / "nonidle.json", {"0": [[0, 2]]})
        features = {
            "raw.left_pose": {"dtype": "float32"},
            "raw.state_pose": {"dtype": "float32"},
        }
        _register_root(root, [2], features=features)

        dataset = MultiLeRobotDatasetV3(
            dataset_dirs=[str(root)],
            delta_timestamps={"action": [-0.1, 0.0], "observation.state": [-0.1, 0.0]},
            nonidle_filter_path=filter_path,
            hetero_bridge={
                "enabled": True,
                "canonical": {
                    "action_key": "action",
                    "state_key": "observation.state",
                    "image_keys": {},
                },
                "embodiments": {
                    "embodiment": {
                        "path_patterns": ["embodiment"],
                        "action": {"left_pose": "raw.left_pose"},
                        "state": {"left_pose": "raw.state_pose"},
                    }
                },
            },
        )

        item = dataset[1]
        self.assertEqual(tuple(item["action"].shape), (2, 16))
        self.assertEqual(tuple(item["observation.state"].shape), (2, 16))
        self.assertEqual(item["action"][:, :7].tolist(), [[0.0] * 7, [1.0] * 7])
        self.assertEqual(item["observation.state"][:, :7].tolist(), [[10.0] * 7, [11.0] * 7])
        self.assertEqual(item["embodiment"], "embodiment")


if __name__ == "__main__":
    unittest.main()
