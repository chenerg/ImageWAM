import unittest
from pathlib import Path
from unittest.mock import patch

import datasets

from imagewam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset


class _FakeHFDataset:
    def __len__(self):
        return 4

    def unique(self, column):
        assert column == "episode_index"
        return [0, 1, 2, 3]


class _FakeMetadata:
    total_episodes = 4
    video_keys = ["camera"]

    def get_video_file_path(self, episode_index, video_key):
        assert video_key == "camera"
        # All episodes are consolidated into the same chunked video file.
        return Path("videos/camera/chunk-000/file-000.mp4")


class LeRobotDatasetCacheTest(unittest.TestCase):
    def test_checks_shared_video_file_only_once(self):
        dataset = LeRobotDataset.__new__(LeRobotDataset)
        dataset.hf_dataset = _FakeHFDataset()
        dataset.episodes = None
        dataset.meta = _FakeMetadata()
        dataset.root = Path("/tmp/dataset")

        with patch.object(Path, "exists", autospec=True, return_value=True) as exists:
            self.assertTrue(dataset._check_cached_episodes_sufficient())

        exists.assert_called_once_with(
            Path("/tmp/dataset/videos/camera/chunk-000/file-000.mp4")
        )

    def test_builds_index_mapping_without_running_transform(self):
        dataset = LeRobotDataset.__new__(LeRobotDataset)
        dataset.hf_dataset = datasets.Dataset.from_dict(
            {"index": [4, 5, 9], "value": [10, 20, 30]}
        )

        def fail_if_called(_batch):
            raise AssertionError("The torch transform should not run while building the index mapping")

        dataset.hf_dataset.set_transform(fail_if_called)

        self.assertEqual(dataset._build_absolute_to_relative_idx(), {4: 0, 5: 1, 9: 2})


if __name__ == "__main__":
    unittest.main()
