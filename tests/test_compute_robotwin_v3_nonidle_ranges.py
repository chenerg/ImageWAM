import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "compute_robotwin_v3_nonidle_ranges.py"
_SPEC = importlib.util.spec_from_file_location("compute_robotwin_v3_nonidle_ranges", _SCRIPT_PATH)
compute_v3 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = compute_v3
_SPEC.loader.exec_module(compute_v3)


def _write_fake_lerobot_v3(root: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 10,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        ),
        encoding="utf-8",
    )

    episode_meta_dir = root / "meta" / "episodes" / "chunk-000"
    episode_meta_dir.mkdir(parents=True)
    episode_table = pa.table(
        {
            "episode_index": [0, 1],
            "length": [6, 4],
            "dataset_from_index": [0, 6],
            "dataset_to_index": [6, 10],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 0],
        }
    )
    pq.write_table(episode_table, episode_meta_dir / "file-000.parquet")

    # Episode 0 idle mask: [False, True, True, False, False, True].
    # Episode 1 idle mask: [False, True, True, False].
    # Terminal idle spans are intentionally kept by the script.
    deltas = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    data_table = pa.table(
        {
            "index": list(range(10)),
            "episode_index": [0] * 6 + [1] * 4,
            "action": [[delta] for delta in deltas],
            "observation.state": [[0.0] for _ in deltas],
        }
    )
    pq.write_table(data_table, data_dir / "file-000.parquet")


class ComputeRobotwinV3NonidleRangesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_compute_ranges_from_lerobot_v3_chunk_files(self):
        dataset_root = self.tmp_path / "dataset"
        output = self.tmp_path / "nonidle_ranges.json"
        _write_fake_lerobot_v3(dataset_root)

        payload = compute_v3.compute_nonidle_ranges(
            dataset_root,
            output,
            thresholds=compute_v3.IdleThresholds(
                idle_l2_threshold=1e-3,
                idle_arm_l2_threshold=1e-3,
                idle_gripper_l2_threshold=1e-3,
                min_idle_len=2,
                min_non_idle_len=1,
            ),
            progress_every=0,
        )

        self.assertTrue(output.exists())
        self.assertEqual(payload["source_format"], "lerobot_v3")
        self.assertEqual(payload["episodes"], {"0": [[0, 1], [3, 6]], "1": [[0, 1], [3, 4]]})
        self.assertEqual(payload["summary"]["episodes"], 2)
        self.assertEqual(payload["summary"]["data_files"], 1)
        self.assertEqual(payload["summary"]["total_steps"], 10)
        self.assertEqual(payload["summary"]["idle_steps"], 5)
        self.assertEqual(payload["summary"]["kept_steps"], 6)

        disk_payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(disk_payload["episodes"], payload["episodes"])

    def test_keep_ranges_leave_terminal_idle(self):
        idle = np.array([False, True, True, False, True, True])
        ranges = compute_v3._compute_keep_ranges(idle, min_idle_len=2, min_non_idle_len=1)
        self.assertEqual(ranges, [[0, 1], [3, 6]])


if __name__ == "__main__":
    unittest.main()
