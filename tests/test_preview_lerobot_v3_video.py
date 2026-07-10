import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "preview_lerobot_v3_video.py"
_SPEC = importlib.util.spec_from_file_location("preview_lerobot_v3_video", _SCRIPT_PATH)
preview = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = preview
_SPEC.loader.exec_module(preview)


def _write_video(path: Path, values: list[int], fps: int) -> None:
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in values:
            image = np.full((16, 16, 3), value, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_fake_dataset(root: Path) -> None:
    feature_name = preview.DEFAULT_VIDEO_FEATURE_NAME
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 10,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {feature_name: {"dtype": "video", "shape": [16, 16, 3]}},
            }
        ),
        encoding="utf-8",
    )
    episode_dir = root / "meta" / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "length": [3, 2],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 0],
                f"videos/{feature_name}/chunk_index": [0, 0],
                f"videos/{feature_name}/file_index": [0, 0],
                f"videos/{feature_name}/from_timestamp": [0.0, 0.3],
                f"videos/{feature_name}/to_timestamp": [0.3, 0.6],
            }
        ),
        episode_dir / "file-000.parquet",
    )
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 0, 1, 1],
                "frame_index": [0, 1, 2, 0, 2],
            }
        ),
        data_dir / "file-000.parquet",
    )
    _write_video(
        root / "videos" / feature_name / "chunk-000" / "file-000.mp4",
        [10, 20, 30, 200, 50, 240],
        10,
    )


class PreviewLeRobotV3VideoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name) / "dataset"
        _write_fake_dataset(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exports_only_requested_episode(self):
        output = preview.export_episode_video(
            self.root,
            episode_index=1,
            output_dir=Path(self.tmp.name) / "previews",
        )

        with av.open(str(output)) as container:
            frames = list(container.decode(video=0))
        self.assertEqual(len(frames), 2)
        self.assertEqual(
            output.name,
            "dataset_episode_000001_observation.images.cam_high.mp4",
        )
        means = [float(frame.to_ndarray(format="rgb24").mean()) for frame in frames]
        self.assertGreater(means[0], 150)
        self.assertGreater(means[1], 200)

    def test_rejects_unknown_video_feature(self):
        with self.assertRaisesRegex(KeyError, "Available video features"):
            preview.export_episode_video(self.root, video_feature_name="missing.camera")

    def test_requires_overwrite_for_existing_output(self):
        output = Path(self.tmp.name) / "preview.mp4"
        output.touch()
        with self.assertRaises(FileExistsError):
            preview.export_episode_video(self.root, output_path=output)


if __name__ == "__main__":
    unittest.main()
