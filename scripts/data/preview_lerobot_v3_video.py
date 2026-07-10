#!/usr/bin/env python3
"""Export one episode camera stream from a local LeRobot Dataset v3 dataset."""

from __future__ import annotations

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import pyarrow.parquet as pq


DEFAULT_OUTPUT_DIR = Path("./output/previews")
DEFAULT_VIDEO_FEATURE_NAME = "observation.images.cam_high"


def _load_episode(dataset_dir: Path, episode_index: int) -> dict[str, Any]:
    episode_files = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        raise FileNotFoundError(
            f"No LeRobot v3 episode metadata found under {dataset_dir / 'meta' / 'episodes'}"
        )

    for episode_file in episode_files:
        table = pq.read_table(episode_file, filters=[("episode_index", "=", episode_index)])
        if table.num_rows:
            if table.num_rows > 1:
                raise ValueError(f"Episode index {episode_index} occurs more than once in the metadata")
            return table.to_pylist()[0]

    raise IndexError(f"Episode index {episode_index} was not found in {dataset_dir}")


def _safe_feature_name(feature_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", feature_name).strip("._-") or "video"


def _resolve_source_video(
    dataset_dir: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    video_feature_name: str,
) -> Path:
    feature = info.get("features", {}).get(video_feature_name)
    if feature is None:
        available = sorted(
            name for name, value in info.get("features", {}).items() if value.get("dtype") == "video"
        )
        raise KeyError(
            f"Video feature {video_feature_name!r} does not exist. Available video features: {available}"
        )
    if feature.get("dtype") != "video":
        raise ValueError(f"Feature {video_feature_name!r} has dtype {feature.get('dtype')!r}, not 'video'")

    video_path_template = info.get("video_path")
    if not isinstance(video_path_template, str):
        raise ValueError("meta/info.json does not contain a valid video_path template")

    prefix = f"videos/{video_feature_name}"
    try:
        relative_path = video_path_template.format(
            video_key=video_feature_name,
            chunk_index=int(episode[f"{prefix}/chunk_index"]),
            file_index=int(episode[f"{prefix}/file_index"]),
        )
    except KeyError as exc:
        raise KeyError(
            f"Episode {episode['episode_index']} has no video metadata for {video_feature_name!r}"
        ) from exc

    source_path = dataset_dir / relative_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source_path}")
    return source_path


def _load_episode_frame_indices(
    dataset_dir: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
) -> list[int]:
    data_path_template = info.get("data_path")
    if not isinstance(data_path_template, str):
        raise ValueError("meta/info.json does not contain a valid data_path template")
    try:
        relative_path = data_path_template.format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
        )
    except KeyError as exc:
        raise KeyError(
            f"Episode {episode['episode_index']} is missing required data file metadata: {exc.args[0]}"
        ) from exc

    data_path = dataset_dir / relative_path
    if not data_path.is_file():
        raise FileNotFoundError(f"Episode data parquet does not exist: {data_path}")
    table = pq.read_table(
        data_path,
        columns=["frame_index"],
        filters=[("episode_index", "=", int(episode["episode_index"]))],
    )
    frame_indices = [int(value) for value in table["frame_index"].to_pylist()]
    expected_count = int(episode["length"])
    if len(frame_indices) != expected_count:
        raise ValueError(
            f"Episode {episode['episode_index']} metadata expects {expected_count} frames, "
            f"but its data parquet contains {len(frame_indices)} rows"
        )
    if not frame_indices:
        raise ValueError(f"Episode {episode['episode_index']} contains no frames to export")
    if frame_indices[0] < 0 or any(
        current <= previous for previous, current in zip(frame_indices, frame_indices[1:], strict=False)
    ):
        raise ValueError(
            f"Episode {episode['episode_index']} frame_index values must be non-negative and strictly increasing: "
            f"{frame_indices}"
        )
    return frame_indices


def _encode_episode_segment(
    source_path: Path,
    output_path: Path,
    *,
    from_timestamp: float,
    frame_indices: list[int],
    fps: float,
    codec: str,
) -> int:
    """Decode selected original frame indices and encode them as a standalone MP4."""
    if not frame_indices:
        raise ValueError("At least one frame index is required")
    if fps <= 0:
        raise ValueError(f"Dataset fps must be positive, got {fps}")

    written = 0
    with av.open(str(source_path), mode="r") as input_container:
        if not input_container.streams.video:
            raise ValueError(f"No video stream found in {source_path}")
        input_stream = input_container.streams.video[0]
        # Seeking is keyframe-based, so decode from the preceding keyframe and discard
        # frames until the episode boundary.
        if input_stream.time_base is not None:
            seek_offset = max(0, int(from_timestamp / float(input_stream.time_base)))
            input_container.seek(seek_offset, stream=input_stream, backward=True)

        with av.open(str(output_path), mode="w") as output_container:
            output_stream = output_container.add_stream(codec, rate=Fraction(str(fps)))
            output_stream.pix_fmt = "yuv420p"
            output_stream.width = input_stream.codec_context.width
            output_stream.height = input_stream.codec_context.height

            boundary_tolerance = 0.5 / fps
            source_frame_index = 0
            requested_position = 0
            for frame in input_container.decode(input_stream):
                timestamp = frame.time
                if timestamp is None and frame.pts is not None and frame.time_base is not None:
                    timestamp = float(frame.pts * frame.time_base)
                if timestamp is not None and timestamp < from_timestamp - boundary_tolerance:
                    continue

                if source_frame_index == frame_indices[requested_position]:
                    # Recreate the frame so timestamps from the shared source MP4 do not
                    # leak into the standalone output timeline.
                    output_frame = av.VideoFrame.from_ndarray(
                        frame.to_ndarray(format="rgb24"), format="rgb24"
                    )
                    for packet in output_stream.encode(output_frame):
                        output_container.mux(packet)
                    written += 1
                    requested_position += 1
                source_frame_index += 1
                if requested_position == len(frame_indices):
                    break

            for packet in output_stream.encode():
                output_container.mux(packet)

    if written != len(frame_indices):
        raise ValueError(
            f"Could only decode {written}/{len(frame_indices)} requested episode frames from {source_path} "
            f"starting at {from_timestamp:.6f}s; last requested frame_index={frame_indices[-1]}"
        )
    return written


def export_episode_video(
    dataset_dir: Path | str,
    *,
    episode_index: int = 0,
    video_feature_name: str = DEFAULT_VIDEO_FEATURE_NAME,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    output_path: Path | str | None = None,
    overwrite: bool = False,
    codec: str = "libx264",
) -> Path:
    """Export one episode's video feature and return the output path."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot info.json does not exist: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError(
            f"Expected a LeRobot Dataset v3 dataset, got codebase_version={info.get('codebase_version')!r}"
        )

    episode = _load_episode(dataset_dir, episode_index)
    source_path = _resolve_source_video(dataset_dir, info, episode, video_feature_name)
    frame_indices = _load_episode_frame_indices(dataset_dir, info, episode)
    prefix = f"videos/{video_feature_name}"
    try:
        from_timestamp = float(episode[f"{prefix}/from_timestamp"])
        fps = float(info["fps"])
    except KeyError as exc:
        raise KeyError(f"Required LeRobot v3 metadata field is missing: {exc.args[0]}") from exc

    if output_path is None:
        output_dir = Path(output_dir).expanduser()
        output_path = output_dir / (
            f"{_safe_feature_name(dataset_dir.name)}_episode_{episode_index:06d}_"
            f"{_safe_feature_name(video_feature_name)}.mp4"
        )
    else:
        output_path = Path(output_path).expanduser()
    output_path = output_path.resolve()
    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"Output path must end in .mp4: {output_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    try:
        _encode_episode_segment(
            source_path,
            temporary_path,
            from_timestamp=from_timestamp,
            frame_indices=frame_indices,
            fps=fps,
            codec=codec,
        )
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Local LeRobot Dataset v3 root directory")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--video-feature-name", default=DEFAULT_VIDEO_FEATURE_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--codec", default="libx264", help="PyAV/FFmpeg output codec (default: libx264)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = export_episode_video(
        args.dataset_dir,
        episode_index=args.episode_index,
        video_feature_name=args.video_feature_name,
        output_dir=args.output_dir,
        output_path=args.output_path,
        overwrite=args.overwrite,
        codec=args.codec,
    )
    print(f"Saved preview video: {output_path}")


if __name__ == "__main__":
    main()
