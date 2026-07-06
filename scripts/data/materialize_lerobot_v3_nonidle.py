#!/usr/bin/env python3
"""Materialize a LeRobot v3 dataset with idle frames physically removed.

The input dataset is never modified. The output dataset is written as a fresh
LeRobot v3 root, so episode indices, frame indices, timestamps, and global
indices are rebuilt from zero by the LeRobot writer.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ACTION_CANDIDATES = ("action", "action.default")
STATE_CANDIDATES = ("observation.state", "observation.state.default")
DEFAULT_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


@dataclass(frozen=True)
class IdleThresholds:
    idle_l2_threshold: float
    idle_arm_l2_threshold: float
    idle_gripper_l2_threshold: float
    min_idle_len: int
    min_non_idle_len: int


@dataclass
class EpisodePlan:
    source_episode_index: int
    source_episode_pos: int
    source_start: int
    source_end: int
    keep_ranges: list[list[int]]
    output_episode_index: int | None = None

    @property
    def source_length(self) -> int:
        return self.source_end - self.source_start

    @property
    def kept_frames(self) -> int:
        return sum(end - start for start, end in self.keep_ranges)


def _get_tqdm() -> Any | None:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm


def _iter_progress(
    iterable: Any,
    *,
    enabled: bool,
    desc: str,
    unit: str,
    total: int | None = None,
    leave: bool = True,
) -> Any:
    tqdm = _get_tqdm() if enabled else None
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave)


def _progress_message(message: str, *, enabled: bool) -> None:
    if not enabled:
        return
    tqdm = _get_tqdm() if enabled else None
    if tqdm is None:
        print(message, flush=True)
        return
    tqdm.write(message)


def _to_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _select_column(names: list[str], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"Could not find {label} column. Tried {candidates}; available columns: {names}")


def _value_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _column_values_to_numpy(values: list[Any]) -> np.ndarray:
    rows = [_value_to_numpy(value).astype(np.float64, copy=False) for value in values]
    arr = np.stack(rows, axis=0)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _raw_item_for_filtering(dataset: Any, idx: int) -> dict[str, Any]:
    raw_getter = getattr(dataset, "get_raw_item", None)
    if callable(raw_getter):
        return dict(raw_getter(idx))

    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None:
        return dict(hf_dataset[idx])

    return dict(dataset[idx])


def _batch_raw_items(dataset: Any, start: int, end: int) -> dict[str, Any]:
    """Read a contiguous slice of raw (parquet) rows in one shot.

    Falls back to per-frame access if the dataset does not expose a slicable
    ``hf_dataset``. Video frames are never decoded here — this only reads the
    scalar/vector columns stored in parquet.
    """
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None:
        try:
            rows = dict(hf_dataset[start:end])
            # HuggingFace returns a dict {column: [values...]}; sanity-check it
            # actually looks column-major before trusting it.
            if rows and all(isinstance(v, (list, tuple)) for v in rows.values()):
                return rows
        except Exception:
            pass
    return {key: [_raw_item_for_filtering(dataset, idx)[key] for idx in range(start, end)]
            for key in _raw_item_for_filtering(dataset, start)}


def _split_arm_gripper(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if delta.shape[1] >= 14:
        arm = np.concatenate([delta[:, :6], delta[:, 7:13]], axis=1)
        gripper = delta[:, [6, 13]]
        return arm, gripper
    return delta, np.zeros((delta.shape[0], 0), dtype=delta.dtype)


def _idle_mask(
    action: np.ndarray,
    state: np.ndarray,
    *,
    idle_l2_threshold: float,
    idle_arm_l2_threshold: float,
    idle_gripper_l2_threshold: float,
) -> np.ndarray:
    delta = action - state
    target_l2 = np.linalg.norm(delta, axis=1)
    arm_delta, gripper_delta = _split_arm_gripper(delta)
    arm_l2 = np.linalg.norm(arm_delta, axis=1)
    gripper_l2 = np.linalg.norm(gripper_delta, axis=1) if gripper_delta.shape[1] else np.zeros_like(target_l2)
    return (
        (target_l2 <= idle_l2_threshold)
        | (
            (arm_l2 <= idle_arm_l2_threshold)
            & (gripper_l2 <= idle_gripper_l2_threshold)
        )
    )


def _true_spans(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return starts, ends


def _compute_keep_ranges(
    idle: np.ndarray,
    *,
    min_idle_len: int,
    min_non_idle_len: int,
) -> list[list[int]]:
    idle_starts, idle_ends = _true_spans(idle)
    remove_mask = np.zeros(len(idle), dtype=bool)
    for start, end in zip(idle_starts, idle_ends, strict=True):
        if int(end) == len(idle):
            continue
        if int(end - start) >= min_idle_len:
            remove_mask[int(start):int(end)] = True

    keep_starts, keep_ends = _true_spans(~remove_mask)
    ranges: list[list[int]] = []
    for start, end in zip(keep_starts, keep_ends, strict=True):
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i >= min_non_idle_len:
            ranges.append([start_i, end_i])
    return ranges


def _iter_episode_rows(episodes: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(len(episodes)):
        row = episodes[idx]
        if hasattr(row, "to_dict"):
            row = row.to_dict()
        rows.append(dict(row))
    return rows


def _load_action_state_for_episode(dataset: Any, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    if start >= end:
        raise ValueError(f"Episode range is empty: [{start}, {end})")
    rows = _batch_raw_items(dataset, start, end)
    names = list(rows)
    action_key = _select_column(names, ACTION_CANDIDATES, "action")
    state_key = _select_column(names, STATE_CANDIDATES, "state")
    action = _column_values_to_numpy(list(rows[action_key]))
    state = _column_values_to_numpy(list(rows[state_key]))
    if action.shape != state.shape:
        raise ValueError(f"Action/state shape mismatch: action={action.shape}, state={state.shape}")
    return action, state


def compute_episode_plans(
    dataset: Any,
    thresholds: IdleThresholds,
    *,
    max_episodes: int | None = None,
    show_progress: bool = True,
) -> list[EpisodePlan]:
    plans: list[EpisodePlan] = []
    rows = _iter_episode_rows(dataset.meta.episodes)
    if max_episodes is not None:
        rows = rows[:max_episodes]

    episode_iter = _iter_progress(
        enumerate(rows),
        enabled=show_progress,
        desc="Computing non-idle ranges",
        unit="episode",
        total=len(rows),
    )
    for episode_pos, row in episode_iter:
        episode_index = int(_to_scalar(row.get("episode_index", episode_pos)))
        start = int(_to_scalar(row["dataset_from_index"]))
        end = int(_to_scalar(row["dataset_to_index"]))
        action, state = _load_action_state_for_episode(dataset, start, end)
        idle = _idle_mask(
            action,
            state,
            idle_l2_threshold=thresholds.idle_l2_threshold,
            idle_arm_l2_threshold=thresholds.idle_arm_l2_threshold,
            idle_gripper_l2_threshold=thresholds.idle_gripper_l2_threshold,
        )
        ranges = _compute_keep_ranges(
            idle,
            min_idle_len=thresholds.min_idle_len,
            min_non_idle_len=thresholds.min_non_idle_len,
        )
        plans.append(
            EpisodePlan(
                source_episode_index=episode_index,
                source_episode_pos=episode_pos,
                source_start=start,
                source_end=end,
                keep_ranges=ranges,
            )
        )
    return plans


def _strip_default_features(features: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in features.items() if key not in DEFAULT_FEATURE_KEYS}


def _make_writer_value(value: Any, feature: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value
    arr = _value_to_numpy(value)
    if feature.get("dtype") in {"image", "video"}:
        return arr
    if tuple(feature.get("shape", ())) == (1,) and arr.ndim == 0:
        return arr.reshape(1)
    return arr


def _frame_for_writer(source_item: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    if "task" not in source_item:
        raise KeyError("Source LeRobot item does not contain a 'task' string.")

    frame: dict[str, Any] = {"task": source_item["task"]}
    for key, feature in features.items():
        if key in DEFAULT_FEATURE_KEYS:
            continue
        if key not in source_item:
            raise KeyError(f"Source LeRobot item is missing feature {key!r}")
        frame[key] = _make_writer_value(source_item[key], feature)
    return frame


def _source_video_lookups(
    source_dataset: Any, episode_index: int
) -> tuple[list[str], dict[str, Path], dict[str, float]]:
    """Resolve per-camera video file paths and episode start offsets.

    Mirrors ``LeRobotDataset._query_videos``: each episode's frames are stored
    sequentially inside a concatenated mp4, so a frame's absolute timestamp in
    the file is ``from_timestamp + episode_relative_timestamp``.
    """
    meta = source_dataset.meta
    video_keys = list(getattr(meta, "video_keys", []) or [])
    if not video_keys:
        return [], {}, {}
    ep = meta.episodes[episode_index]
    video_paths: dict[str, Path] = {}
    from_ts: dict[str, float] = {}
    for vid_key in video_keys:
        video_paths[vid_key] = source_dataset.root / meta.get_video_file_path(episode_index, vid_key)
        from_ts[vid_key] = float(ep[f"videos/{vid_key}/from_timestamp"])
    return video_keys, video_paths, from_ts


def _decode_chunk_for_cameras(
    video_keys: list[str],
    video_paths: dict[str, Path],
    from_ts: dict[str, float],
    timestamps: list[float],
    *,
    tolerance_s: float,
    video_backend: str | None,
) -> dict[str, Any]:
    """Batch-decode one contiguous span of frames for every camera.

    Each camera issues a single ``decode_video_frames`` call with all timestamps
    at once, so the container is opened once and frames are decoded in a single
    forward pass instead of once per frame.
    """
    from lerobot.datasets.video_utils import decode_video_frames

    decoded: dict[str, Any] = {}
    for vid_key in video_keys:
        shifted = [from_ts[vid_key] + float(t) for t in timestamps]
        decoded[vid_key] = decode_video_frames(
            video_paths[vid_key], shifted, tolerance_s, video_backend
        )
    return decoded


def _task_string_at(source_dataset: Any, task_index: int) -> str:
    tasks = getattr(source_dataset.meta, "tasks", None)
    if tasks is None:
        raise KeyError("Source dataset meta has no 'tasks' table to resolve task strings.")
    row = tasks.iloc[int(task_index)]
    return getattr(row, "name", None) or str(row.iloc[0])


def _prepare_chunk_batch(
    source_dataset: Any,
    *,
    video_keys: list[str],
    video_paths: dict[str, Path],
    from_ts: dict[str, float],
    tolerance_s: float,
    video_backend: str | None,
    abs_lo: int,
    abs_hi: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch everything needed to emit a chunk: parquet rows + decoded video.

    This is the only failure-prone step (video decode). Kept separate from the
    emit loop so that, on failure, the per-frame fallback can safely re-read the
    whole chunk without risking double ``add_frame`` calls or progress overshoot.
    """
    rows = _batch_raw_items(source_dataset, abs_lo, abs_hi)
    timestamps = [float(_to_scalar(t)) for t in rows["timestamp"]]
    decoded = (
        _decode_chunk_for_cameras(
            video_keys,
            video_paths,
            from_ts,
            timestamps,
            tolerance_s=tolerance_s,
            video_backend=video_backend,
        )
        if video_keys
        else {}
    )
    return rows, decoded


def _emit_chunk_batch(
    source_dataset: Any,
    output_dataset: Any,
    features: dict[str, Any],
    rows: dict[str, Any],
    decoded: dict[str, Any],
    *,
    abs_lo: int,
    abs_hi: int,
    frame_progress_iter: Any,
) -> None:
    """Build per-frame dicts from a prepared chunk and feed them to add_frame."""
    for i in range(abs_hi - abs_lo):
        source_item: dict[str, Any] = {}
        for key in features:
            if key in DEFAULT_FEATURE_KEYS:
                continue
            if key in decoded:
                source_item[key] = decoded[key][i]
            elif key in rows:
                source_item[key] = rows[key][i]
        task_idx = int(_to_scalar(rows["task_index"][i]))
        source_item["task"] = _task_string_at(source_dataset, task_idx)
        next(frame_progress_iter)
        output_dataset.add_frame(_frame_for_writer(source_item, features))


def _write_chunk_per_frame(
    source_dataset: Any,
    output_dataset: Any,
    features: dict[str, Any],
    *,
    abs_lo: int,
    abs_hi: int,
    frame_progress_iter: Any,
) -> None:
    """Fallback: read + write a chunk one frame at a time via ``__getitem__``."""
    for source_idx in range(abs_lo, abs_hi):
        next(frame_progress_iter)
        source_item = source_dataset[source_idx]
        output_dataset.add_frame(_frame_for_writer(source_item, features))


def _materialize_episode(
    source_dataset: Any,
    output_dataset: Any,
    plan: EpisodePlan,
    features: dict[str, Any],
    *,
    decode_chunk_size: int,
    show_progress: bool,
    frame_progress_iter: Any,
) -> None:
    """Write all kept frames of one source episode, batch-decoding per chunk."""
    video_keys, video_paths, from_ts = _source_video_lookups(
        source_dataset, plan.source_episode_index
    )
    tolerance_s = float(getattr(source_dataset, "tolerance_s", 1e-4))
    video_backend = getattr(source_dataset, "video_backend", None)
    chunk = max(1, int(decode_chunk_size))

    for rng_start, rng_end in plan.keep_ranges:
        pos = rng_start
        while pos < rng_end:
            chunk_end = min(rng_end, pos + chunk)
            abs_lo = plan.source_start + pos
            abs_hi = plan.source_start + chunk_end
            try:
                rows, decoded = _prepare_chunk_batch(
                    source_dataset,
                    video_keys=video_keys,
                    video_paths=video_paths,
                    from_ts=from_ts,
                    tolerance_s=tolerance_s,
                    video_backend=video_backend,
                    abs_lo=abs_lo,
                    abs_hi=abs_hi,
                )
            except Exception as exc:  # noqa: BLE001 - fall back to the safe per-frame path
                _progress_message(
                    f"batch decode failed for [{abs_lo},{abs_hi}) "
                    f"({exc!r}); falling back to per-frame read",
                    enabled=show_progress,
                )
                _write_chunk_per_frame(
                    source_dataset,
                    output_dataset,
                    features,
                    abs_lo=abs_lo,
                    abs_hi=abs_hi,
                    frame_progress_iter=frame_progress_iter,
                )
            else:
                _emit_chunk_batch(
                    source_dataset,
                    output_dataset,
                    features,
                    rows,
                    decoded,
                    abs_lo=abs_lo,
                    abs_hi=abs_hi,
                    frame_progress_iter=frame_progress_iter,
                )
            pos = chunk_end


def _load_lerobot_dataset(dataset_dir: Path, video_backend: str | None) -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "materialize_lerobot_v3_nonidle.py requires the external `lerobot` package "
            "with LeRobot v3 dataset support."
        ) from exc

    kwargs = {
        "repo_id": dataset_dir.name,
        "root": dataset_dir,
        "download_videos": True,
        "video_backend": video_backend,
    }
    if "return_uint8" in inspect.signature(LeRobotDataset).parameters:
        kwargs["return_uint8"] = True
    return LeRobotDataset(**kwargs)


def _create_output_dataset(
    source_dataset: Any,
    output_dir: Path,
    *,
    repo_id: str,
    video_backend: str | None,
    vcodec: str | None = None,
    encoder_threads: int | None = None,
    image_writer_threads: int | None = None,
    streaming_encoding: bool = False,
) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta = source_dataset.meta
    features = _strip_default_features(meta.features)
    kwargs = {
        "repo_id": repo_id,
        "fps": int(meta.fps),
        "features": features,
        "root": output_dir,
        "robot_type": getattr(meta, "robot_type", None),
        "use_videos": bool(getattr(meta, "video_keys", [])),
        "video_backend": video_backend,
    }
    # Write-side tuning. These are filtered against LeRobotDataset.create's
    # signature below, so unsupported keys are dropped silently on older
    # lerobot versions.
    if vcodec is not None:
        kwargs["vcodec"] = vcodec
    if encoder_threads is not None:
        kwargs["encoder_threads"] = encoder_threads
    if image_writer_threads is not None and image_writer_threads > 0:
        kwargs["image_writer_threads"] = image_writer_threads
    if streaming_encoding:
        kwargs["streaming_encoding"] = True
    for attr in ("chunks_size", "data_files_size_in_mb", "video_files_size_in_mb"):
        if hasattr(meta, attr):
            kwargs[attr] = getattr(meta, attr)

    supported_kwargs = _filter_supported_kwargs(LeRobotDataset.create, kwargs)
    output_dataset = LeRobotDataset.create(**supported_kwargs)

    deferred_chunk_settings = {
        key: kwargs[key]
        for key in ("chunks_size", "data_files_size_in_mb", "video_files_size_in_mb")
        if key in kwargs and key not in supported_kwargs
    }
    if deferred_chunk_settings:
        chunk_settings_target = output_dataset
        if not hasattr(chunk_settings_target, "update_chunk_settings"):
            chunk_settings_target = getattr(output_dataset, "meta", None)
        update_chunk_settings = getattr(chunk_settings_target, "update_chunk_settings", None)
        if callable(update_chunk_settings):
            update_chunk_settings(**deferred_chunk_settings)

    return output_dataset


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
    shutil.rmtree(output_dir)


def materialize_nonidle_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    thresholds: IdleThresholds,
    video_backend: str | None = "pyav",
    overwrite: bool = False,
    max_episodes: int | None = None,
    repo_id: str | None = None,
    parallel_encoding: bool = True,
    vcodec: str | None = None,
    encoder_threads: int | None = None,
    image_writer_threads: int | None = None,
    streaming_encoding: bool = False,
    decode_chunk_size: int = 256,
    show_progress: bool = True,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if streaming_encoding:
        _progress_message(
            "WARNING: streaming_encoding is enabled. Under encoder pressure the "
            "streaming encoder DROPS video frames (see lerobot video_utils.feed_frame), "
            "which desyncs the output video from its parquet rows. Only use this for "
            "speed and verify frame counts afterwards, or leave it off for a correct "
            "materialization.",
            enabled=show_progress,
        )

    source_dataset = _load_lerobot_dataset(input_dir, video_backend)
    plans = compute_episode_plans(
        source_dataset,
        thresholds,
        max_episodes=max_episodes,
        show_progress=show_progress,
    )

    total_frames = sum(plan.source_length for plan in plans)
    kept_frames = sum(plan.kept_frames for plan in plans)
    if kept_frames == 0:
        raise ValueError("Idle filtering removed all frames; refusing to create an empty dataset.")

    _prepare_output_dir(output_dir, overwrite)
    output_dataset = _create_output_dataset(
        source_dataset,
        output_dir,
        repo_id=repo_id or f"{input_dir.name}_nonidle",
        video_backend=video_backend,
        vcodec=vcodec,
        encoder_threads=encoder_threads,
        image_writer_threads=image_writer_threads,
        streaming_encoding=streaming_encoding,
    )

    output_episode_index = 0
    frame_progress = _iter_progress(
        range(kept_frames),
        enabled=show_progress,
        desc="Writing kept frames",
        unit="frame",
        total=kept_frames,
    )
    frame_progress_iter = iter(frame_progress)
    try:
        for count, plan in enumerate(plans, start=1):
            if plan.kept_frames == 0:
                _progress_message(
                    f"skipping source episode {plan.source_episode_index}: no kept frames",
                    enabled=show_progress,
                )
                continue
            plan.output_episode_index = output_episode_index
            _materialize_episode(
                source_dataset,
                output_dataset,
                plan,
                source_dataset.meta.features,
                decode_chunk_size=decode_chunk_size,
                show_progress=show_progress,
                frame_progress_iter=frame_progress_iter,
            )
            output_dataset.save_episode(parallel_encoding=parallel_encoding)
            _progress_message(
                f"wrote episode {output_episode_index} from source episode "
                f"{plan.source_episode_index}: kept {plan.kept_frames}/{plan.source_length} frames",
                enabled=show_progress,
            )
            output_episode_index += 1
    finally:
        if hasattr(frame_progress, "close"):
            frame_progress.close()
        output_dataset.finalize()

    report = {
        "format": "imagewam_lerobot_v3_materialized_nonidle_v1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "thresholds": {
            "idle_l2_threshold": thresholds.idle_l2_threshold,
            "idle_arm_l2_threshold": thresholds.idle_arm_l2_threshold,
            "idle_gripper_l2_threshold": thresholds.idle_gripper_l2_threshold,
            "min_idle_len": thresholds.min_idle_len,
            "min_non_idle_len": thresholds.min_non_idle_len,
        },
        "summary": {
            "source_episodes": len(plans),
            "output_episodes": sum(plan.output_episode_index is not None for plan in plans),
            "total_frames": total_frames,
            "kept_frames": kept_frames,
            "removed_frames": total_frames - kept_frames,
            "kept_rate": kept_frames / max(total_frames, 1),
        },
        "episodes": [
            {
                "source_episode_index": plan.source_episode_index,
                "source_episode_pos": plan.source_episode_pos,
                "source_length": plan.source_length,
                "output_episode_index": plan.output_episode_index,
                "kept_frames": plan.kept_frames,
                "removed_frames": plan.source_length - plan.kept_frames,
                "keep_ranges": plan.keep_ranges,
            }
            for plan in plans
        ],
    }

    report_path = output_dir / "nonidle_materialize_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote materialization report: {report_path}")
    print(
        f"episodes={report['summary']['output_episodes']}/{len(plans)} "
        f"total_frames={total_frames} kept_frames={kept_frames} "
        f"removed_frames={total_frames - kept_frames} "
        f"kept_rate={report['summary']['kept_rate'] * 100:.2f}%"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--idle-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--idle-arm-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--idle-gripper-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--min-idle-len", type=int, default=5)
    parser.add_argument("--min-non-idle-len", type=int, default=1)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--repo-id", default=None, help="Repo id stored in output metadata.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--no-parallel-encoding",
        action="store_true",
        help="Encode camera videos sequentially when saving each episode.",
    )
    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=256,
        help=(
            "Number of kept frames decoded per batched decode_video_frames call. Larger "
            "reduces per-frame container-open/seek overhead; smaller bounds peak memory. "
            "Each camera holds chunk*H*W*C*4 bytes (float32) during decode."
        ),
    )
    parser.add_argument(
        "--vcodec",
        default="h264",
        help=(
            "Output video codec (passed to LeRobotDataset.create). Default 'h264' (libx264) "
            "is much faster to encode than lerobot's 'libsvtav1' default. Use 'auto' for a "
            "hardware encoder when available, or 'libsvtav1' for smallest files."
        ),
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Threads per camera encoder process. None lets ffmpeg decide.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=None,
        help=(
            "Background threads for writing the per-frame temp PNGs that the non-streaming "
            "encoder reads back. Drop-free (blocks on backpressure). Defaults to "
            "max(1, cpu//2) when supported by the lerobot version."
        ),
    )
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help=(
            "Encode output videos as frames are added (overlaps encode with decode). WARNING: "
            "the streaming encoder DROPS frames when its queue fills, which desyncs video from "
            "parquet. Off by default; only enable for speed and verify frame counts afterwards."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    thresholds = IdleThresholds(
        idle_l2_threshold=args.idle_l2_threshold,
        idle_arm_l2_threshold=args.idle_arm_l2_threshold,
        idle_gripper_l2_threshold=args.idle_gripper_l2_threshold,
        min_idle_len=args.min_idle_len,
        min_non_idle_len=args.min_non_idle_len,
    )
    image_writer_threads = args.image_writer_threads
    if image_writer_threads is None:
        image_writer_threads = max(1, (os.cpu_count() or 2) // 2)
    materialize_nonidle_dataset(
        args.input_dir,
        args.output_dir,
        thresholds=thresholds,
        video_backend=args.video_backend,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        repo_id=args.repo_id,
        parallel_encoding=not args.no_parallel_encoding,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        image_writer_threads=image_writer_threads,
        streaming_encoding=args.streaming_encoding,
        decode_chunk_size=args.decode_chunk_size,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
