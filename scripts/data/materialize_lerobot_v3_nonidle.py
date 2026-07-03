#!/usr/bin/env python3
"""Materialize a LeRobot v3 dataset with idle frames physically removed.

The input dataset is never modified. The output dataset is written as a fresh
LeRobot v3 root, so episode indices, frame indices, timestamps, and global
indices are rebuilt from zero by the LeRobot writer.
"""

from __future__ import annotations

import argparse
import copy
import json
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
    raw_items = [dataset.get_raw_item(idx) for idx in range(start, end)]
    if not raw_items:
        raise ValueError(f"Episode range is empty: [{start}, {end})")
    names = list(raw_items[0])
    action_key = _select_column(names, ACTION_CANDIDATES, "action")
    state_key = _select_column(names, STATE_CANDIDATES, "state")
    action = _column_values_to_numpy([item[action_key] for item in raw_items])
    state = _column_values_to_numpy([item[state_key] for item in raw_items])
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


def _expanded_source_indices(plan: EpisodePlan) -> list[int]:
    indices: list[int] = []
    for start, end in plan.keep_ranges:
        indices.extend(range(plan.source_start + start, plan.source_start + end))
    return indices


def _load_lerobot_dataset(dataset_dir: Path, video_backend: str | None) -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "materialize_lerobot_v3_nonidle.py requires the external `lerobot` package "
            "with LeRobot v3 dataset support."
        ) from exc

    return LeRobotDataset(
        repo_id=dataset_dir.name,
        root=dataset_dir,
        download_videos=True,
        video_backend=video_backend,
        return_uint8=True,
    )


def _create_output_dataset(
    source_dataset: Any,
    output_dir: Path,
    *,
    repo_id: str,
    video_backend: str | None,
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
    for attr in ("chunks_size", "data_files_size_in_mb", "video_files_size_in_mb"):
        if hasattr(meta, attr):
            kwargs[attr] = getattr(meta, attr)

    try:
        return LeRobotDataset.create(**kwargs)
    except TypeError:
        for attr in ("chunks_size", "data_files_size_in_mb", "video_files_size_in_mb"):
            kwargs.pop(attr, None)
        return LeRobotDataset.create(**kwargs)


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
    show_progress: bool = True,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

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
            for source_idx in _expanded_source_indices(plan):
                next(frame_progress_iter)
                source_item = source_dataset[source_idx]
                output_dataset.add_frame(_frame_for_writer(source_item, source_dataset.meta.features))
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
    materialize_nonidle_dataset(
        args.input_dir,
        args.output_dir,
        thresholds=thresholds,
        video_backend=args.video_backend,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        repo_id=args.repo_id,
        parallel_encoding=not args.no_parallel_encoding,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
