#!/usr/bin/env python3
"""Compute non-idle keep ranges for a LeRobot v3 dataset.

The output JSON does not modify the dataset. It can be passed to the ImageWAM
LeRobot v3 dataloader via `data.train.nonidle_filter_path=...`.

This script targets the LeRobot v3 chunk/file layout:

    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet

It reads only metadata plus action/state columns from data parquet files. Videos
are never decoded or re-encoded.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ACTION_CANDIDATES = ("action", "action.default")
STATE_CANDIDATES = ("observation.state", "observation.state.default")
INDEX_CANDIDATES = ("index",)
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


@dataclass(frozen=True)
class IdleThresholds:
    idle_l2_threshold: float
    idle_arm_l2_threshold: float
    idle_gripper_l2_threshold: float
    min_idle_len: int
    min_non_idle_len: int


@dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    dataset_from_index: int
    dataset_to_index: int
    data_chunk_index: int
    data_file_index: int

    @property
    def length(self) -> int:
        return self.dataset_to_index - self.dataset_from_index


def _to_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_int(value: Any) -> int:
    return int(_to_scalar(value))


def _select_column(names: list[str], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"Could not find {label} column. Tried {candidates}; available columns: {names}")


def _column_to_numpy(table: Any, column_name: str, *, dtype: Any = np.float64) -> np.ndarray:
    arr = np.asarray(table[column_name].to_pylist(), dtype=dtype)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _index_column_to_numpy(table: Any, column_name: str) -> np.ndarray:
    return np.asarray([_to_int(value) for value in table[column_name].to_pylist()], dtype=np.int64)


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


def _read_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot info.json not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _data_file_path(dataset_dir: Path, info: dict[str, Any], chunk_index: int, file_index: int) -> Path:
    data_path = info.get("data_path") or DEFAULT_DATA_PATH
    if not isinstance(data_path, str):
        raise ValueError(f"Invalid data_path in info.json: {data_path!r}")
    try:
        formatted = data_path.format(
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=chunk_index,
            episode_index=file_index,
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported LeRobot v3 data_path template: {data_path!r}") from exc
    return dataset_dir / formatted


def _load_episode_specs(dataset_dir: Path, *, max_episodes: int | None = None) -> list[EpisodeSpec]:
    import pyarrow.parquet as pq

    episode_paths = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No LeRobot v3 episode metadata parquet files found under {dataset_dir}")

    rows: list[dict[str, Any]] = []
    for path in episode_paths:
        rows.extend(pq.read_table(path).to_pylist())

    specs: list[EpisodeSpec] = []
    for row in sorted(rows, key=lambda item: _to_int(item["episode_index"])):
        specs.append(
            EpisodeSpec(
                episode_index=_to_int(row["episode_index"]),
                dataset_from_index=_to_int(row["dataset_from_index"]),
                dataset_to_index=_to_int(row["dataset_to_index"]),
                data_chunk_index=_to_int(row["data/chunk_index"]),
                data_file_index=_to_int(row["data/file_index"]),
            )
        )
    if max_episodes is not None:
        specs = specs[:max_episodes]
    return specs


def _load_data_file_columns(path: Path) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    if not path.exists():
        raise FileNotFoundError(f"LeRobot v3 data parquet not found: {path}")

    schema = pq.read_schema(path)
    names = list(schema.names)
    action_key = _select_column(names, ACTION_CANDIDATES, "action")
    state_key = _select_column(names, STATE_CANDIDATES, "state")
    columns = [action_key, state_key]

    index_key: str | None = None
    try:
        index_key = _select_column(names, INDEX_CANDIDATES, "index")
        columns.insert(0, index_key)
    except KeyError:
        pass

    table = pq.read_table(path, columns=columns)
    indices = _index_column_to_numpy(table, index_key) if index_key is not None else None
    action = _column_to_numpy(table, action_key)
    state = _column_to_numpy(table, state_key)
    if action.shape != state.shape:
        raise ValueError(f"Action/state shape mismatch in {path}: action={action.shape}, state={state.shape}")
    return indices, action, state


def _select_episode_rows(
    spec: EpisodeSpec,
    *,
    file_start_index: int,
    indices: np.ndarray | None,
    action: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if indices is None:
        start = spec.dataset_from_index - file_start_index
        end = spec.dataset_to_index - file_start_index
        ep_action = action[start:end]
        ep_state = state[start:end]
    else:
        mask = (indices >= spec.dataset_from_index) & (indices < spec.dataset_to_index)
        ep_action = action[mask]
        ep_state = state[mask]

    if ep_action.shape[0] != spec.length:
        raise ValueError(
            f"Episode {spec.episode_index} expected {spec.length} rows, found {ep_action.shape[0]} "
            f"for data file chunk={spec.data_chunk_index} file={spec.data_file_index}"
        )
    return ep_action, ep_state


def compute_nonidle_ranges(
    dataset_dir: Path,
    output: Path,
    *,
    thresholds: IdleThresholds,
    max_episodes: int | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    info = _read_info(dataset_dir)
    specs = _load_episode_specs(dataset_dir, max_episodes=max_episodes)
    if not specs:
        raise FileNotFoundError(f"No LeRobot v3 episodes found under {dataset_dir}")

    specs_by_file: dict[tuple[int, int], list[EpisodeSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_file[(spec.data_chunk_index, spec.data_file_index)].append(spec)

    episode_ranges: dict[str, list[list[int]]] = {}
    total_steps = 0
    kept_steps = 0
    total_idle_steps = 0
    processed = 0

    for chunk_file, file_specs in sorted(specs_by_file.items()):
        chunk_index, file_index = chunk_file
        data_path = _data_file_path(dataset_dir, info, chunk_index, file_index)
        indices, action, state = _load_data_file_columns(data_path)
        file_start_index = min(spec.dataset_from_index for spec in file_specs)

        for spec in sorted(file_specs, key=lambda item: item.episode_index):
            ep_action, ep_state = _select_episode_rows(
                spec,
                file_start_index=file_start_index,
                indices=indices,
                action=action,
                state=state,
            )
            idle = _idle_mask(
                ep_action,
                ep_state,
                idle_l2_threshold=thresholds.idle_l2_threshold,
                idle_arm_l2_threshold=thresholds.idle_arm_l2_threshold,
                idle_gripper_l2_threshold=thresholds.idle_gripper_l2_threshold,
            )
            ranges = _compute_keep_ranges(
                idle,
                min_idle_len=thresholds.min_idle_len,
                min_non_idle_len=thresholds.min_non_idle_len,
            )
            episode_ranges[str(spec.episode_index)] = ranges
            total_steps += int(ep_action.shape[0])
            kept_steps += sum(end - start for start, end in ranges)
            total_idle_steps += int(idle.sum())
            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                print(f"processed {processed}/{len(specs)} episodes...", flush=True)

    payload = {
        "format": "imagewam_nonidle_ranges_v1",
        "source_format": "lerobot_v3",
        "dataset_dir": str(dataset_dir),
        "thresholds": {
            "idle_l2_threshold": thresholds.idle_l2_threshold,
            "idle_arm_l2_threshold": thresholds.idle_arm_l2_threshold,
            "idle_gripper_l2_threshold": thresholds.idle_gripper_l2_threshold,
            "min_idle_len": thresholds.min_idle_len,
            "min_non_idle_len": thresholds.min_non_idle_len,
        },
        "summary": {
            "episodes": len(specs),
            "data_files": len(specs_by_file),
            "total_steps": total_steps,
            "idle_steps": total_idle_steps,
            "idle_rate": total_idle_steps / max(total_steps, 1),
            "kept_steps": kept_steps,
            "kept_rate": kept_steps / max(total_steps, 1),
        },
        "episodes": episode_ranges,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote LeRobot v3 keep-ranges JSON: {output}")
    print(
        f"episodes={len(specs)} data_files={len(specs_by_file)} total_steps={total_steps} "
        f"idle_rate={payload['summary']['idle_rate'] * 100:.2f}% "
        f"kept_rate={payload['summary']['kept_rate'] * 100:.2f}%"
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--idle-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--idle-arm-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--idle-gripper-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--min-idle-len", type=int, default=5)
    parser.add_argument("--min-non-idle-len", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compute_nonidle_ranges(
        args.dataset_dir,
        args.output,
        thresholds=IdleThresholds(
            idle_l2_threshold=args.idle_l2_threshold,
            idle_arm_l2_threshold=args.idle_arm_l2_threshold,
            idle_gripper_l2_threshold=args.idle_gripper_l2_threshold,
            min_idle_len=args.min_idle_len,
            min_non_idle_len=args.min_non_idle_len,
        ),
        max_episodes=args.max_episodes,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
