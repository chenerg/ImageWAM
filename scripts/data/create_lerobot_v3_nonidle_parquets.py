#!/usr/bin/env python3
"""Create a standalone filtered LeRobot v3 dataset containing non-idle frames.

The source dataset is not modified. The output dataset contains filtered
``data`` and ``meta/episodes`` parquet files, copies all other metadata, and
uses symlinks to reuse the source ``videos`` and ``images`` directories.

Output data parquet files rebuild ``index`` globally from zero while preserving
the source ``frame_index`` and ``timestamp``. Episode metadata preserves its
original row/file partitioning and updates ``length``, ``dataset_from_index``,
and ``dataset_to_index`` to match the filtered rows.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
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


@dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    dataset_from_index: int
    dataset_to_index: int
    data_chunk_index: int
    data_file_index: int
    meta_path: Path

    @property
    def length(self) -> int:
        return self.dataset_to_index - self.dataset_from_index

    @property
    def data_file_key(self) -> tuple[int, int]:
        return self.data_chunk_index, self.data_file_index


@dataclass(frozen=True)
class EpisodePlan:
    spec: EpisodeSpec
    keep_ranges: list[list[int]]
    keep_positions: np.ndarray

    @property
    def kept_frames(self) -> int:
        return int(self.keep_positions.shape[0])


def _load_episode_specs(dataset_dir: Path) -> list[EpisodeSpec]:
    import pyarrow.parquet as pq

    episode_paths = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No LeRobot v3 episode metadata parquet files found under {dataset_dir}")

    specs: list[EpisodeSpec] = []
    for meta_path in episode_paths:
        for row in pq.read_table(meta_path).to_pylist():
            specs.append(
                EpisodeSpec(
                    episode_index=_to_int(row["episode_index"]),
                    dataset_from_index=_to_int(row["dataset_from_index"]),
                    dataset_to_index=_to_int(row["dataset_to_index"]),
                    data_chunk_index=_to_int(row["data/chunk_index"]),
                    data_file_index=_to_int(row["data/file_index"]),
                    meta_path=meta_path,
                )
            )
    return sorted(specs, key=lambda item: item.episode_index)


def _episode_row_positions(
    spec: EpisodeSpec,
    *,
    file_start_index: int,
    indices: np.ndarray | None,
    row_count: int,
) -> np.ndarray:
    if indices is None:
        start = spec.dataset_from_index - file_start_index
        end = spec.dataset_to_index - file_start_index
        if start < 0 or end > row_count or start > end:
            raise ValueError(
                f"Episode {spec.episode_index} range [{spec.dataset_from_index}, {spec.dataset_to_index}) "
                f"is outside its data file rows."
            )
        positions = np.arange(start, end, dtype=np.int64)
    else:
        positions = np.flatnonzero(
            (indices >= spec.dataset_from_index) & (indices < spec.dataset_to_index)
        ).astype(np.int64, copy=False)

    if int(positions.shape[0]) != spec.length:
        raise ValueError(
            f"Episode {spec.episode_index} expected {spec.length} rows, found {positions.shape[0]} "
            f"for data file chunk={spec.data_chunk_index} file={spec.data_file_index}"
        )
    return positions


def _positions_from_ranges(episode_positions: np.ndarray, keep_ranges: list[list[int]]) -> np.ndarray:
    if not keep_ranges:
        return np.empty((0,), dtype=np.int64)
    chunks = [episode_positions[start:end] for start, end in keep_ranges]
    return np.concatenate(chunks).astype(np.int64, copy=False)


def _compute_episode_plans(
    dataset_dir: Path,
    info: dict[str, Any],
    specs: list[EpisodeSpec],
    thresholds: IdleThresholds,
    *,
    progress_every: int,
) -> dict[int, EpisodePlan]:
    specs_by_file: dict[tuple[int, int], list[EpisodeSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_file[spec.data_file_key].append(spec)

    plans: dict[int, EpisodePlan] = {}
    processed = 0
    for chunk_index, file_index in sorted(specs_by_file):
        data_path = _data_file_path(dataset_dir, info, chunk_index, file_index)
        indices, action, state = _load_data_file_columns(data_path)
        file_specs = sorted(specs_by_file[(chunk_index, file_index)], key=lambda item: item.episode_index)
        file_start_index = min(spec.dataset_from_index for spec in file_specs)

        for spec in file_specs:
            positions = _episode_row_positions(
                spec,
                file_start_index=file_start_index,
                indices=indices,
                row_count=int(action.shape[0]),
            )
            ep_action = action[positions]
            ep_state = state[positions]
            idle = _idle_mask(
                ep_action,
                ep_state,
                idle_l2_threshold=thresholds.idle_l2_threshold,
                idle_arm_l2_threshold=thresholds.idle_arm_l2_threshold,
                idle_gripper_l2_threshold=thresholds.idle_gripper_l2_threshold,
            )
            keep_ranges = _compute_keep_ranges(
                idle,
                min_idle_len=thresholds.min_idle_len,
                min_non_idle_len=thresholds.min_non_idle_len,
            )
            plans[spec.episode_index] = EpisodePlan(
                spec=spec,
                keep_ranges=keep_ranges,
                keep_positions=_positions_from_ranges(positions, keep_ranges),
            )
            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                print(f"computed non-idle ranges for {processed}/{len(specs)} episodes...", flush=True)
    return plans


def _validate_dataset_paths(source_dataset_dir: Path, output_dataset_dir: Path) -> None:
    if not source_dataset_dir.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source_dataset_dir}")
    if (
        source_dataset_dir == output_dataset_dir
        or source_dataset_dir in output_dataset_dir.parents
        or output_dataset_dir in source_dataset_dir.parents
    ):
        raise ValueError(
            "Source and output dataset directories must not be identical or nested: "
            f"source={source_dataset_dir}, output={output_dataset_dir}"
        )
    if output_dataset_dir.exists() or output_dataset_dir.is_symlink():
        raise FileExistsError(f"Output dataset path already exists: {output_dataset_dir}")


def _copy_dataset_scaffold(source_dataset_dir: Path, staging_dir: Path) -> None:
    """Copy everything except data/episodes/media into the staging dataset."""

    def ignore(path: str, names: list[str]) -> set[str]:
        relative_path = Path(path).resolve().relative_to(source_dataset_dir)
        if relative_path == Path("."):
            return {name for name in ("data", "videos", "images") if name in names}
        if relative_path == Path("meta") and "episodes" in names:
            return {"episodes"}
        return set()

    shutil.copytree(
        source_dataset_dir,
        staging_dir,
        dirs_exist_ok=True,
        symlinks=True,
        ignore=ignore,
    )


def _link_media_directories(source_dataset_dir: Path, staging_dir: Path) -> None:
    for name in ("videos", "images"):
        source_path = source_dataset_dir / name
        if source_path.is_dir():
            (staging_dir / name).symlink_to(source_path.resolve(), target_is_directory=True)


def _update_output_info(staging_dir: Path, kept_frames: int) -> None:
    info_path = staging_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Copied LeRobot info.json not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_frames"] = kept_frames
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative_data_output_path(dataset_dir: Path, data_path: Path, data_output_dir: Path) -> Path:
    try:
        rel = data_path.relative_to(dataset_dir / "data")
    except ValueError as exc:
        raise ValueError(f"Data path is not under {dataset_dir / 'data'}: {data_path}") from exc
    return data_output_dir / rel


def _replace_int_column(table: Any, name: str, values: list[int] | range) -> Any:
    import pyarrow as pa

    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise KeyError(f"Required column {name!r} not found in parquet schema: {table.schema.names}")
    field = table.schema.field(column_index)
    array = pa.array(list(values), type=field.type)
    return table.set_column(column_index, field, array)


def _write_data_nonidle(
    dataset_dir: Path,
    info: dict[str, Any],
    specs: list[EpisodeSpec],
    plans: dict[int, EpisodePlan],
    *,
    data_output_dir: Path,
) -> dict[int, list[int]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    specs_by_file: dict[tuple[int, int], list[EpisodeSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_file[spec.data_file_key].append(spec)

    output_indices_by_episode: dict[int, list[int]] = {spec.episode_index: [] for spec in specs}
    next_output_index = 0

    for chunk_index, file_index in sorted(specs_by_file):
        data_path = _data_file_path(dataset_dir, info, chunk_index, file_index)
        table = pq.read_table(data_path)
        position_to_episode: dict[int, int] = {}
        keep_positions: list[int] = []
        for spec in specs_by_file[(chunk_index, file_index)]:
            for pos in plans[spec.episode_index].keep_positions.tolist():
                pos_i = int(pos)
                if pos_i in position_to_episode:
                    raise ValueError(f"Duplicate kept row position {pos_i} in {data_path}")
                position_to_episode[pos_i] = spec.episode_index
                keep_positions.append(pos_i)

        keep_positions = sorted(keep_positions)
        out_table = table.take(pa.array(keep_positions, type=pa.int64()))
        new_indices = range(next_output_index, next_output_index + len(keep_positions))
        out_table = _replace_int_column(out_table, "index", new_indices)

        for new_index, source_pos in enumerate(keep_positions, start=next_output_index):
            output_indices_by_episode[position_to_episode[source_pos]].append(new_index)
        next_output_index += len(keep_positions)

        out_path = _relative_data_output_path(dataset_dir, data_path, data_output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out_table, out_path)

    return output_indices_by_episode


def _episode_intervals_from_output_indices(
    specs: list[EpisodeSpec],
    output_indices_by_episode: dict[int, list[int]],
) -> dict[int, tuple[int, int, int]]:
    intervals: dict[int, tuple[int, int, int]] = {}
    expected_next = 0
    for spec in sorted(specs, key=lambda item: item.episode_index):
        indices = output_indices_by_episode.get(spec.episode_index, [])
        if not indices:
            intervals[spec.episode_index] = (0, expected_next, expected_next)
            continue
        expected = list(range(expected_next, expected_next + len(indices)))
        if indices != expected:
            raise ValueError(
                f"Filtered rows for episode {spec.episode_index} are not contiguous in output index order: "
                f"expected {expected[:3]}... got {indices[:3]}..."
            )
        start = expected_next
        stop = expected_next + len(indices)
        intervals[spec.episode_index] = (len(indices), start, stop)
        expected_next = stop
    return intervals


def _write_meta_nonidle(
    dataset_dir: Path,
    specs: list[EpisodeSpec],
    intervals: dict[int, tuple[int, int, int]],
    *,
    meta_output_dir: Path,
) -> None:
    import pyarrow.parquet as pq

    meta_paths = sorted({spec.meta_path for spec in specs})
    for meta_path in meta_paths:
        table = pq.read_table(meta_path)
        rows = table.to_pylist()
        lengths: list[int] = []
        from_indices: list[int] = []
        to_indices: list[int] = []
        for row in rows:
            episode_index = _to_int(row["episode_index"])
            length, start, stop = intervals[episode_index]
            lengths.append(length)
            from_indices.append(start)
            to_indices.append(stop)

        table = _replace_int_column(table, "length", lengths)
        table = _replace_int_column(table, "dataset_from_index", from_indices)
        table = _replace_int_column(table, "dataset_to_index", to_indices)

        rel = meta_path.relative_to(dataset_dir / "meta")
        out_path = meta_output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)


def create_nonidle_parquets(
    source_dataset_dir: Path,
    output_dataset_dir: Path,
    *,
    thresholds: IdleThresholds,
    progress_every: int = 100,
) -> dict[str, Any]:
    source_dataset_dir = source_dataset_dir.expanduser().resolve()
    requested_output_path = output_dataset_dir.expanduser().absolute()
    output_dataset_dir = requested_output_path.resolve()
    _validate_dataset_paths(source_dataset_dir, output_dataset_dir)
    if requested_output_path.is_symlink():
        raise FileExistsError(f"Output dataset path already exists: {requested_output_path}")

    info = _read_info(source_dataset_dir)
    specs = _load_episode_specs(source_dataset_dir)
    if not specs:
        raise FileNotFoundError(f"No LeRobot v3 episodes found under {source_dataset_dir}")

    plans = _compute_episode_plans(
        source_dataset_dir,
        info,
        specs,
        thresholds,
        progress_every=progress_every,
    )

    total_frames = sum(spec.length for spec in specs)
    kept_frames = sum(plan.kept_frames for plan in plans.values())
    output_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dataset_dir.name}.tmp-",
            dir=output_dataset_dir.parent,
        )
    )
    try:
        _copy_dataset_scaffold(source_dataset_dir, staging_dir)
        _link_media_directories(source_dataset_dir, staging_dir)
        output_indices_by_episode = _write_data_nonidle(
            source_dataset_dir,
            info,
            specs,
            plans,
            data_output_dir=staging_dir / "data",
        )
        intervals = _episode_intervals_from_output_indices(specs, output_indices_by_episode)
        _write_meta_nonidle(
            source_dataset_dir,
            specs,
            intervals,
            meta_output_dir=staging_dir / "meta",
        )
        _update_output_info(staging_dir, kept_frames)

        report = {
            "format": "imagewam_lerobot_v3_nonidle_dataset_v2",
            "source_dataset_dir": str(source_dataset_dir),
            "output_dataset_dir": str(output_dataset_dir),
            "thresholds": {
                "idle_l2_threshold": thresholds.idle_l2_threshold,
                "idle_arm_l2_threshold": thresholds.idle_arm_l2_threshold,
                "idle_gripper_l2_threshold": thresholds.idle_gripper_l2_threshold,
                "min_idle_len": thresholds.min_idle_len,
                "min_non_idle_len": thresholds.min_non_idle_len,
            },
            "summary": {
                "episodes": len(specs),
                "data_files": len({spec.data_file_key for spec in specs}),
                "total_frames": total_frames,
                "kept_frames": kept_frames,
                "removed_frames": total_frames - kept_frames,
                "kept_rate": kept_frames / max(total_frames, 1),
            },
            "episodes": [
                {
                    "episode_index": spec.episode_index,
                    "source_length": spec.length,
                    "kept_frames": plans[spec.episode_index].kept_frames,
                    "keep_ranges": plans[spec.episode_index].keep_ranges,
                    "dataset_from_index": intervals[spec.episode_index][1],
                    "dataset_to_index": intervals[spec.episode_index][2],
                }
                for spec in specs
            ],
        }
        report_path = staging_dir / "nonidle_parquet_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staging_dir.replace(output_dataset_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(f"Wrote standalone non-idle dataset: {output_dataset_dir}")
    print(f"Wrote report: {output_dataset_dir / 'nonidle_parquet_report.json'}")
    print(
        f"episodes={len(specs)} total_frames={total_frames} kept_frames={kept_frames} "
        f"removed_frames={total_frames - kept_frames} kept_rate={report['summary']['kept_rate'] * 100:.2f}%"
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dataset_dir", type=Path)
    parser.add_argument("output_dataset_dir", type=Path)
    parser.add_argument("--idle-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--idle-arm-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--idle-gripper-l2-threshold", type=float, default=1e-3)
    parser.add_argument("--min-idle-len", type=int, default=5)
    parser.add_argument("--min-non-idle-len", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    create_nonidle_parquets(
        args.source_dataset_dir,
        args.output_dataset_dir,
        thresholds=IdleThresholds(
            idle_l2_threshold=args.idle_l2_threshold,
            idle_arm_l2_threshold=args.idle_arm_l2_threshold,
            idle_gripper_l2_threshold=args.idle_gripper_l2_threshold,
            min_idle_len=args.min_idle_len,
            min_non_idle_len=args.min_non_idle_len,
        ),
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
