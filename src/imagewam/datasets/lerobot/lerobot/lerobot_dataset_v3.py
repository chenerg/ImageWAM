from __future__ import annotations

import logging
import json
import os
import time
import traceback
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm

from .datasets.video_utils import _PROFILE_CTX, _profile_add
from .datasets.utils import get_delta_indices
from .lerobot_dataset import HeteroLeRobotBridge, _is_bridge_enabled

logger = logging.getLogger(__name__)


def _patch_external_lerobot_tail_clamp() -> None:
    """Fallback to pyav for torchcodec EOF failures in external LeRobot v3.

    Some concatenated LeRobot videos have metadata timestamps that drift by one
    frame relative to the MP4 PTS near episode/video tails. With a relaxed
    tolerance this is harmless in the middle of the video, but at the final
    frame torchcodec can raise "Requested next frame while there are no more
    frames left to decode."  In that narrow case, retry through LeRobot's pyav
    backend, which seeks by timestamp instead of torchcodec's approximate
    frame-index path and still applies the same tolerance_s check.
    """
    try:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
        import lerobot.datasets.video_utils as video_utils
    except Exception:
        return

    if getattr(video_utils, "_imagewam_tail_clamp_patched", False):
        return

    decode_video_frames_torchcodec = getattr(video_utils, "decode_video_frames_torchcodec", None)
    decode_video_frames_torchvision = getattr(video_utils, "decode_video_frames_torchvision", None)
    get_safe_default_codec = getattr(video_utils, "get_safe_default_codec", None)
    if decode_video_frames_torchcodec is None or decode_video_frames_torchvision is None:
        return

    def _profiled_torchcodec_decode(video_path, timestamps, tolerance_s):
        from torchcodec.decoders import VideoDecoder

        profile_on = _PROFILE_CTX.get() is not None
        t0 = time.perf_counter() if profile_on else 0.0
        decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
        if profile_on:
            t1 = time.perf_counter()
            _profile_add("external.codec.open", t1 - t0)
            t0 = t1
        metadata = decoder.metadata
        average_fps = metadata.average_fps
        frame_indices = [round(ts * average_fps) for ts in timestamps]
        if profile_on and frame_indices:
            profile = _PROFILE_CTX.get()
            frame_span = float(max(frame_indices) - min(frame_indices))
            max_frame_index = float(max(frame_indices))
            _profile_add("external.codec.requested_frames", float(len(frame_indices)))
            _profile_add("external.codec.frame_span", frame_span)
            _profile_add("external.codec.max_frame_index", max_frame_index)
            if profile is not None:
                profile["external.codec.frame_span_per_call_max"] = max(
                    float(profile.get("external.codec.frame_span_per_call_max", 0.0)),
                    frame_span,
                )
                profile["external.codec.max_frame_index_per_call_max"] = max(
                    float(profile.get("external.codec.max_frame_index_per_call_max", 0.0)),
                    max_frame_index,
                )
        frames_batch = decoder.get_frames_at(indices=frame_indices)
        if profile_on:
            t1 = time.perf_counter()
            _profile_add("external.codec.decode", t1 - t0)

        loaded_frames = []
        loaded_ts = []
        for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
            loaded_frames.append(frame)
            loaded_ts.append(pts.item())

        query_ts = torch.tensor(timestamps, dtype=torch.float32)
        loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
        dist = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
        min_, argmin_ = dist.min(1)
        is_within_tol = min_ < tolerance_s
        assert is_within_tol.all(), (
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            "It means that the closest frame that can be loaded from the video is too far away in time."
            "This might be due to synchronization issues with timestamps during data collection."
            "To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts_tensor}"
            f"\nvideo: {video_path}"
        )
        closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
        return closest_frames.type(torch.float32) / 255

    def _decode_video_frames_tail_clamp(video_path, timestamps, tolerance_s, backend=None):
        profile_on = _PROFILE_CTX.get() is not None
        t0 = time.perf_counter() if profile_on else 0.0
        if backend is None:
            backend = get_safe_default_codec() if get_safe_default_codec is not None else "torchcodec"
        actual_backend = backend
        try:
            if backend == "torchcodec":
                try:
                    return _profiled_torchcodec_decode(video_path, timestamps, tolerance_s)
                except RuntimeError as exc:
                    if "no more frames left to decode" not in str(exc):
                        raise
                    actual_backend = "pyav_eof_fallback"
                    if profile_on:
                        _profile_add("codec.pyav_eof_fallbacks", 1.0)
                    return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, "pyav")
            if backend in {"pyav", "video_reader"}:
                return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
            raise ValueError(f"Unsupported video backend: {backend}")
        finally:
            if profile_on:
                elapsed = time.perf_counter() - t0
                _profile_add("external.codec.total", elapsed)
                _profile_add(f"external.codec.{actual_backend}", elapsed)
                _profile_add("external.codec.calls", 1.0)

    video_utils.decode_video_frames = _decode_video_frames_tail_clamp
    if hasattr(lerobot_dataset_module, "decode_video_frames"):
        lerobot_dataset_module.decode_video_frames = _decode_video_frames_tail_clamp
    video_utils._imagewam_tail_clamp_patched = True


class _V3DatasetEntry:
    def __init__(
        self,
        dataset: Any,
        root: Path,
        repo_id: str,
        episodes: list[int] | None,
        num_frames: int | None = None,
        episode_lengths: list[int] | None = None,
        info: dict[str, Any] | None = None,
        delta_indices: dict[str, list[int]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.root = root
        self.repo_id = repo_id
        self.episodes = episodes
        self._num_frames = int(num_frames) if num_frames is not None else None
        self.episode_lengths = episode_lengths
        self.info = info
        self.delta_indices = delta_indices

    @property
    def num_frames(self) -> int:
        if self._num_frames is not None:
            return self._num_frames
        return len(self.dataset)

    @property
    def num_episodes(self) -> int:
        if self.episode_lengths is not None:
            return len(self.episode_lengths)
        return int(getattr(self.dataset, "num_episodes", len(self.episodes or [])))


class MultiLeRobotDatasetV3(torch.utils.data.Dataset):
    """Compatibility adapter around external LeRobotDataset v3.

    The public surface intentionally mirrors the local v2-ish
    ``MultiLeRobotDataset`` enough for ``BaseLerobotDataset`` and
    ``RobotVideoDataset`` to stay backend-agnostic.
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        nonidle_filter_path: str | Path | None = None,
        hetero_bridge: dict | None = None,
        init_num_workers: int = 1,
        index_cache_path: str | Path | None = None,
        **_: Any,
    ) -> None:
        super().__init__()

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset as LeRobotDatasetV3
        except ImportError as exc:
            raise ImportError(
                "lerobot_backend='v3' requires the external `lerobot` package. "
                "Install lerobot in the active environment or use lerobot_backend='v2'."
            ) from exc
        _patch_external_lerobot_tail_clamp()

        self.dataset_dirs = [str(p) for p in dataset_dirs]
        self.ds_names = list(self.dataset_dirs)
        self.ds_roots = [Path(p) for p in self.dataset_dirs]
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(self.ds_names, 0.0001)
        self.hetero_bridge = HeteroLeRobotBridge(hetero_bridge, self.ds_names) if _is_bridge_enabled(hetero_bridge) else None
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.nonidle_filter_path = None if nonidle_filter_path is None else Path(nonidle_filter_path).expanduser()
        self._strict_nonidle = self.nonidle_filter_path is not None
        self._nonidle_filtered_indices: list[int] | None = None
        self._nonidle_keep_indices_by_episode_pos: list[list[int]] | None = None
        self._nonidle_raw_index_to_keep_rank: dict[int, int] | None = None
        self.during_training = True
        self._lerobot_cls = LeRobotDatasetV3
        self._datasets: list[_V3DatasetEntry] = []
        logger.info(
            "Initializing LeRobot v3 adapter: roots=%d init_num_workers=%s video_backend=%s",
            len(self.ds_roots),
            init_num_workers,
            video_backend or "<auto>",
        )

        def build_dataset(dataset_idx: int, ds_root: Path, ds_name: str, selected_episodes: list[int] | None):
            child_delta_timestamps = (
                self.hetero_bridge.map_delta_timestamps(delta_timestamps, dataset_idx)
                if self.hetero_bridge is not None
                else delta_timestamps
            )
            dataset = LeRobotDatasetV3(
                repo_id=ds_name,
                root=ds_root,
                episodes=selected_episodes,
                image_transforms=None if self._strict_nonidle else image_transforms,
                delta_timestamps=None if self._strict_nonidle else child_delta_timestamps,
                tolerance_s=self.tolerances_s[ds_name],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            delta_indices = (
                get_delta_indices(child_delta_timestamps, int(getattr(dataset, "fps")))
                if child_delta_timestamps is not None
                else None
            )
            return dataset, delta_indices

        def build_entry(dataset_idx: int, ds_root: Path, ds_name: str) -> tuple[int, _V3DatasetEntry]:
            try:
                selected_episodes = episodes[ds_name] if episodes else None
                dataset, delta_indices = build_dataset(dataset_idx, ds_root, ds_name, selected_episodes)
                entry = _V3DatasetEntry(
                    dataset=dataset,
                    root=ds_root,
                    repo_id=ds_name,
                    episodes=list(selected_episodes) if selected_episodes is not None else None,
                    delta_indices=delta_indices,
                )
                return dataset_idx, entry
            except Exception as exc:
                logging.error("Exception while processing LeRobot v3 root: %s", ds_root)
                traceback.print_exc()
                raise exc

        root_iter = list(zip(self.ds_roots, self.ds_names, strict=True))
        init_num_workers = max(1, int(init_num_workers))
        if init_num_workers == 1:
            entries = [
                build_entry(dataset_idx, ds_root, ds_name)
                for dataset_idx, (ds_root, ds_name) in enumerate(
                    tqdm(root_iter, desc="Initializing LeRobot v3 roots")
                )
            ]
        else:
            entries = []
            with ThreadPoolExecutor(max_workers=init_num_workers) as executor:
                futures = [
                    executor.submit(build_entry, dataset_idx, ds_root, ds_name)
                    for dataset_idx, (ds_root, ds_name) in enumerate(root_iter)
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Initializing LeRobot v3 roots ({init_num_workers} threads)",
                ):
                    entries.append(future.result())
            entries.sort(key=lambda item: item[0])

        self._datasets = [entry for _, entry in entries]

        cached_lengths = self._load_index_cache(index_cache_path, root_iter, episodes)
        if cached_lengths is not None:
            self._episode_lengths_by_dataset = cached_lengths
            logger.info(
                "Loaded LeRobot v3 post-init index cache with %d roots from %s.",
                len(self._episode_lengths_by_dataset),
                index_cache_path,
            )
        else:
            self._episode_lengths_by_dataset = [self._episode_lengths(entry) for entry in self._datasets]
        for entry, lengths in zip(self._datasets, self._episode_lengths_by_dataset, strict=True):
            entry.episode_lengths = lengths

        self._build_offsets_from_entries()
        self.episode_data_index = self._build_global_episode_data_index()
        self._load_nonidle_filter()
        self.stats = {}
        self._save_index_cache(index_cache_path)

    def _build_offsets_from_entries(self) -> None:
        self._frame_offsets = [0]
        for entry in self._datasets:
            self._frame_offsets.append(self._frame_offsets[-1] + entry.num_frames)

        self._episode_offsets = [0]
        for lengths in self._episode_lengths_by_dataset:
            self._episode_offsets.append(self._episode_offsets[-1] + len(lengths))

    def _build_global_episode_data_index(self) -> dict[str, torch.Tensor]:
        starts = []
        ends = []
        cursor = 0
        for lengths in self._episode_lengths_by_dataset:
            for length in lengths:
                starts.append(cursor)
                cursor += int(length)
                ends.append(cursor)
        return {
            "from": torch.LongTensor(starts),
            "to": torch.LongTensor(ends),
        }

    def _entry_episode_indices(self, entry: _V3DatasetEntry) -> list[int]:
        if entry.episodes is not None:
            return [int(ep_idx) for ep_idx in entry.episodes]

        meta = getattr(entry.dataset, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if episodes is not None:
            column_names = getattr(episodes, "column_names", None)
            if column_names is None:
                features = getattr(episodes, "features", None)
                column_names = list(features) if features is not None else []
            if "episode_index" in column_names:
                return self._int_column(episodes, "episode_index")

        return list(range(int(entry.num_episodes)))

    def _load_nonidle_filter(self) -> None:
        if self.nonidle_filter_path is None:
            return
        if not self.nonidle_filter_path.exists():
            raise FileNotFoundError(f"Non-idle filter JSON not found: {self.nonidle_filter_path}")

        payload = json.loads(self.nonidle_filter_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "episodes" in payload:
            episode_ranges = payload["episodes"]
        else:
            episode_ranges = payload
        if not isinstance(episode_ranges, dict):
            raise ValueError(
                f"Non-idle filter JSON must contain an episode range mapping, got {type(episode_ranges)}"
            )

        filtered_indices: list[int] = []
        keep_by_episode_pos: list[list[int]] = []
        raw_to_rank: dict[int, int] = {}
        global_episode_pos = 0
        for entry, lengths in zip(self._datasets, self._episode_lengths_by_dataset, strict=True):
            episode_indices = self._entry_episode_indices(entry)
            if len(episode_indices) != len(lengths):
                raise RuntimeError(
                    f"LeRobot v3 episode index/length mismatch for root={entry.root}: "
                    f"{len(episode_indices)} indices vs {len(lengths)} lengths"
                )

            for local_episode_pos, episode_idx in enumerate(episode_indices):
                ep_start = int(self.episode_data_index["from"][global_episode_pos].item())
                ep_end = int(self.episode_data_index["to"][global_episode_pos].item())
                ranges = episode_ranges.get(str(episode_idx), episode_ranges.get(int(episode_idx), None))
                if ranges is None:
                    keep_indices = list(range(ep_start, ep_end))
                else:
                    keep_list: list[int] = []
                    episode_len = int(lengths[local_episode_pos])
                    for raw_start, raw_end in ranges:
                        start = max(0, int(raw_start))
                        end = min(episode_len, int(raw_end))
                        if end <= start:
                            continue
                        keep_list.extend(range(ep_start + start, ep_start + end))
                    keep_indices = sorted(set(keep_list))
                keep_indices = [int(idx) for idx in keep_indices]
                keep_by_episode_pos.append(keep_indices)
                for keep_rank, raw_idx in enumerate(keep_indices):
                    raw_to_rank[int(raw_idx)] = keep_rank
                filtered_indices.extend(keep_indices)
                global_episode_pos += 1

        if len(filtered_indices) == 0:
            raise ValueError(f"Non-idle filter removed all frames: {self.nonidle_filter_path}")

        self._nonidle_filtered_indices = filtered_indices
        self._nonidle_keep_indices_by_episode_pos = keep_by_episode_pos
        self._nonidle_raw_index_to_keep_rank = raw_to_rank
        logger.info(
            "Loaded LeRobot v3 non-idle filter %s: kept %d/%d frames.",
            self.nonidle_filter_path,
            len(filtered_indices),
            self._frame_offsets[-1],
        )

    def _entry_info(self, entry: _V3DatasetEntry) -> dict[str, Any]:
        if entry.info is not None:
            return entry.info
        if entry.dataset is not None:
            meta = getattr(entry.dataset, "meta", None)
            info = getattr(meta, "info", None)
            if info is not None:
                return dict(info)
        info_path = entry.root / "meta" / "info.json"
        with info_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _root_signature(root: Path) -> dict[str, Any]:
        info_path = root / "meta" / "info.json"
        info_stat = info_path.stat()
        episodes_dir = root / "meta" / "episodes"
        episode_files = []
        if episodes_dir.exists():
            for path in sorted(episodes_dir.rglob("*.parquet")):
                stat = path.stat()
                episode_files.append(
                    {
                        "path": str(path.relative_to(root)),
                        "mtime_ns": int(stat.st_mtime_ns),
                        "size": int(stat.st_size),
                    }
                )
        return {
            "info_mtime_ns": int(info_stat.st_mtime_ns),
            "info_size": int(info_stat.st_size),
            "episodes": episode_files,
        }

    @staticmethod
    def _episodes_signature(episodes: list[int] | None) -> list[int] | None:
        return None if episodes is None else [int(ep_idx) for ep_idx in episodes]

    def _load_index_cache(
        self,
        index_cache_path: str | Path | None,
        root_iter: list[tuple[Path, str]],
        episodes: dict | None,
    ) -> list[list[int]] | None:
        if index_cache_path is None or str(index_cache_path).strip() == "":
            return None
        cache_path = Path(index_cache_path).expanduser()
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("Ignoring unreadable LeRobot v3 index cache %s: %s", cache_path, exc)
            return None
        if payload.get("version") != 1:
            return None
        records = payload.get("roots")
        if not isinstance(records, list) or len(records) != len(root_iter):
            return None

        lengths_by_dataset: list[list[int]] = []
        for record, (ds_root, ds_name) in zip(records, root_iter, strict=True):
            selected_episodes = episodes[ds_name] if episodes else None
            if record.get("root") != str(ds_root) or record.get("repo_id") != ds_name:
                return None
            if record.get("episodes") != self._episodes_signature(selected_episodes):
                return None
            if record.get("signature") != self._root_signature(ds_root):
                return None
            episode_lengths = [int(length) for length in record.get("episode_lengths", [])]
            lengths_by_dataset.append(episode_lengths)
        return lengths_by_dataset

    def _save_index_cache(self, index_cache_path: str | Path | None) -> None:
        if index_cache_path is None or str(index_cache_path).strip() == "":
            return
        cache_path = Path(index_cache_path).expanduser()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for entry in self._datasets:
            records.append(
                {
                    "root": str(entry.root),
                    "repo_id": entry.repo_id,
                    "episodes": self._episodes_signature(entry.episodes),
                    "num_frames": int(entry.num_frames),
                    "episode_lengths": [int(length) for length in (entry.episode_lengths or [])],
                    "info": self._entry_info(entry),
                    "signature": self._root_signature(entry.root),
                }
            )
        payload = {"version": 1, "roots": records}
        tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))
            os.replace(tmp_path, cache_path)
            logger.info("Saved LeRobot v3 index cache with %d roots to %s", len(records), cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _as_int(value: Any) -> int:
        if hasattr(value, "as_py"):
            value = value.as_py()
        elif hasattr(value, "item"):
            value = value.item()
        return int(value)

    @classmethod
    def _int_column(cls, episodes: Any, name: str) -> list[int]:
        column = episodes[name]
        if hasattr(column, "to_pylist"):
            values = column.to_pylist()
        else:
            values = list(column)
        return [cls._as_int(value) for value in values]

    @classmethod
    def _episode_lengths(cls, entry: _V3DatasetEntry) -> list[int]:
        dataset = entry.dataset
        meta = getattr(dataset, "meta", None)
        episodes = getattr(meta, "episodes", None)
        selected = entry.episodes
        if episodes is None:
            raise RuntimeError(f"LeRobot v3 metadata is missing `episodes` for root={entry.root}")

        column_names = getattr(episodes, "column_names", None)
        if column_names is None:
            features = getattr(episodes, "features", None)
            column_names = list(features) if features is not None else []
        required = {"dataset_from_index", "dataset_to_index"}
        if selected is not None:
            required.add("episode_index")
        missing = sorted(required.difference(column_names))
        if missing:
            raise RuntimeError(
                f"LeRobot v3 `meta.episodes` for root={entry.root} is missing required columns: {missing}"
            )

        from_indices = cls._int_column(episodes, "dataset_from_index")
        to_indices = cls._int_column(episodes, "dataset_to_index")
        if len(from_indices) != len(to_indices):
            raise RuntimeError(
                f"LeRobot v3 `meta.episodes` for root={entry.root} has mismatched from/to lengths: "
                f"{len(from_indices)} vs {len(to_indices)}"
            )
        if selected is None:
            return [end - start for start, end in zip(from_indices, to_indices, strict=True)]

        episode_indices = cls._int_column(episodes, "episode_index")
        selected_episodes = [int(ep_idx) for ep_idx in selected]
        row_by_episode = {ep_idx: row_idx for row_idx, ep_idx in enumerate(episode_indices)}
        lengths: list[int] = []
        for ep_idx in selected_episodes:
            if ep_idx not in row_by_episode:
                raise IndexError(f"Episode {ep_idx} is missing from LeRobot v3 metadata for root={entry.root}")
            row_idx = row_by_episode[ep_idx]
            lengths.append(to_indices[row_idx] - from_indices[row_idx])
        return lengths

    def set_during_training(self, during_training: bool) -> None:
        self.during_training = bool(during_training)

    @property
    def fps(self) -> int:
        return int(getattr(self._datasets[0].dataset, "fps"))

    @property
    def video(self) -> bool:
        meta = getattr(self._datasets[0].dataset, "meta", None)
        info = getattr(meta, "info", {}) or {}
        return bool(info.get("video", True))

    @property
    def num_frames(self) -> int:
        if self._nonidle_filtered_indices is not None:
            return len(self._nonidle_filtered_indices)
        return self._frame_offsets[-1]

    @property
    def num_episodes(self) -> int:
        return self._episode_offsets[-1]

    def __len__(self) -> int:
        return self.num_frames

    def _resolve_frame_index(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self._frame_offsets[-1]:
            raise IndexError(f"Index {idx} out of bounds.")
        dataset_idx = bisect_right(self._frame_offsets, idx) - 1
        local_idx = idx - self._frame_offsets[dataset_idx]
        return dataset_idx, local_idx

    def _resolve_episode_pos_for_frame(self, idx: int) -> int:
        starts = self.episode_data_index["from"]
        ends = self.episode_data_index["to"]
        episode_pos = bisect_right(starts.tolist(), int(idx)) - 1
        if episode_pos < 0 or int(idx) >= int(ends[episode_pos].item()):
            raise IndexError(f"Frame index {idx} is not contained in any episode.")
        return int(episode_pos)

    def _resolve_episode_index(self, episode_idx: int) -> tuple[int, int]:
        if episode_idx < 0 or episode_idx >= self.num_episodes:
            raise IndexError(f"Episode index {episode_idx} out of bounds.")
        dataset_idx = bisect_right(self._episode_offsets, episode_idx) - 1
        local_episode_idx = episode_idx - self._episode_offsets[dataset_idx]
        return dataset_idx, local_episode_idx

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._nonidle_filtered_indices is not None:
            return self._getitem_strict_nonidle(idx)

        profile_on = _PROFILE_CTX.get() is not None
        dataset_idx, local_idx = self._resolve_frame_index(idx)
        t0 = time.perf_counter() if profile_on else 0.0
        item = dict(self._datasets[dataset_idx].dataset[local_idx])
        if profile_on:
            t1 = time.perf_counter()
            _profile_add("v3.external_dataset_get", t1 - t0)
            t0 = t1
        item["dataset_index"] = torch.tensor(dataset_idx)
        if self.hetero_bridge is not None:
            item = self.hetero_bridge.format_item(item, dataset_idx)
            item["dataset_index"] = torch.tensor(dataset_idx)
            if profile_on:
                _profile_add("v3.hetero_bridge", time.perf_counter() - t0)
        return item

    def _getitem_strict_nonidle(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        if (
            self._nonidle_filtered_indices is None
            or self._nonidle_keep_indices_by_episode_pos is None
            or self._nonidle_raw_index_to_keep_rank is None
        ):
            raise RuntimeError("Strict non-idle tables are not initialized.")

        profile_on = _PROFILE_CTX.get() is not None
        raw_idx = int(self._nonidle_filtered_indices[idx])
        dataset_idx, local_idx = self._resolve_frame_index(raw_idx)
        entry = self._datasets[dataset_idx]
        t0 = time.perf_counter() if profile_on else 0.0

        item = self._load_anchor_item(entry, local_idx)
        if profile_on:
            t1 = time.perf_counter()
            _profile_add("v3.strict.anchor_get", t1 - t0)
            t0 = t1

        if entry.delta_indices is not None:
            episode_pos = self._resolve_episode_pos_for_frame(raw_idx)
            query_indices, padding = self._get_strict_nonidle_query_indices(
                raw_idx,
                episode_pos,
                entry.delta_indices,
            )
            item = {**item, **padding}
            query_result = self._query_strict_nonidle(entry, dataset_idx, query_indices)
            item.update(query_result)
            if profile_on:
                t1 = time.perf_counter()
                _profile_add("v3.strict.query", t1 - t0)
                t0 = t1

        self._apply_image_transforms(item, entry)
        self._ensure_task_string(item, entry)
        item["dataset_index"] = torch.tensor(dataset_idx)
        if self.hetero_bridge is not None:
            item = self.hetero_bridge.format_item(item, dataset_idx)
            item["dataset_index"] = torch.tensor(dataset_idx)
            if profile_on:
                _profile_add("v3.hetero_bridge", time.perf_counter() - t0)
        return item

    def _get_strict_nonidle_query_indices(
        self,
        raw_idx: int,
        episode_pos: int,
        delta_indices: dict[str, list[int]],
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        if self._nonidle_keep_indices_by_episode_pos is None or self._nonidle_raw_index_to_keep_rank is None:
            raise RuntimeError("Strict non-idle tables are not initialized.")
        keep_indices = self._nonidle_keep_indices_by_episode_pos[episode_pos]
        if len(keep_indices) == 0:
            raise IndexError(f"Episode position {episode_pos} has no non-idle frames.")
        if raw_idx not in self._nonidle_raw_index_to_keep_rank:
            raise IndexError(f"Raw index {raw_idx} is not in the non-idle filter.")

        keep_rank = self._nonidle_raw_index_to_keep_rank[raw_idx]
        query_indices: dict[str, list[int]] = {}
        padding: dict[str, torch.Tensor] = {}
        for key, deltas in delta_indices.items():
            cur_indices = []
            cur_padding = []
            for delta in deltas:
                target_rank = keep_rank + int(delta)
                is_pad = target_rank < 0 or target_rank >= len(keep_indices)
                clamped_rank = max(0, min(len(keep_indices) - 1, target_rank))
                cur_indices.append(int(keep_indices[clamped_rank]))
                cur_padding.append(bool(is_pad))
            query_indices[key] = cur_indices
            padding[f"{key}_is_pad"] = torch.BoolTensor(cur_padding)
        return query_indices, padding

    def _load_anchor_item(self, entry: _V3DatasetEntry, local_idx: int) -> dict[str, Any]:
        if self.during_training:
            return dict(entry.dataset[local_idx])
        return dict(self._get_raw_item(entry, local_idx))

    def _query_strict_nonidle(
        self,
        entry: _V3DatasetEntry,
        dataset_idx: int,
        query_indices: dict[str, list[int]],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        visual_keys = self._camera_keys(entry)
        depth_keys = self._depth_keys(entry)
        raw_cache: dict[int, dict[str, Any]] = {}
        full_cache: dict[int, dict[str, Any]] = {}

        for key, global_indices in query_indices.items():
            if key in visual_keys and not self.during_training:
                continue
            values = []
            for global_idx in global_indices:
                q_dataset_idx, q_local_idx = self._resolve_frame_index(int(global_idx))
                if q_dataset_idx != dataset_idx:
                    raise RuntimeError(
                        "Strict non-idle query crossed dataset boundaries, which should be impossible "
                        f"within one episode: anchor_dataset={dataset_idx}, query_dataset={q_dataset_idx}"
                    )
                if key in visual_keys:
                    sample = full_cache.get(q_local_idx)
                    if sample is None:
                        sample = dict(entry.dataset[q_local_idx])
                        full_cache[q_local_idx] = sample
                else:
                    sample = raw_cache.get(q_local_idx)
                    if sample is None:
                        sample = dict(self._get_raw_item(entry, q_local_idx))
                        raw_cache[q_local_idx] = sample
                values.append(self._as_tensor_for_stack(sample[key]))

            if len(values) == 0:
                continue
            stacked = torch.stack(values, dim=0)
            result[key] = stacked
        return result

    def _apply_image_transforms(self, item: dict[str, Any], entry: _V3DatasetEntry) -> None:
        if self.image_transforms is None:
            return
        for key in self._camera_keys(entry).difference(self._depth_keys(entry)):
            if key in item:
                item[key] = self.image_transforms(item[key])

    @staticmethod
    def _as_tensor_for_stack(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    @staticmethod
    def _get_raw_item(entry: _V3DatasetEntry, local_idx: int) -> dict[str, Any]:
        if hasattr(entry.dataset, "get_raw_item"):
            return entry.dataset.get_raw_item(local_idx)
        return entry.dataset[local_idx]

    @staticmethod
    def _camera_keys(entry: _V3DatasetEntry) -> set[str]:
        meta = getattr(entry.dataset, "meta", None)
        camera_keys = getattr(meta, "camera_keys", None)
        if camera_keys is not None:
            return set(camera_keys)
        features = getattr(meta, "features", {}) if meta is not None else {}
        return {
            key
            for key, feature in features.items()
            if isinstance(feature, dict) and feature.get("dtype") in {"image", "video"}
        }

    @staticmethod
    def _depth_keys(entry: _V3DatasetEntry) -> set[str]:
        meta = getattr(entry.dataset, "meta", None)
        depth_keys = getattr(meta, "depth_keys", None)
        return set(depth_keys or [])

    @staticmethod
    def _scalar_to_int(value: Any) -> int:
        if hasattr(value, "item"):
            value = value.item()
        return int(value)

    def _ensure_task_string(self, item: dict[str, Any], entry: _V3DatasetEntry) -> None:
        if "task" in item or "task_index" not in item:
            return
        meta = getattr(entry.dataset, "meta", None)
        tasks = getattr(meta, "tasks", None)
        if tasks is None:
            return
        task_idx = self._scalar_to_int(item["task_index"])
        if hasattr(tasks, "iloc"):
            item["task"] = tasks.iloc[task_idx].name
        elif isinstance(tasks, dict):
            item["task"] = tasks[task_idx]
        else:
            item["task"] = tasks[task_idx]

    def get_episode_data(self, episode_idx: int) -> dict[str, torch.Tensor]:
        dataset_idx, local_episode_idx = self._resolve_episode_index(episode_idx)
        entry = self._datasets[dataset_idx]
        selected_episode = (
            entry.episodes[local_episode_idx]
            if entry.episodes is not None
            else local_episode_idx
        )
        plain_dataset = self._lerobot_cls(
            repo_id=entry.repo_id,
            root=entry.root,
            episodes=[int(selected_episode)],
            delta_timestamps=None,
            download_videos=False,
        )

        stacked: dict[str, list[Any]] = {}
        passthrough: dict[str, Any] = {}
        for frame_idx in range(len(plain_dataset)):
            sample = plain_dataset[frame_idx]
            for key, value in sample.items():
                if key.startswith("images."):
                    continue
                if isinstance(value, torch.Tensor):
                    stacked.setdefault(key, []).append(value)
                elif key not in passthrough:
                    passthrough[key] = value

        result: dict[str, Any] = {}
        for key, values in stacked.items():
            result[key] = torch.stack(
                [value if value.ndim > 0 else value.unsqueeze(0) for value in values],
                dim=0,
            )
            if result[key].ndim == 2 and result[key].shape[-1] == 1:
                result[key] = result[key].squeeze(-1)
        result.update(passthrough)

        if self.hetero_bridge is not None:
            result = self.hetero_bridge.format_item(result, dataset_idx)
        return result
