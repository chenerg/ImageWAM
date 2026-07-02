# LeRobot v2/v3 Dataset 与 Non-idle 过滤实现差异

本文总结 ImageWAM 中 LeRobot v2 backend、LeRobot v3 backend 的实现差异，以及两者当前 non-idle/no-op 过滤逻辑的差异。

相关代码入口：

- backend 选择与通用 dataset 包装：`src/imagewam/datasets/lerobot/base_lerobot_dataset.py`
- RoboTwin 图像/动作样本包装：`src/imagewam/datasets/lerobot/robot_video_dataset.py`
- 本地 LeRobot v2 实现：`src/imagewam/datasets/lerobot/lerobot/lerobot_dataset.py`
- 外部 LeRobot v3 适配层：`src/imagewam/datasets/lerobot/lerobot/lerobot_dataset_v3.py`
- RoboTwin non-idle JSON 预计算：`scripts/data/compute_robotwin_nonidle_ranges.py`
- FLUX.2 4B + RoboTwin v3 专用训练入口：`scripts/flux2/run_train_flux2_4b_robotwin_v3.sh`

## 1. 背景

ImageWAM 的训练 dataset 入口不是直接暴露原始 LeRobot dataset，而是经过两层包装：

```text
RobotVideoDataset
  └── BaseLerobotDataset
        ├── MultiLeRobotDataset    # lerobot_backend="v2"
        └── MultiLeRobotDatasetV3  # lerobot_backend="v3"
```

`RobotVideoDataset` 负责把 LeRobot 样本整理成 ImageWAM 训练需要的格式，例如多相机拼接、图像增强、action/state processor、Qwen 文本缓存读取等。

`BaseLerobotDataset` 负责根据配置生成 `delta_timestamps`、episode split、backend 参数，并实例化 v2 或 v3 的 multi-root dataset。

`MultiLeRobotDataset` 和 `MultiLeRobotDatasetV3` 负责真正访问底层 LeRobot 数据。两者暴露出相似接口，但内部数据访问方式不同。

## 2. LeRobot v2 backend 实现

v2 backend 使用仓库内的本地实现：

```text
MultiLeRobotDataset
  └── LeRobotDataset
        ├── LeRobotDatasetMetadata
        ├── HuggingFace datasets parquet loader
        └── ImageWAM 自定义视频/索引/过滤逻辑
```

### 2.1 数据加载

v2 的 `LeRobotDataset` 会读取本地 `meta/` 信息，然后通过 HuggingFace `load_dataset("parquet", data_files=...)` 加载每个 episode 的 parquet 文件。

主要状态包括：

- `self.hf_dataset`：parquet 行数据，包含 action、state、timestamp、episode_index 等非视频字段。
- `self.meta`：LeRobot metadata，包括 fps、episode 信息、video keys、feature schema。
- `self.episode_data_index`：每个 episode 在全局帧序列中的 `[from, to)` 范围。
- `self.delta_indices`：由 `delta_timestamps` 按 fps 转换得到的整数帧偏移。

### 2.2 上下文帧查询

v2 自己控制窗口采样逻辑。`__getitem__()` 会：

1. 根据 dataloader index 得到原始帧 index。
2. 找到该帧所属 episode。
3. 调用 `_get_query_indices()` 根据 `delta_indices` 计算图像、state、action 的上下文帧 index。
4. 对越出 episode 边界的 index 做 clamp，并生成 padding mask。
5. 读取 parquet 字段和视频帧。

因为 query index 由 ImageWAM 本地代码生成，所以 v2 可以在这里深度介入采样时间线。

## 3. LeRobot v3 backend 实现

v3 backend 是对外部 `lerobot` 包中 `LeRobotDataset` 的兼容适配：

```text
MultiLeRobotDatasetV3
  └── external lerobot.datasets.lerobot_dataset.LeRobotDataset
```

### 3.1 数据加载

v3 不再直接使用本仓库里的 v2 `LeRobotDataset` 读取 parquet。它为每个 root 构造一个外部 LeRobot v3 dataset：

```python
LeRobotDatasetV3(
    repo_id=ds_name,
    root=ds_root,
    episodes=selected_episodes,
    image_transforms=image_transforms,
    delta_timestamps=child_delta_timestamps,
    tolerance_s=...,
    download_videos=...,
    video_backend=...,
)
```

ImageWAM 的 v3 适配层主要补齐这些能力：

- 多 root 合并。
- episode length 与全局 frame offset 计算。
- `episode_data_index` 重建。
- v3 初始化 index cache。
- `dataset_index` 注入。
- heterogeneous dataset bridge。
- torchcodec 末帧 EOF fallback patch。

### 3.2 上下文帧查询

v3 有两条路径：

1. 未启用 `nonidle_filter_path` 时，ImageWAM 把 `delta_timestamps` 直接传给外部 LeRobot v3 dataset。适配层的 `__getitem__()` 只负责把全局 index 映射成某个 root 内的 local index、调用 `dataset[local_idx]`、补充 `dataset_index` 和 hetero bridge 格式化。
2. 启用 `nonidle_filter_path` 时，ImageWAM 进入 strict non-idle 路径。适配层构造外部 dataset 时会传入 `delta_timestamps=None` 和 `image_transforms=None`，外部 dataset 只负责读取单帧 anchor 或单帧 query；多帧上下文窗口由 `MultiLeRobotDatasetV3` 自己按过滤后的时间线重建。

因此，v3 当前的普通读取路径仍复用外部 LeRobot v3；但 non-idle 场景下已经不是旧的 anchor-only 过滤，而是在适配层里实现了类似 v2 的 filtered timeline query。

## 4. Non-idle JSON 的含义

RoboTwin 中会存在大量 no-op/idle 帧。ImageWAM 使用预计算 JSON 记录每个 episode 中应该保留的 non-idle 区间：

```bash
bash scripts/data/precompute_noops_lerobot.sh
```

默认输出：

```text
${ROBOTWIN_ROOT}/nonidle_ranges.json
```

JSON 中的关键结构是：

```json
{
  "format": "imagewam_nonidle_ranges_v1",
  "episodes": {
    "0": [[start0, end0], [start1, end1]],
    "1": [[start0, end0]]
  }
}
```

区间是 episode 内的局部帧范围，采用 `[start, end)` 语义。

生成逻辑在 `scripts/data/compute_robotwin_nonidle_ranges.py` 中：

- 读取 episode parquet。
- 取 `action` 与 `observation.state`。
- 计算 `delta = action - state`。
- 根据整体 L2、arm L2、gripper L2 阈值判断 idle。
- 删除长度达到 `min_idle_len` 的 idle 段。
- 保留连续 non-idle range。
- episode 末尾 idle/no-op 段会保留，因为它们可能包含释放、稳定、收尾等成功轨迹信息。

这个 JSON 不会改写原始数据集，只作为 dataloader 的过滤索引。

## 5. v2 non-idle 实现

v2 是完整的“过滤后时间线”实现。它不是在 sampler 层简单跳过若干 dataloader index，而是在 `LeRobotDataset` 内部把“过滤后的样本序号”映射回原始 parquet/video 时间线，并且让图像、state、action 的上下文窗口也沿过滤后的时间线移动。

整体调用链是：

```text
BaseLerobotDataset(nonidle_filter_path=...)
  └── MultiLeRobotDataset(nonidle_filter_path=...)
        └── LeRobotDataset(nonidle_filter_path=...)
              ├── _load_nonidle_filter()
              ├── __len__()
              ├── __getitem__()
              └── _get_query_indices()
```

`BaseLerobotDataset` 只负责把配置里的 `nonidle_filter_path` 传下去；真正的过滤逻辑在 v2 的 `LeRobotDataset` 里完成。

### 5.1 构建索引表

`LeRobotDataset._load_nonidle_filter()` 会读取 `nonidle_ranges.json`，并构建三类索引：

```python
self._nonidle_filtered_indices
self._nonidle_keep_indices_by_episode_pos
self._nonidle_raw_index_to_keep_rank
```

含义：

- `_nonidle_filtered_indices`：过滤后的全局采样 index 列表。`__len__()` 返回它的长度，`__getitem__()` 用它把 dataloader index 映射回原始帧。
- `_nonidle_keep_indices_by_episode_pos`：每个 episode 保留下来的原始帧 index 列表。
- `_nonidle_raw_index_to_keep_rank`：原始帧 index 到过滤后 episode 内 rank 的映射。

构建过程更具体地说：

1. 读取 JSON；如果顶层有 `episodes` 字段，就使用 `payload["episodes"]`，否则把整个 payload 当作 episode range mapping。
2. 遍历当前 dataset 选中的 episode。这里的顺序来自 `_selected_episode_indices`，如果训练配置做了 episode split 或 `episode_index_filter`，只会遍历被选中的 episode。
3. 通过 `episode_data_index["from"]` / `episode_data_index["to"]` 找到该 episode 在 `hf_dataset` 中的全局帧范围 `[ep_start, ep_end)`。
4. 如果 JSON 里没有这个 episode 的 ranges，则保留整个 episode：`range(ep_start, ep_end)`。这意味着缺省行为是“不滤掉这个 episode”。
5. 如果 JSON 里有 ranges，则把每个 episode 内局部 `[raw_start, raw_end)` 转成全局帧 index：

```python
start = max(0, int(raw_start))
end = min(ep_end - ep_start, int(raw_end))
keep_indices.extend(range(ep_start + start, ep_start + end))
```

6. 对 `keep_indices` 做 `sorted(set(...))`，去重并保持时间顺序。
7. 把每个保留帧写入三张表：

```text
filtered_indices.extend(keep_indices)
keep_by_episode_pos[episode_pos] = keep_indices
raw_to_rank[raw_idx] = keep_rank
```

如果最终 `filtered_indices` 为空，会直接报错，避免训练时拿到一个长度为 0 的 dataset。

注意：v2 的 JSON key 用 episode index 查找，代码同时支持字符串 key 和整数 key：

```python
ranges = episode_ranges.get(str(episode_idx), episode_ranges.get(int(episode_idx), None))
```

### 5.2 主帧过滤

当 dataloader 请求第 `idx` 个样本时，v2 会先做映射：

```python
raw_idx = self._nonidle_filtered_indices[idx]
```

所以训练样本的主帧不会落在被过滤掉的 no-op 区间。

这一点也会影响 dataset 长度：

```python
def num_frames(self):
    if self._nonidle_filtered_indices is not None:
        return len(self._nonidle_filtered_indices)
    return len(self.hf_dataset)
```

也就是说，DataLoader 看到的是过滤后的长度；`idx` 是过滤后时间线上的序号，`raw_idx` 才是原始 parquet/video 时间线上的帧 index。

### 5.3 上下文帧也沿过滤后时间线采样

v2 的关键点是 `_get_query_indices()` 对 non-idle 有专门分支：

```text
raw_idx -> keep_rank -> keep_rank + delta -> keep_indices[clamped_rank]
```

也就是说，图像、state、action 的上下文窗口不是在原始 episode 时间线上做 `raw_idx + delta`，而是在过滤后的 non-idle 序列上做 `keep_rank + delta`。

示例：

```text
原始时间线: A B idle idle C D
过滤后:     A B C D
```

如果当前主帧是 `C`，并且需要前一帧：

- v2 会取 `B`。
- 不会取 `C` 前面原始时间线里的 idle 帧。

因此 v2 的过滤更彻底：主帧和上下文帧都基于 non-idle 后的时间线。

### 5.4 padding 与边界 clamp

上下文窗口可能越过过滤后 episode 的开头或结尾。v2 不会跨 episode 取帧，而是 clamp 到当前 episode 的第一个或最后一个保留帧，并生成 padding mask：

```text
target_rank = keep_rank + delta
is_pad = target_rank < 0 or target_rank >= len(keep_indices)
clamped_rank = clamp(target_rank, 0, len(keep_indices) - 1)
query_idx = keep_indices[clamped_rank]
```

对应输出里会多出类似这些 key：

```text
observation.images.xxx_is_pad
observation.state.xxx_is_pad
action.xxx_is_pad
```

这和未启用 non-idle 时的逻辑一致：越界位置会被 clamp，同时用 `*_is_pad` 告诉后续模型/processor 哪些时间步是 padding。

### 5.5 parquet 字段和视频字段如何读取

v2 的 anchor row 直接来自 `hf_dataset[raw_idx]`。如果配置了 `delta_timestamps`，`_get_query_indices()` 会为每个字段生成一组 query indices：

- 非视频字段，例如 `observation.state`、`action`，通过 `hf_dataset.select(q_idx)` 读取，并 `torch.stack` 成时间维。
- 视频字段先根据 query indices 计算 query timestamps，再调用 `decode_video_frames(video_path, query_ts, tolerance_s, video_backend)` 解码对应帧。

视频 timestamp 有一个快路径：对于均匀 fps 数据，它不再额外从 parquet 读 `timestamp` 列，而是用 anchor timestamp 和帧 index 差值计算：

```text
query_ts = current_ts + (query_idx - raw_idx) / fps
```

这样 non-idle 后的上下文 query index 仍然能落回原始视频时间线上正确的位置，同时避免每个样本反复读 parquet timestamp 列。

### 5.6 多 root 下的行为

`MultiLeRobotDataset` 会为每个 root 构造一个 v2 `LeRobotDataset`，每个子 dataset 独立读取同一个 `nonidle_filter_path` 并建立自己的过滤表。随后 multi dataset 把多个子 dataset 按顺序拼接：

```text
global idx -> child dataset idx -> child local idx -> child LeRobotDataset.__getitem__()
```

所以 v2 的 non-idle 过滤发生在每个子 dataset 内部。`MultiLeRobotDataset.num_frames` 是各子 dataset 过滤后长度之和，`dataset_index` 在外层注入。

### 5.7 不会改写原始数据

v2 non-idle 只改变 dataloader 看到的索引空间，不会删除 parquet 行、不会裁剪 mp4，也不会修改 `meta/`。同一个数据 root 可以带 filter 训练，也可以不带 filter 完整读取。

## 6. v3 non-idle 实现

v3 当前实现的是 strict non-idle filtered timeline。它仍然复用外部 LeRobot v3 dataset 负责底层单帧 parquet/video 读取，但上下文窗口不再交给外部 v3 的 `delta_timestamps` 逻辑，而是在 ImageWAM 的 `MultiLeRobotDatasetV3` 适配层中重建。

### 6.1 构建过滤列表

`MultiLeRobotDatasetV3._load_nonidle_filter()` 会读取同一个 `nonidle_ranges.json`，为所有 root/episode 构建三张表：

```python
self._nonidle_filtered_indices
self._nonidle_keep_indices_by_episode_pos
self._nonidle_raw_index_to_keep_rank
```

含义和 v2 一致。区别是 v3 没有直接使用 v2 的 `get_episode_data_index()`；它先从外部 v3 metadata 中读取每个 episode 的 `dataset_from_index` / `dataset_to_index`，得到 episode length，再由适配层重建跨 root 的全局 `episode_data_index`。

构建过程：

1. 为每个 root 初始化一个外部 `LeRobotDatasetV3`。
2. 从 `meta.episodes` 中读取 episode index 和 episode length。
3. 根据 `_episode_lengths_by_dataset` 构建全局 frame offset 和全局 `episode_data_index`。
4. 读取 `nonidle_ranges.json`，把 episode 内局部 `[start, end)` 转成全局 frame index。
5. 写入 `_nonidle_filtered_indices`、`_nonidle_keep_indices_by_episode_pos`、`_nonidle_raw_index_to_keep_rank`。

v3 还支持 `lerobot_v3_index_cache`。这个 cache 存的是 root signature、episode selection、episode lengths 等初始化索引信息，用来减少重复启动时扫描 v3 metadata 的成本；它不是 non-idle JSON 的替代品。

### 6.2 主帧过滤

v3 的 `num_frames` 会在启用过滤时返回过滤后长度：

```python
if self._nonidle_filtered_indices is not None:
    return len(self._nonidle_filtered_indices)
```

`__getitem__()` 会先映射：

```python
raw_idx = self._nonidle_filtered_indices[idx]
dataset_idx, local_idx = self._resolve_frame_index(raw_idx)
item = external_v3_dataset[local_idx]
```

所以 v3 当前也能保证 dataloader 的主采样帧来自 non-idle 区间。

### 6.3 strict 模式下接管上下文窗口

只要传入了 `nonidle_filter_path`，v3 adapter 会设置 `_strict_nonidle=True`。构造外部 v3 dataset 时会刻意关闭外部的多帧窗口和图像 transform：

```python
image_transforms=None if self._strict_nonidle else image_transforms
delta_timestamps=None if self._strict_nonidle else child_delta_timestamps
```

然后 adapter 自己把 `child_delta_timestamps` 转成 `delta_indices`，保存在 `_V3DatasetEntry.delta_indices` 中。`__getitem__()` 检测到 `_nonidle_filtered_indices` 后会进入 `_getitem_strict_nonidle()`：

```text
idx -> raw_idx -> dataset_idx/local_idx -> anchor item
raw_idx -> episode_pos -> keep_rank -> query_indices
query_indices -> _query_strict_nonidle()
```

其中 `_get_strict_nonidle_query_indices()` 的 rank 逻辑和 v2 对齐：

```text
raw_idx -> keep_rank -> keep_rank + delta -> keep_indices[clamped_rank]
```

因此，当前 v3 strict non-idle 下，图像、state、action 的上下文窗口也沿过滤后的 non-idle 时间线采样。

同样示例：

```text
原始时间线: A B idle idle C D
过滤后:     A B C D
```

如果当前主帧是 `C`，并且需要前一帧：

- v3 strict non-idle 会取 `B`。
- 不会取 `C` 前面原始时间线里的 idle 帧。

### 6.4 v3 strict 模式如何读取 query 帧

v3 strict 模式没有直接批量 select parquet。它对每个 query index 做全局到 root-local 的映射：

```text
global query idx -> dataset_idx/q_local_idx
```

然后按字段类型选择读取方式：

- 视觉字段：调用外部 `entry.dataset[q_local_idx]`，让外部 v3 dataset 解码该帧图像/视频。
- 非视觉字段：优先调用外部 dataset 的 `get_raw_item(q_local_idx)`，避免不必要的视频解码；如果没有 `get_raw_item`，退回 `entry.dataset[q_local_idx]`。

同一个 local index 会放进 `raw_cache` 或 `full_cache`，避免一个样本内重复读取同一帧。

### 6.5 v3 strict 模式的限制

v3 strict non-idle 依赖外部 v3 dataset 的单帧读取能力。它会自己重建上下文时间线，但底层视频 seek、parquet row 读取、task 字段格式等仍由外部 v3 实现决定。

另外，strict query 会检查 query index 没有跨 dataset root。如果出现跨 root，说明 episode offset 或过滤表有问题，会直接抛错。

## 7. v2/v3 dataset 差异总结

| 维度 | v2 backend | v3 backend |
| --- | --- | --- |
| 底层实现 | 仓库内本地 `LeRobotDataset` | 外部 `lerobot` 包的 `LeRobotDataset` |
| parquet 访问 | ImageWAM 直接通过 HF datasets 加载 | 外部 LeRobot v3 dataset 内部处理 |
| 视频读取 | 本地代码控制视频 timestamp 查询与 decode | 外部 v3 控制，ImageWAM 只 patch EOF fallback |
| `delta_timestamps` | ImageWAM 转为 `delta_indices` 后自己算 query index | 普通路径传给外部 v3；strict non-idle 路径由 ImageWAM 转为 `delta_indices` |
| 多 root | `MultiLeRobotDataset` 合并多个本地 v2 dataset | `MultiLeRobotDatasetV3` 合并多个外部 v3 dataset |
| episode index | v2 metadata + `get_episode_data_index()` | v3 metadata 的 `dataset_from_index/dataset_to_index` 重建 |
| 上下文窗口 | ImageWAM 本地 `_get_query_indices()` 控制 | 普通路径由外部 v3 控制；strict non-idle 路径由 ImageWAM adapter 控制 |
| non-idle 支持 | 完整过滤后时间线 | strict filtered timeline，主帧和上下文帧都按过滤后时间线 |

## 8. non-idle 差异总结

| 维度 | v2 non-idle | v3 non-idle |
| --- | --- | --- |
| 使用同一 JSON | 是 | 是 |
| 主帧过滤 | 是 | 是 |
| `__len__()` 反映过滤后长度 | 是 | 是 |
| dataloader index 映射到原始帧 | 是 | 是 |
| 每 episode 保留帧表 | 是 | 是 |
| 原始帧到过滤 rank 映射 | 是 | 是 |
| 上下文帧沿过滤后时间线采样 | 是 | 是，strict non-idle 路径 |
| 上下文帧可能包含 idle | 通常不会，除非 JSON 保留了该 idle 段 | 通常不会，除非 JSON 保留了该 idle 段 |
| 底层单帧读取 | 本地 HF datasets + 本地 video decode | 外部 LeRobot v3 dataset |
| 实现复杂度 | 高，因本地控制 parquet/video/query index | 中等，adapter 控制 filtered timeline，但单帧读取复用外部 v3 |

一句话总结：

```text
v2: 主帧和上下文帧都按 non-idle 后的时间线采样。
v3: 当前 strict non-idle 下，主帧和上下文帧也都按 non-idle 后的时间线采样；普通无 filter 路径仍由外部 v3 处理上下文。
```

## 9. v3 strict non-idle 为什么这样实现

v2 能直接完整实现 non-idle 时间线，是因为 ImageWAM 本地掌控以下逻辑：

- `idx` 属于哪个 episode。
- `delta_timestamps` 到整数 `delta_indices` 的转换。
- 每个字段对应哪些 query indices。
- query index 越界时如何 clamp 和生成 padding mask。
- parquet 字段和视频帧如何根据 query indices 读取。

v3 的普通路径把这些上下文窗口细节封装在外部 LeRobot v3 dataset 内部。为了在不 fork 外部 v3 读取器的情况下支持 filtered timeline，当前实现采用了 adapter-level strict 模式：

1. 构造外部 v3 dataset 时关闭外部 `delta_timestamps`。
2. 在 ImageWAM adapter 中维护 keep indices 和 raw-to-rank。
3. 按 filtered rank 计算 query indices。
4. 对每个 query index 调用外部 v3 dataset 的单帧读取能力。

这个方案的好处是：filtered timeline 行为接近 v2，同时保留外部 LeRobot v3 对 v3 数据格式、视频 backend、raw item 读取等能力的支持。

代价是：strict 模式下 query 帧是逐帧读取再 stack，不像 v2 那样可以对非视频字段直接 `hf_dataset.select(q_idx)` 批量读取。因此 v3 strict non-idle 的行为更正确，但性能特征会更依赖外部 v3 dataset 的单帧读取效率和视频 seek 成本。

## 10. RoboTwin v3 + FLUX.2 4B 训练入口

专用脚本：

```bash
bash scripts/flux2/run_train_flux2_4b_robotwin_v3.sh
```

脚本默认设置：

```bash
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/robotwin2.0}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${DATA_ROOT}/robotwin2.0}"
export NONIDLE_FILTER_PATH="${NONIDLE_FILTER_PATH:-${ROBOTWIN_ROOT}/nonidle_ranges.json}"
export TASK_TYPE="robotwin"
export FLUX2_VARIANT="4b"
```

并传入关键 Hydra override：

```bash
data=robotwin_omnigen2_v3
data.train.dataset_dirs=[${ROBOTWIN_ROOT}]
data.val.dataset_dirs=[${ROBOTWIN_ROOT}]
data.train.nonidle_filter_path=${NONIDLE_FILTER_PATH}
data.val.nonidle_filter_path=${NONIDLE_FILTER_PATH}
```

如果数据集或 filter JSON 不在默认位置，可以覆盖：

```bash
ROBOTWIN_ROOT=/path/to/robotwin2.0 \
NONIDLE_FILTER_PATH=/path/to/nonidle_ranges.json \
bash scripts/flux2/run_train_flux2_4b_robotwin_v3.sh
```

## 11. 实践建议

如果目标是复现现有 RoboTwin v2 训练行为，v2 backend 和当前 v3 strict non-idle 在时间线语义上已经基本对齐：主帧和上下文帧都沿 filtered timeline 采样。

如果目标是使用 LeRobot v3 数据格式、torchcodec backend、v3 index cache 等能力，可以使用 v3 backend，并传入同一份 `nonidle_ranges.json`。当前 v3 strict non-idle 不再只是过滤 anchor frame，也会重建上下文窗口。

如果后续实验发现 v3 strict non-idle 性能瓶颈明显，应优先优化 query 帧读取路径，例如对非视觉字段做更批量化的 raw row 读取，或减少视觉 query 的重复视频 seek。
