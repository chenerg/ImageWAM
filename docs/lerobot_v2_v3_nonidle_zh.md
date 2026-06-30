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

v3 的上下文帧查询由外部 LeRobot v3 dataset 内部根据 `delta_timestamps` 完成。ImageWAM 适配层的 `__getitem__()` 基本只是：

1. 把全局 index 映射成某个 root 内的 local index。
2. 调用外部 v3 dataset 的 `dataset[local_idx]`。
3. 补充 `dataset_index` 和 hetero bridge 格式化。

因此，v3 当前没有像 v2 那样在 ImageWAM 本地重新实现 `_get_query_indices()`。这是 v2 和 v3 non-idle 行为差异的核心原因。

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

v2 是完整的“过滤后时间线”实现。

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

### 5.2 主帧过滤

当 dataloader 请求第 `idx` 个样本时，v2 会先做映射：

```python
raw_idx = self._nonidle_filtered_indices[idx]
```

所以训练样本的主帧不会落在被过滤掉的 no-op 区间。

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

## 6. v3 non-idle 实现

v3 当前实现的是“anchor 帧过滤”。

### 6.1 构建过滤列表

`MultiLeRobotDatasetV3._load_nonidle_filter()` 会读取同一个 `nonidle_ranges.json`，为所有 root/episode 构建：

```python
self._nonidle_filtered_indices
```

它会根据 v3 metadata 中的 episode length 和全局 `episode_data_index`，把 episode 内局部 `[start, end)` range 转换成全局原始帧 index。

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

### 6.3 上下文帧仍由外部 v3 backend 按原始时间线读取

v3 适配层没有重写外部 LeRobot v3 dataset 的窗口查询逻辑。`delta_timestamps` 已经传给外部 dataset，因此上下文帧由外部实现按原始时间戳读取。

同样示例：

```text
原始时间线: A B idle idle C D
过滤后:     A B C D
```

如果当前主帧是 `C`，并且需要前一帧：

- v3 的主帧 `C` 一定来自 non-idle 区间。
- 但上下文帧仍可能来自原始时间线中的 idle 帧，具体取决于外部 LeRobot v3 对 `delta_timestamps` 的解析。

因此 v3 当前不是完整的“过滤后时间线”实现，而是“anchor frame only”实现。

## 7. v2/v3 dataset 差异总结

| 维度 | v2 backend | v3 backend |
| --- | --- | --- |
| 底层实现 | 仓库内本地 `LeRobotDataset` | 外部 `lerobot` 包的 `LeRobotDataset` |
| parquet 访问 | ImageWAM 直接通过 HF datasets 加载 | 外部 LeRobot v3 dataset 内部处理 |
| 视频读取 | 本地代码控制视频 timestamp 查询与 decode | 外部 v3 控制，ImageWAM 只 patch EOF fallback |
| `delta_timestamps` | ImageWAM 转为 `delta_indices` 后自己算 query index | 传给外部 v3 dataset |
| 多 root | `MultiLeRobotDataset` 合并多个本地 v2 dataset | `MultiLeRobotDatasetV3` 合并多个外部 v3 dataset |
| episode index | v2 metadata + `get_episode_data_index()` | v3 metadata 的 `dataset_from_index/dataset_to_index` 重建 |
| 上下文窗口 | ImageWAM 本地 `_get_query_indices()` 控制 | 外部 LeRobot v3 控制 |
| non-idle 支持 | 完整过滤后时间线 | 当前为 anchor 帧过滤 |

## 8. non-idle 差异总结

| 维度 | v2 non-idle | v3 non-idle |
| --- | --- | --- |
| 使用同一 JSON | 是 | 是 |
| 主帧过滤 | 是 | 是 |
| `__len__()` 反映过滤后长度 | 是 | 是 |
| dataloader index 映射到原始帧 | 是 | 是 |
| 每 episode 保留帧表 | 是 | 当前未保存为独立表 |
| 原始帧到过滤 rank 映射 | 是 | 当前未实现 |
| 上下文帧沿过滤后时间线采样 | 是 | 否 |
| 上下文帧可能包含 idle | 通常不会，除非 JSON 保留了该 idle 段 | 可能会 |
| 实现复杂度 | 高，因本地控制 query index | 较低，因复用外部 v3 dataset |

一句话总结：

```text
v2: 主帧和上下文帧都按 non-idle 后的时间线采样。
v3: 当前只保证主帧来自 non-idle 区间，上下文仍按原始时间线读取。
```

## 9. 为什么 v3 不直接复刻 v2 行为

v2 能完整实现 non-idle 时间线，是因为 ImageWAM 本地掌控以下逻辑：

- `idx` 属于哪个 episode。
- `delta_timestamps` 到整数 `delta_indices` 的转换。
- 每个字段对应哪些 query indices。
- query index 越界时如何 clamp 和生成 padding mask。
- parquet 字段和视频帧如何根据 query indices 读取。

v3 的上下文窗口逻辑封装在外部 LeRobot v3 dataset 内部。ImageWAM 适配层拿到的是已经根据 `delta_timestamps` 取好的 sample。因此要让 v3 完全复刻 v2，需要至少做一类更大改动：

1. 绕开外部 v3 的 `delta_timestamps` 窗口查询，只请求单帧，然后在 ImageWAM 适配层按过滤后 rank 自己重建多帧上下文。
2. 或者扩展/patch 外部 LeRobot v3 dataset，让它支持按一个自定义 filtered timeline 查询上下文帧。

当前实现选择了更小的改动：先让 v3 支持 non-idle anchor 过滤，保证训练主样本不会落在 no-op 上，同时保持外部 v3 backend 的数据读取路径不变。

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

如果目标是复现现有 RoboTwin v2 训练行为，v2 backend 的 non-idle 实现更严格。

如果目标是使用 LeRobot v3 数据格式、torchcodec backend、v3 index cache 等能力，当前 v3 backend 已经可以使用同一份 `nonidle_ranges.json` 过滤主采样帧，但需要接受上下文帧仍可能包含原始时间线中的 idle 帧。

如果后续实验发现 v3 anchor-only 过滤与 v2 完整过滤存在明显训练差距，应优先补齐 v3 的 filtered timeline query：为 v3 维护每 episode 的 keep indices 和 raw-to-rank 映射，并在 ImageWAM 适配层接管上下文窗口读取。
