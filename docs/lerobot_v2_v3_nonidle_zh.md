# LeRobot v2/v3 Dataset 与 Non-idle 过滤实现差异

本文总结 ImageWAM 中 LeRobot v2 backend、LeRobot v3 backend 的实现差异，以及两者当前 non-idle/no-op 过滤逻辑的差异。

相关代码入口：

- backend 选择与通用 dataset 包装：`src/imagewam/datasets/lerobot/base_lerobot_dataset.py`
- RoboTwin 图像/动作样本包装：`src/imagewam/datasets/lerobot/robot_video_dataset.py`
- 本地 LeRobot v2 实现：`src/imagewam/datasets/lerobot/lerobot/lerobot_dataset.py`
- LeRobot v3/v4 本地实现：`src/imagewam/datasets/lerobot/lerobot_v4/lerobot_dataset.py`
- RoboTwin non-idle JSON 预计算：`scripts/data/compute_robotwin_nonidle_ranges.py`
- FLUX.2 Klein + RoboTwin v3 训练入口：`scripts/flux2/run_train_flux2_klein_imagewam.sh`，设置 `TASK_TYPE=robotwin_v3`

## 1. 背景

ImageWAM 的训练 dataset 入口不是直接暴露原始 LeRobot dataset，而是经过两层包装：

```text
RobotVideoDataset
  └── BaseLerobotDataset
        ├── MultiLeRobotDataset    # lerobot_backend="v2"
        └── MultiLeRobotDatasetV3  # lerobot_backend="v3"，当前是 lerobot_v4.MultiLeRobotDataset 的别名
```

`RobotVideoDataset` 负责把 LeRobot 样本整理成 ImageWAM 训练需要的格式，例如多相机拼接、图像增强、action/state processor、Qwen 文本缓存读取等。

`BaseLerobotDataset` 负责根据配置生成 `delta_timestamps`、episode split、backend 参数，并实例化 v2 或 v3 的 multi-root dataset。

`MultiLeRobotDataset` 和 `MultiLeRobotDatasetV3` 负责真正访问底层 LeRobot 数据。当前 `MultiLeRobotDatasetV3` 不再指向旧的外部 v3 adapter，而是直接指向 `lerobot_v4/lerobot_dataset.py` 里的 `MultiLeRobotDataset`。

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

v3 backend 当前使用仓库内 `lerobot_v4` 实现。代码里保留 `MultiLeRobotDatasetV3` 这个名字作为 backend 兼容入口，但它实际是 `lerobot_v4.MultiLeRobotDataset` 的别名：

```text
MultiLeRobotDatasetV3
  └── imagewam.datasets.lerobot.lerobot_v4.lerobot_dataset.MultiLeRobotDataset
        └── imagewam.datasets.lerobot.lerobot_v4.lerobot_dataset.LeRobotDataset
```

### 3.1 数据加载

v3 不再使用旧的 `src/imagewam/datasets/lerobot/lerobot/lerobot_dataset_v3.py` 外部 adapter。`BaseLerobotDataset` 会把 `lerobot_backend="v3"` 映射到 `lerobot_v4.MultiLeRobotDataset`，并为每个 root 构造一个 v4 `LeRobotDataset`：

```python
LeRobotDataset(
    ds_name,
    root=ds_root,
    episodes=selected_episodes,
    image_transforms=image_transforms,
    delta_timestamps=delta_timestamps,
    tolerance_s=...,
    download_videos=...,
    video_backend=...,
)
```

这里 `ds_name` 用作 `repo_id` / dataset 标识，`ds_root` 是真实本地数据目录。因为显式传入了 `root=ds_root`，本地读取不会再落到 `HF_LEROBOT_HOME / ds_name`。

当前 v3 backend 主要保留这些能力：

- 多 root 合并。
- `dataset_index` 注入。
- v3.0 chunked parquet / video layout 读取。
- `delta_timestamps` 直接交给 v4 `LeRobotDataset` 处理。
- `get_episode_data()` 会从 v3.0 chunk parquet 中按 `episode_index` 过滤单个 episode。

### 3.2 上下文帧查询

v3 backend 现在没有旧 adapter 的 strict non-idle 分支。`BaseLerobotDataset` 会把 `delta_timestamps` 直接传给 `lerobot_v4.MultiLeRobotDataset`，后者再传给每个 v4 `LeRobotDataset`。多帧上下文窗口由 v4 `LeRobotDataset` 内部根据 `delta_timestamps` 处理。

旧 `MultiLeRobotDatasetV3` adapter 支持的 `nonidle_filter_path`、`hetero_bridge`、`lerobot_v3_init_num_workers` 和 `lerobot_v3_index_cache` 不再传给 v4 backend；当前代码会记录 warning 并忽略这些参数。

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

当前 v3 backend 已经切到 `lerobot_v4.MultiLeRobotDataset`，不再使用旧 `MultiLeRobotDatasetV3` adapter 的 strict non-idle 实现。

因此，`nonidle_filter_path` 在 `lerobot_backend="v3"` 下会被忽略，并记录 warning。v3 backend 的 `num_frames` 直接来自各个 v4 子 dataset 的长度，`__getitem__()` 只负责把全局 index 映射到某个子 dataset 的 local index，再注入 `dataset_index`。

如果需要 no-op/idle 过滤，有两个当前可用路径：

1. 使用 v2 backend，继续走仓库内 `MultiLeRobotDataset` 的 non-idle 过滤。
2. 先用 materialize 脚本把 v3 数据物理裁剪成新的 LeRobot v3 root，再用 v3 backend 读取裁剪后的数据。

旧文档中提到的 `_nonidle_filtered_indices`、`_strict_nonidle`、`_query_strict_nonidle()`、`lerobot_v3_index_cache` 都属于已移除的外部 v3 adapter 路径，不适用于当前 v4 alias。

## 7. v2/v3 dataset 差异总结

| 维度 | v2 backend | v3 backend |
| --- | --- | --- |
| 底层实现 | 仓库内本地 `lerobot/lerobot_dataset.py` | 仓库内本地 `lerobot_v4/lerobot_dataset.py` |
| parquet 访问 | ImageWAM 通过 HF datasets 加载 v2 layout | v4 `load_nested_dataset()` 加载 v3.0 chunked layout |
| 视频读取 | 本地代码控制 video timestamp 查询与 decode | v4 本地 video utils 控制 decode |
| `delta_timestamps` | ImageWAM 转为 `delta_indices` 后自己算 query index | 传给 v4 `LeRobotDataset` 处理 |
| 多 root | `MultiLeRobotDataset` 合并多个本地 v2 dataset | `MultiLeRobotDatasetV3` alias 合并多个 v4 dataset |
| episode index | v2 metadata + `get_episode_data_index()` | v4 metadata 的 `dataset_from_index/dataset_to_index` |
| 上下文窗口 | ImageWAM 本地 `_get_query_indices()` 控制 | v4 `LeRobotDataset` 内部控制 |
| non-idle 支持 | 完整过滤后时间线 | 当前忽略 `nonidle_filter_path` |

## 8. non-idle 差异总结

| 维度 | v2 non-idle | v3 non-idle |
| --- | --- | --- |
| 使用同一 JSON | 是 | 否，当前忽略 |
| 主帧过滤 | 是 | 否 |
| `__len__()` 反映过滤后长度 | 是 | 否 |
| dataloader index 映射到原始帧 | 是 | 否 |
| 每 episode 保留帧表 | 是 | 否 |
| 原始帧到过滤 rank 映射 | 是 | 否 |
| 上下文帧沿过滤后时间线采样 | 是 | 否 |
| 上下文帧可能包含 idle | 通常不会，除非 JSON 保留了该 idle 段 | 可能，取决于原始 v3 数据是否已经物理裁剪 |
| 底层单帧读取 | 本地 HF datasets + 本地 video decode | v4 本地 HF datasets + video decode |
| 实现复杂度 | 高，因本地控制 parquet/video/query index | 低，当前不做 adapter-level 过滤 |

一句话总结：

```text
v2: 主帧和上下文帧都按 non-idle 后的时间线采样。
v3: 当前读取 v4 dataset 原始时间线；如需过滤，应先物理裁剪 v3 数据或使用 v2 backend。
```

## 9. v3 backend 参数兼容性

`BaseLerobotDataset` 仍保留 `lerobot_v3_init_num_workers`、`lerobot_v3_index_cache`、`nonidle_filter_path`、`hetero_bridge` 等配置入口，避免旧配置直接报错。但当 `lerobot_backend="v3"` 时，这些旧 adapter 参数不会传给 `lerobot_v4.MultiLeRobotDataset`，当前实现会记录 warning 并忽略它们。

## 10. RoboTwin v3 + FLUX.2 Klein 训练入口

RoboTwin v3 不再使用单独的专用脚本，而是通过通用 FLUX.2 Klein 入口指定 `TASK_TYPE=robotwin_v3`：

```bash
TASK_TYPE=robotwin_v3 FLUX2_VARIANT=4b \
bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

这个入口会生成对应的 task 名：

```bash
# FLUX2_VARIANT=4b
robotwin_v3_flux2_klein_4b_base_imagewam

# FLUX2_VARIANT=9b
robotwin_v3_flux2_klein_9b_base_imagewam
```

这些 task 配置会使用 `configs/data/robotwin_v3_omnigen2.yaml`，关键默认值包括：

```bash
data.data_root=./data/robotwin2.0_v3
data.robotwin_root=${data.data_root}/robotwin2.0_v3
data.train.lerobot_backend=v3
```

如果数据集不在默认位置，可以直接传 Hydra override：

```bash
TASK_TYPE=robotwin_v3 FLUX2_VARIANT=4b \
bash scripts/flux2/run_train_flux2_klein_imagewam.sh \
  data.data_root=/path/to/data_root \
  data.robotwin_root=/path/to/robotwin2.0_v3
```

## 11. 实践建议

如果目标是复现现有 RoboTwin v2 non-idle 训练行为，应继续使用 v2 backend，或先把 v3 数据物理裁剪后再训练。

如果目标是使用 LeRobot v3.0/v4 chunked 数据格式，可以使用 `lerobot_backend=v3`。这个 backend 当前会走 `lerobot_v4.MultiLeRobotDataset`，不会使用旧 `lerobot_v3_index_cache`，也不会消费 `nonidle_ranges.json`。

如果后续需要恢复 v3 adapter-level non-idle，需要在 `lerobot_v4.MultiLeRobotDataset` 内重新实现过滤后的 index space 和上下文窗口查询，而不是依赖已删除的旧 `lerobot_dataset_v3.py`。
