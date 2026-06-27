# ImageWAM 图形模型与子模块对比

本文基于 `D:\chenerg\imagewam` 仓库源码，整理 ImageWAM 中使用到的视觉/图形模型、各模型包含的主要子模块，以及不同模型栈之间的子模块差异。

> 说明：用户原始路径写作 `D:\chener\imagewam`，实际仓库路径为 `D:\chenerg\imagewam`。

## 1. 总体结构

`ImageWAM` 的统一骨架可以概括为：

```text
ImageWAM
  ├─ video_expert        # 图像编辑/视频生成视觉专家
  ├─ action_expert       # 与视觉专家 block 协议匹配的 ActionDiT
  ├─ mot                 # Mixture-of-Transformers，混合视觉 token 与 action token
  ├─ vae / ae            # 图像或视频 latent 编解码器
  ├─ text_encoder        # 可选文本/多模态编码器
  ├─ tokenizer           # 可选 tokenizer
  ├─ proprio_encoder     # 可选机器人状态投影层
  ├─ video scheduler     # video/image latent 扩散或 flow matching 调度器
  └─ action scheduler    # action latent 扩散或 flow matching 调度器
```

源码参考：

- `src/imagewam/models/backbones/imagewam.py`
- `src/imagewam/models/backbones/mot.py`
- `src/imagewam/models/backbones/*_video_expert.py`
- `src/imagewam/models/backbones/action_dit*.py`

`MoT` 会检查所有 expert 是否共享相同的层数、attention head 数、KV head 数、head dim 和 `block_protocol`。因此每个视觉基座都有一个匹配的 ActionDiT 变体。

## 2. 使用到的视觉/图形模型

| 模型栈 | 配置入口 | 视觉基座 / video_expert | 动作专家 | VAE / AE | 文本/条件模块 | MoT 协议 | 主要定位 |
|---|---|---|---|---|---|---|---|
| Wan2.2 legacy | `configs/model/imagewam.yaml`, `imagewam_joint.yaml`, `imagewam_idm.yaml`, `imagewam_noise_idm.yaml` | `WanVideoDiT` | `ActionDiT` | `WanVideoVAE` | Wan/T5 text encoder，可选加载 | `wan22` | 视频扩散 WAM 路径，接近 FastWAM |
| OmniGen2 | `configs/model/imagewam_omnigen2.yaml` | `OmniGen2VideoExpert` | `ActionDiTOmnigen2` | Diffusers `AutoencoderKL` | Qwen2.5-VL tokenizer/model，可选加载 | `omnigen2` | 图像编辑基座 |
| OmniGen2 NoiseIDM / CacheIDM | `imagewam_noise_idm_omnigen2.yaml`, `imagewam_cache_idm_omnigen2.yaml` | 同 OmniGen2 | 同 OmniGen2 | 同 OmniGen2 | 同 OmniGen2 | `omnigen2` | OmniGen2 的 IDM 训练/推理变体，不是新的视觉基座 |
| Ovis-U1 | `configs/model/imagewam_ovis_u1.yaml` | `OvisU1VideoExpert` 包装 Ovis visual generator / Yak MMDiT | `ActionDiTYak` | `visual_generator.get_vae()` | 可选 Ovis condition encoder | `yak` | Yak/Ovis double+single block 图像编辑路径 |
| FLUX.2 Klein 4B/9B | `imagewam_flux2_klein_4b_base.yaml`, `imagewam_flux2_klein_9b_base.yaml` | `Flux2VideoExpert` | `ActionDiTFlux2` | FLUX.2 `AutoEncoder` | Qwen3-4B 或 Qwen3-8B，可选加载 | `flux2` | FLUX.2 图像编辑路径，支持 LoRA |
| DIM / SANA | `configs/model/imagewam_dim.yaml` | `DimVideoExpert` 包装 SANA/DIM model | `ActionDiTSana` | DIM 自带 VAE | Qwen2.5-VL MLLM + projector + processor | `sana` | SANA/DIM 图像编辑路径，带 condition cache |

## 3. 各模型子模块

### 3.1 Wan2.2 legacy

主要构造入口是 `ImageWAM.from_wan22_pretrained()`。

子模块：

- `video_expert`: `WanVideoDiT`
  - `patch_embedding`
  - `text_embedding`
  - `time_embedding`
  - `time_projection`
  - `blocks: ModuleList[DiTBlock]`
  - `head`
  - `freqs`
  - 可选 `action_embedding`
- `action_expert`: `ActionDiT`
  - `action_encoder`
  - `text_embedding`
  - `time_embedding`
  - `blocks: ModuleList[DiTBlock]`
  - `head`
- `vae`: `WanVideoVAE`
  - `encoder`
  - latent `conv1`
  - latent `conv2`
  - `decoder`
- `text_encoder`: Wan text encoder，可在训练中关闭并使用预计算 context。
- `mot`: `MoT({"video": video_expert, "action": action_expert})`

特点：

- 视觉分支是视频扩散 DiT。
- block 协议是 `wan22`。
- action expert 必须和 video expert 的 `num_layers`、`num_heads`、`attn_head_dim` 对齐。

### 3.2 OmniGen2

主要构造入口是 `ImageWAM.from_omnigen2_pretrained()`。

子模块：

- `video_expert`: `OmniGen2VideoExpert`
  - `transformer`
  - `blocks = transformer.layers`
  - OmniGen2 RoPE / QKV / post block 逻辑
- `action_expert`: `ActionDiTOmnigen2`
  - `action_encoder`
  - `blocks: ModuleList[OmniGen2TransformerBlock]`
  - `head: LuminaLayerNormContinuous`
- `vae`: Diffusers `AutoencoderKL`
- `text_encoder`: `Qwen2_5_VLModel`，可选加载
- `tokenizer`: `AutoTokenizer`
- `mot`: `MoT`，协议 `omnigen2`

特点：

- action expert 的 `num_layers`、`num_heads`、`num_kv_heads`、`attn_head_dim` 会按加载的 OmniGen2 transformer 运行时覆盖。
- released OmniGen2 checkpoint 的结构可能和配置默认值不同，因此不能只看 YAML 中的占位值。

### 3.3 Ovis-U1

主要构造入口是 `ImageWAM.from_ovis_u1_pretrained()`。

子模块：

- `ovis_model`: `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
- `visual_generator = ovis_model.get_visual_generator()`
- `video_expert`: `OvisU1VideoExpert`
  - `transformer`
  - `double_blocks`
  - `single_blocks`
  - `txt_in`
  - `img_in`
  - `time_in`
  - `vector_in`
  - `pe_embedder`
  - `final_layer`
- `action_expert`: `ActionDiTYak`
  - `action_encoder`
  - `double_blocks`
  - `single_blocks`
  - `head: LastLayer`
- `vae`: `visual_generator.get_vae()`
- 可选 `ovis_condition_encoder`
- `mot`: `MoT`，协议 `yak`

特点：

- 结构分为 double blocks 和 single blocks。
- action expert 的 `residual_dim`、`num_heads`、`num_layers_double`、`num_layers_single` 会按 Ovis/Yak 视觉专家覆盖。

### 3.4 FLUX.2 Klein

主要构造入口是 `ImageWAM.from_flux2_klein_pretrained()`。

子模块：

- `video_expert`: `Flux2VideoExpert`
  - `transformer`
  - `double_blocks`
  - `single_blocks`
  - `final_layer`
  - latent packing/unpacking helpers
- 可选 LoRA：
  - 默认 target suffixes: `qkv`, `proj`, `linear1`, `linear2`, `img_mlp.0`, `img_mlp.2`, `txt_mlp.0`, `txt_mlp.2`
- `action_expert`: `ActionDiTFlux2`
  - `action_encoder`
  - `double_blocks`
  - `single_blocks`
  - `head: Flux2ActionHead`
- `vae`: FLUX.2 `AutoEncoder`
- `text_encoder`: 可选 Qwen3 causal LM wrapper
  - 4B: `Qwen/Qwen3-4B`, `text_dim=7680`
  - 9B: `Qwen/Qwen3-8B`, `text_dim=12288`
- `mot`: `MoT`，协议 `flux2`

特点：

- 4B 和 9B 的 text dim、head 数、double/single block 数不同。
- 支持对 FLUX.2 transformer 的线性层挂 LoRA。
- 视觉路径是图像编辑，不是长视频生成。

### 3.5 DIM / SANA

主要构造入口是 `ImageWAM.from_dim_pretrained()`。

子模块：

- `video_expert`: `DimVideoExpert`
  - `model`
  - `blocks = model.blocks`
  - `vae`
  - SANA/DIM latent encode/decode helpers
  - condition image / latent condition handling
- `action_expert`: `ActionDiTSana`
  - `action_encoder`
  - `_ActionSanaBlock[]`
  - `head_norm`
  - `head`
- `_ActionSanaBlock`
  - self-attention QKV
  - cross-attention Q/KV
  - q/k norm
  - MLP
- `text_encoder`: `Qwen2_5_VLForConditionalGeneration`，可选加载
- `processor`: `AutoProcessor`
- `dim_projector`: `mlp2x_gelu`, `2048 -> caption_dim`
- `mot`: `MoT`，协议 `sana`

特点：

- 相比其他模型，多了显式 condition/cache 分支。
- SANA 路径中部分注意力是线性注意力。
- `action_dit_config` 中的 `attn_dim`、`context_dim`、`num_heads`、`num_layers` 会按加载的 DIM/SANA video expert 覆盖。

## 4. 子模块差异对比

| 差异点 | Wan2.2 | OmniGen2 | Ovis-U1 | FLUX.2 Klein | DIM/SANA |
|---|---|---|---|---|---|
| 视觉模型类型 | 视频扩散 DiT | 图像编辑 transformer | Ovis/Yak 图像编辑 MMDiT | FLUX.2 图像编辑 transformer | SANA/DIM 图像编辑模型 |
| 视觉 block 结构 | 单一 `blocks` | 单一 `transformer.layers` | `double_blocks + single_blocks` | `double_blocks + single_blocks` | 单一 `model.blocks` |
| action expert | `ActionDiT` | `ActionDiTOmnigen2` | `ActionDiTYak` | `ActionDiTFlux2` | `ActionDiTSana` |
| action block 结构 | 单一 `blocks` | 单一 OmniGen2 blocks | double + single | double + single | 单一 SANA blocks |
| VAE/AE | `WanVideoVAE` | Diffusers `AutoencoderKL` | Ovis VAE | FLUX.2 `AutoEncoder` | DIM VAE |
| 文本条件 | Wan/T5 hidden states | Qwen2.5-VL hidden states | Ovis condition encoder 或预计算条件 | Qwen3 hidden states | Qwen2.5-VL MLLM + projector |
| text dim | 常见 4096 | 2048 | 4096 | 7680 / 12288 | `video_expert.caption_dim` |
| MoT protocol | `wan22` | `omnigen2` | `yak` | `flux2` | `sana` |
| 是否有 GQA 对齐 | 基础 MHA | 有 `num_kv_heads` | 通常按 Yak 结构对齐 | 按 FLUX.2 结构对齐 | 按 SANA 结构对齐 |
| 是否支持 LoRA | 未见该路径专门配置 | 未见该路径专门配置 | 未见该路径专门配置 | 支持 `flux2_lora_config` | 未见该路径专门配置 |
| 是否有 condition cache 特化 | IDM/CacheIDM 变体有 | NoiseIDM/CacheIDM 变体有 | 有 video cache 推理路径 | 有 video cache 推理路径 | 有 SANA condition cache |
| 主要输出目标 | video latent / action | target image latent / action | target image latent / action | target image latent / action | target image latent / action |

## 5. 结论

`imagewam` 中实际支持的视觉/图形基座可以分为五类：

1. `Wan2.2`: legacy 视频扩散 WAM 路径。
2. `OmniGen2`: 图像编辑基座，包含普通、NoiseIDM、CacheIDM 变体。
3. `Ovis-U1`: Yak/Ovis double+single block 图像编辑路径。
4. `FLUX.2 Klein`: FLUX.2 图像编辑路径，4B/9B 两种规格，支持 LoRA。
5. `DIM/SANA`: SANA/DIM 图像编辑路径，额外带 condition/cache 分支。

这些模型共享 `ImageWAM = visual expert + action expert + MoT` 的顶层设计，但每个视觉基座的 transformer block 组织方式不同，所以必须配套不同的 `ActionDiT` 变体，并通过 MoT 的 `block_protocol` 分流处理。
