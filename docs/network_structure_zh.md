# ImageWAM 网络结构详解

本文从代码实现角度解释 ImageWAM 的网络结构，并穿插 FastWAM 的对应设计，帮助理解 ImageWAM 为什么可以把“视频生成式 world action model”改写成“图像编辑式 world action model”。

代码参考点：

- ImageWAM 主类：`src/imagewam/models/backbones/imagewam.py`
- ImageWAM 混合专家：`src/imagewam/models/backbones/mot.py`
- ImageWAM action expert：`src/imagewam/models/backbones/action_dit*.py`
- ImageWAM 图像编辑基座适配器：`flux2_video_expert.py`、`omnigen2_video_expert.py`、`ovis_u1_video_expert.py`、`dim_video_expert.py`
- FastWAM 对照：`D:/chenerg/fastwam/src/fastwam/models/wan22/fastwam.py`

## 1. 一句话总览

FastWAM 的核心思路是：用 Wan2.2 视频扩散 DiT 做 video expert，用 ActionDiT 做 action expert，通过 MoT 在每层 self-attention 里把 video token 和 action token 混在一起，让动作预测可以利用视觉世界模型的表征。

ImageWAM 保留了这个“video expert + action expert + MoT”的骨架，但把 video expert 从 Wan2.2 视频生成模型换成图像编辑模型基座，例如 FLUX.2、OmniGen2、Ovis-U1、DIM。于是模型不再必须生成长视频，而是学习“给定当前图像，编辑出下一状态图像”，再借助同一套混合注意力结构预测机器人动作。

## 2. 整体模块图

```text
输入样本
  ├─ source image / 当前观测图像
  ├─ target image / 下一状态图像
  ├─ language instruction / 任务指令
  ├─ action sequence / 机器人动作序列
  └─ optional proprio / 机器人状态

        │
        ▼
条件编码
  ├─ 图像 VAE / AE 编码：source image -> source latent tokens
  ├─ 文本编码：instruction -> text hidden states
  └─ proprio encoder：robot state -> 额外 text token

        │
        ├────────────────────────────┐
        ▼                            ▼
Video Expert                    Action Expert
图像编辑 DiT                    ActionDiT 变体
预测 target image noise         预测 action noise

        │                            │
        └──────────── MoT ───────────┘
             每层混合 attention
             video/action token 共享注意力上下文

        │
        ▼
输出
  ├─ pred_video：target image latent 的流匹配/扩散速度场
  └─ pred_action：action latent 的流匹配/扩散速度场
```

这里的 “video” 名字来自 FastWAM 代码遗产。在 ImageWAM 的 FLUX.2、OmniGen2、Ovis-U1 路径里，它更准确地说是 image-editing expert：输入当前图像条件，预测目标图像 latent 的去噪方向。

## 3. 核心对象关系

`ImageWAM` 包含五类核心成员：

```text
ImageWAM
  ├─ video_expert        # 图像编辑/视频生成基座包装器
  ├─ action_expert       # ActionDiT / ActionDiTFlux2 / ActionDiTOmnigen2 / ActionDiTYak
  ├─ mot                 # Mixture-of-Transformers，连接两个 expert
  ├─ vae                 # 图像或视频 latent 编解码器
  ├─ text_encoder        # 可选，训练通常使用预计算 text hidden states
  └─ proprio_encoder     # 可选，把机器人状态投影到 text hidden dim
```

`self.dit = self.mot` 是为了兼容 trainer 的冻结和优化器逻辑：训练时真正被当作主干优化的是 MoT，而 MoT 内部持有 video/action 两个专家。

## 4. 与 FastWAM 的结构对比

| 维度 | FastWAM | ImageWAM |
|---|---|---|
| 基座模型 | Wan2.2 TI2V 视频扩散 DiT | FLUX.2、OmniGen2、Ovis-U1、DIM 等图像编辑模型 |
| video 分支目标 | 生成多帧视频 latent | 编辑/预测下一张图像 latent，部分 legacy 配置仍支持 Wan2.2 视频 |
| action 分支 | ActionDiT | 按基座协议适配的 ActionDiT 变体 |
| 融合方式 | MoT 混合 video/action self-attention | 仍是 MoT，但需要支持多种 block protocol |
| 注意力协议 | 主要是 Wan2.2 block | `wan22`、`omnigen2`、`yak`、`flux2`、`sana` |
| 训练损失 | 通常 action loss，joint/idm 变体含 video loss | video image-edit loss + action loss，常见权重 `0.5 : 1.0` |
| 推理输出 | video frames + action | image edit 结果被包装成单帧 video list + action |
| 设计重点 | 用视频世界模型帮助动作预测 | 验证 WAM 是否真的需要视频生成，还是图像编辑已足够 |

FastWAM 可以理解成“视频版 WAM”。ImageWAM 则是在它的骨架上替换视觉专家，把未来预测的粒度从 video 改为 image editing。

## 5. MoT：连接两个专家的关键层

MoT 全称可以理解为 Mixture-of-Transformers。它不是简单把两个模型输出拼起来，而是在每一个 transformer block 的 self-attention 内部做混合。

普通单专家 block 大致是：

```text
x -> norm/modulation -> Q,K,V -> self-attention -> residual/MLP -> x'
```

MoT 的做法是：

```text
video tokens  -> video block 产生 Qv,Kv,Vv
action tokens -> action block 产生 Qa,Ka,Va

Q = concat(Qv, Qa)
K = concat(Kv, Ka)
V = concat(Vv, Va)

mixed_attention(Q,K,V, mask)

再把 attention 输出按 token 长度切回：
  video attention output  -> video block 的 post 部分
  action attention output -> action block 的 post 部分
```

这样每一层里 action token 都能通过注意力读取视觉 token，而视觉 token 是否读取 action token 由 attention mask 决定。

### 5.1 MoT 的一致性要求

MoT 初始化时会检查多个 expert 是否兼容：

- layer 数必须一致，或在对应协议里有可对齐结构。
- attention head 数必须一致。
- KV head 数必须一致。
- head dim 必须一致。
- block protocol 必须一致。

这就是为什么 ImageWAM 的配置和 factory 会在运行时覆盖部分 ActionDiT 参数。例如 OmniGen2 的 action config 中写了占位层数和 head 参数，但实际会按加载的 OmniGen2 transformer 自动覆盖。

### 5.2 ImageWAM 为什么有多个 block protocol

FastWAM 只需要适配 Wan2.2 block。ImageWAM 支持多种图像编辑基座，每个基座的 transformer block 结构不同：

- `wan22`：Wan2.2 风格 DiTBlock。
- `omnigen2`：OmniGen2 block，Q/K/V、RoPE、post block 写法不同。
- `yak`：Ovis-U1/Yak 风格，分 double blocks 和 single blocks。
- `flux2`：FLUX.2 风格，同样有图文流和图像流结构。
- `sana`：DIM/SANA 风格，使用线性注意力路径。

MoT 的职责就是把这些不同 block 拆成统一的“产生 QKV -> 混合 attention -> 回填 block 后半段”的接口。

## 6. Video Expert：ImageWAM 中实际是图像编辑专家

ImageWAM 提供多个构造入口：

```text
create_imagewam_flux2_klein   -> ImageWAM.from_flux2_klein_pretrained
create_imagewam_omnigen2      -> ImageWAM.from_omnigen2_pretrained
create_imagewam_ovis_u1       -> ImageWAM.from_ovis_u1_pretrained
create_imagewam_dim           -> ImageWAM.from_dim_pretrained
create_imagewam               -> legacy Wan2.2/FastWAM-like path
```

### 6.1 FLUX.2 ImageWAM

推荐入口是 `configs/model/imagewam_flux2_klein_4b_base.yaml`。

关键配置：

```yaml
variant: klein-base-4b
qwen3_model_spec: Qwen/Qwen3-4B
qwen_context_len: 512
pack_proprio_after_text: true

action_dit_config:
  hidden_dim: 1024
  num_heads: 24
  attn_head_dim: 128
  num_layers_double: 5
  num_layers_single: 20
  max_action_horizon: 64

loss:
  lambda_video: 0.5
  lambda_action: 1.0
```

FLUX.2 路径的 text dim 由 variant 决定：

- 4B：`text_dim = 7680`，默认 Qwen3-4B。
- 9B：`text_dim = 12288`，默认 Qwen3-8B。

图像路径大致是：

```text
source image -> AE encode -> source/ref tokens
target image -> AE encode -> target latent tokens
target latent + noise + timestep -> video_expert.pre_dit
source/ref tokens + text tokens -> 作为图像编辑条件
video expert 输出 pred_video
```

### 6.2 OmniGen2 ImageWAM

OmniGen2 路径使用 `OmniGen2VideoExpert` 包装原始 transformer，并使用 Qwen2.5-VL hidden states 作为文本条件。

配置中明确写了一个重要点：action expert 的结构参数会在运行时按加载的 OmniGen2 transformer 覆盖。原因是 MoT 必须让 action expert 与 video expert 的层数、head 数、KV head 数、head dim 对齐。

### 6.3 Ovis-U1 ImageWAM

Ovis-U1 路径从 `AutoModelForCausalLM` 中取出 visual generator，再包装成 `OvisU1VideoExpert`。

对应 action expert 是 `ActionDiTYak`，结构上匹配 Yak/Ovis 的 double block + single block：

```yaml
residual_dim: 1536
num_heads: 12
num_layers_double: 6
num_layers_single: 12
```

### 6.4 DIM ImageWAM

DIM 路径走 SANA 协议。它和其他路径最大的差异是 MoT 里有 `cond` 分支，并且部分 attention 是线性注意力。可以把它理解成：

```text
video target tokens + condition image tokens + action tokens
```

三者在 SANA 专门路径里交互。

## 7. Action Expert：动作序列的扩散/流匹配模型

最基础的 `ActionDiT` 结构如下：

```text
action_tokens [B, T_action, action_dim]
  │
  ▼
Linear action_encoder: action_dim -> hidden_dim
  │
  ├─ time_embedding(timestep) -> t_mod
  ├─ text_embedding(context)  -> context_emb
  └─ RoPE freqs for action positions
  │
  ▼
N 层 DiTBlock
  │
  ▼
Linear head: hidden_dim -> action_dim
```

在 ImageWAM 里，不同基座会使用不同 ActionDiT 变体：

- Wan2.2 legacy：`ActionDiT`
- OmniGen2：`ActionDiTOmnigen2`
- Ovis-U1：`ActionDiTYak`
- FLUX.2：`ActionDiTFlux2`
- DIM：`ActionDiTSana` 或相关 SANA 适配路径

这些变体的共同目标是一样的：把 noisy action sequence 作为 token 序列，预测 action 的 denoising velocity/noise target。

## 8. 训练数据流

ImageWAM 训练的典型输入是：

```text
sample
  ├─ source image / input image
  ├─ target image / video field 中的目标帧
  ├─ action [B, T_action, action_dim]
  ├─ context [B, L, text_dim]
  ├─ context_mask [B, L]
  ├─ action_is_pad
  ├─ image_is_pad
  └─ optional proprio
```

训练时一般不在线跑文本编码器，而是读取预计算的 text hidden states。这一点和 FastWAM 类似：训练入口通常要求 `sample["context"]` 和 `sample["context_mask"]`。

### 8.1 图像分支训练

以 FLUX.2 为例：

```text
target_latent = encode(target_image)
noise_video = randn_like(target_latent)
t_video = sample_training_t()
noisy_target = add_noise(target_latent, noise_video, t_video)
target_video = training_target(target_latent, noise_video, t_video)

video_pre = video_expert.pre_dit(
  x=noisy_target,
  timestep=t_video,
  text/context,
  ref/source image tokens
)
```

图像分支预测的是 target image latent 的 flow matching target。

### 8.2 动作分支训练

```text
noise_action = randn_like(action)
t_action = sample_training_t()
noisy_action = add_noise(action, noise_action, t_action)
target_action = training_target(action, noise_action, t_action)

action_pre = action_expert.pre_dit(
  action_tokens=noisy_action,
  timestep=t_action,
  context=text/context
)
```

### 8.3 MoT 联合前向

```text
tokens_out = mot(
  embeds_all={
    "video": video_pre["tokens"],
    "action": action_pre["tokens"],
  },
  attention_mask=...,
  freqs_all=...,
  context_all=...,
  t_mod_all=...,
)

pred_video  = video_expert.post_dit(tokens_out["video"], video_pre)
pred_action = action_expert.post_dit(tokens_out["action"], action_pre)
```

最终 loss：

```text
loss_total = lambda_video * loss_video + lambda_action * loss_action
```

常见 ImageWAM 图像编辑基座配置是：

```text
lambda_video = 0.5
lambda_action = 1.0
```

FastWAM 的基础配置 `fastwam.yaml` 只写了 `lambda_action: 1.0`，而 joint/idm 变体会显式引入 video loss。

## 9. Attention Mask 的直觉

MoT 是否真的让 action 看到 visual token，完全由 mask 控制。

FastWAM 基础版的 mask 逻辑是：

```text
video -> video：按 video_expert 的视频因果/首帧规则
action -> action：全可见
action -> video：只看第一帧 video tokens
```

FastWAMJoint 改成：

```text
action -> video：看完整 video tokens
```

FastWAMIDM 则引入 teacher forcing：

```text
[noisy_video, cond_video, action]

noisy_video -> noisy_video
cond_video  -> cond_video
action      -> action
action      -> cond_video only
```

ImageWAM 继承了这些思想，但因为不同图像编辑基座 token 组织不同，所以每个 stack 都有自己的 mask 构造函数：

- `_build_mot_attention_mask`
- `_build_mot_attention_mask_omnigen2`
- `_build_mot_attention_mask_ovis_u1`
- `_build_mot_attention_mask_flux2`

直觉上，ImageWAM 的 action branch 通常需要能看到 source image / condition image / target-edit tokens 中有意义的视觉上下文；video branch 是否看 action 取决于具体训练路径和 mask 设计。

## 10. 推理路径

ImageWAM 的 `infer()` 会按 `self.stack` 分流：

```text
stack == "flux2"    -> infer_action_flux2 + infer_video_flux2
stack == "omnigen2" -> infer_action_omnigen2 + infer_video_omnigen2
stack == "ovis_u1"  -> infer_action_ovis_u1 + infer_video_ovis_u1
stack == "dim"      -> infer_dim_separate
else                -> legacy infer_joint
```

这和 FastWAM 有明显差异。FastWAM 的 `infer()` 默认走 `infer_joint()`，同时 denoise video latent 和 action latent。ImageWAM 的主流图像编辑基座通常把 action 推理和 image edit 推理拆开：

```text
Action 推理：
  source image -> video/cache prefill
  noisy action -> action expert
  action expert 通过 MoT 读取 video cache
  多步 scheduler denoise -> action

Image 推理：
  source image + text
  noisy target image latent
  video expert denoise -> edited image
```

最后为了兼容评测接口，单张 edited image 会被包装成：

```python
{"action": action_out, "video": [PIL.Image]}
```

## 11. FastWAM 到 ImageWAM 的迁移关系

可以按下面方式理解两者代码关系：

```text
FastWAM
  ├─ Wan2.2 video DiT
  ├─ ActionDiT
  ├─ MoT(wan22 only)
  └─ video/action joint denoising

ImageWAM
  ├─ image editing video expert
  │   ├─ FLUX.2
  │   ├─ OmniGen2
  │   ├─ Ovis-U1
  │   └─ DIM
  ├─ matched ActionDiT variant
  ├─ MoT(multi protocol)
  └─ image-edit/action denoising
```

也就是说，ImageWAM 不是完全重写 FastWAM，而是保留了 FastWAM 中最关键的动作-视觉融合机制，然后把视觉世界模型替换成图像编辑模型。

## 12. 主要配置对比

### FastWAM 基础配置

```yaml
video_dit_config:
  hidden_dim: 3072
  ffn_dim: 14336
  text_dim: 4096
  num_heads: 24
  attn_head_dim: 128
  num_layers: 30
  fuse_vae_embedding_in_latents: true
  video_attention_mask_mode: first_frame_causal

action_dit_config:
  hidden_dim: 1024
  ffn_dim: 4096
  num_heads: 24
  attn_head_dim: 128
  num_layers: 30
  text_dim: 4096

loss:
  lambda_action: 1.0
```

### ImageWAM FLUX.2 4B 配置

```yaml
variant: klein-base-4b
qwen3_model_spec: Qwen/Qwen3-4B
qwen_context_len: 512
pack_proprio_after_text: true

action_dit_config:
  hidden_dim: 1024
  num_heads: 24
  attn_head_dim: 128
  num_layers_double: 5
  num_layers_single: 20
  max_action_horizon: 64

loss:
  lambda_video: 0.5
  lambda_action: 1.0
```

### ImageWAM OmniGen2 配置

```yaml
qwen_context_len: 128
pack_proprio_after_text: false

action_dit_config:
  residual_dim: 1024
  num_heads: runtime override
  num_kv_heads: runtime override
  attn_head_dim: runtime override
  num_layers: runtime override
  max_action_horizon: 64

loss:
  lambda_video: 0.5
  lambda_action: 1.0
```

### ImageWAM Ovis-U1 配置

```yaml
load_condition_encoder: true

action_dit_config:
  residual_dim: 1536
  num_heads: 12
  num_layers_double: 6
  num_layers_single: 12
  max_action_horizon: 64

loss:
  lambda_video: 0.5
  lambda_action: 1.0
```

## 13. 为什么 ImageWAM 可以只做图像编辑

对机器人策略来说，动作预测最需要的是“当前状态 + 任务语义 + 未来可达状态”的表示。FastWAM 用视频生成模型来建模这个未来状态；ImageWAM 的假设是，很多 manipulation benchmark 中，短时未来的关键变化可以由当前图像到目标/下一状态图像的编辑来提供，不一定需要完整多帧视频。

从网络角度看：

- FastWAM 的 video expert 学的是时序 latent。
- ImageWAM 的 video expert 学的是 image edit latent。
- 二者都通过 MoT 把视觉预测 token 暴露给 action token。
- 因此只要图像编辑 token 能表达任务相关的物体、接触、位姿变化，action expert 就仍然可以学到有效动作。

## 14. 阅读代码建议

建议按下面顺序读：

1. 先读 `configs/model/imagewam_flux2_klein_4b_base.yaml`，理解模型由哪些组件拼出来。
2. 再读 `src/imagewam/runtime.py` 的 `create_imagewam_flux2_klein()`，看配置如何进入构造函数。
3. 读 `ImageWAM.from_flux2_klein_pretrained()`，看 video expert、action expert、MoT 如何创建。
4. 读 `_training_loss_flux2()`，看 source/target image 和 action 如何被加噪、预测、计算 loss。
5. 读 `mot.py` 的 `forward()` 和 `forward_action_with_video_cache()`，理解每层混合 attention。
6. 对照 `D:/chenerg/fastwam/src/fastwam/models/wan22/fastwam.py` 的 `training_loss()` 和 `infer_action()`，看 ImageWAM 保留了哪些 FastWAM 结构。

## 15. 记忆版总结

可以把 ImageWAM 记成下面公式：

```text
ImageWAM = Image Editing DiT + Action DiT + MoT mixed attention
```

和 FastWAM 的关系：

```text
FastWAM  = Video DiT        + Action DiT + MoT
ImageWAM = Image Editing DiT + Action DiT + MoT
```

ImageWAM 的贡献点不在于重新发明 action diffusion，而在于证明视觉分支不一定要是重型视频生成模型；只要图像编辑基座足够强，并且通过 MoT 与 action expert 在 token 层交互，就可以形成有效的 world action model。
