# Spec: 集成 LLaVA-OneVision-2 (8B) 到 sglang 与现有 GRPO 训练框架

- 日期: 2026-05-28
- 目标 checkpoint: `/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct`
- 范围: 让 OV2 跑通 sglang rollout + AReaL FSDP trainer 的端到端 GRPO 流程

## 1. 背景与现状

仓库 (`LLaVA-OneVision-1.5-RL`) 当前只支持 LLaVA-OneVision-**1.5**。集成路径已有:

- Trainer 侧: HF `AutoModelForImageTextToText` + `trust_remote_code`,checkpoint 自带 `modeling_*.py` (OV1.5 由 `cp 3rdparty/modeling/modeling_llavaonevision1_5.py` 注入)
- Rollout 侧: sglang 0.5.2 (editable, `3rdparty/sglang`),已注册 `LLaVAOneVision1_5_ForConditionalGeneration`
- 数据/Workflow: `areal.workflow.vision_rlvr` → `processor(images=..., text=...)` → `pixel_values` + `image_grid_thw` 进 `multi_modal_input`

LLaVA-OneVision-**2** 的 checkpoint 已经自带完整 `auto_map` (modeling/processing/video_processing/configuration),`trust_remote_code=True` 即可加载。Trainer 侧理论上**零改动**,核心工作量在 **sglang rollout 侧**。

## 2. OV2 vs OV1.5 关键差异

| 项 | OV1.5 | OV2 |
|---|---|---|
| `model_type` | `llavaonevision1_5` | `llava_onevision2` |
| `architectures` | `LLaVAOneVision1_5_ForConditionalGeneration` | `LlavaOnevision2ForConditionalGeneration` |
| Vision encoder model_type | `rice_vit` | `onevision_encoder` |
| Vision RoPE | 2D (H, W) — `RiceRotaryEmbedding` | 3D (T, H, W) 4:6:6 split — `VisionRotaryEmbedding` |
| Vision attention | 全注意力 | window attention (cu_seqlens by `frame_windows_size=4`) |
| Patch merger | LayerNorm + 2×Linear (with `spatial_merge_size=2`) | 同结构 + **可选 `patch_positions` 加位置嵌入** |
| Text backbone | Qwen3-8B | Qwen3-8B (相同) |
| Processor extras | `pixel_values`, `image_grid_thw` | `pixel_values`, `image_grid_thw`, **`patch_positions`** |
| 视频 | 走 video path | 视频被 processor 拍扁成 multi-image,模型只走 image path |

## 3. 环境

在基础 `install.sh` 之外,单独提供 `install_ov2.sh`,只做"基础环境装好后,额外升级 transformers 到 v5.x 并拉齐配套依赖":

```bash
# install_ov2.sh
#!/bin/bash
# Prereq: bash install.sh 已跑完,且 sglang 已 editable 安装
uv pip install transformers==5.7.0 openai==2.2.0
```

(transformers 5.7.0 会自动把 huggingface_hub 升到 1.x、tokenizers 升到 0.22、numpy 升到 2.4,这些是配套传递依赖,不需要手动 pin。)

已验证:
- transformers 5.7.0 ✅
- sglang 0.5.2 (editable, `3rdparty/sglang`) ✅
- areal 0.3.4 ✅
- OV2 `AutoConfig` / `AutoProcessor` `trust_remote_code` 加载 ✅

## 4. 设计决策

1. **Vision tower 使用 sglang TP linear**(仿 OV1.5):`VisionAttention(use_qkv_parallel=True)` + `ColumnParallelLinear`/`RowParallelLinear`,即便当前 TP=1 也保持架构对称
2. **`patch_positions` 走 `MultimodalDataItem` 额外字段**:processor 端把 `patch_positions` 与 `image_grid_thw` 一起塞到 `mm_items`,model `get_image_feature` 拼出来传给 vision tower
3. **Trainer 侧零改动**(初版):直接用 HF `trust_remote_code` 加载 OV2 自带 modeling。如果 FSDP 包装出问题再考虑把 modeling 拷到 `3rdparty/modeling/`

## 5. 改动清单

### 5.1 新建 (3 个文件)

| 文件 | 作用 | 估计行数 |
|---|---|---|
| `3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py` | sglang OV2 模型主体 (vision tower + Qwen3Model + LM head + forward + weight loader) | ~700-900 |
| `3rdparty/sglang/python/sglang/srt/multimodal/processors/llava_onevision_2.py` | sglang OV2 multimodal preprocessor (input_text 展开 + 图片预处理 + mrope) | ~250-300 |
| `install_ov2.sh` | 在 `install.sh` 之上额外升级 transformers 到 5.7.0 | ~5 |

### 5.2 修改 (1 处白名单)

| 文件 | 改动 |
|---|---|
| `3rdparty/sglang/python/sglang/srt/configs/model_config.py:731` | `multimodal_model_archs` 列表追加 `"LlavaOnevision2ForConditionalGeneration"` |

### 5.3 暂不做(下个 milestone)

- GRPO 训练配置 (`configs/llavaov2-8b_grpo.yaml`)
- `workflow/vision_rlvr.py` 增加 `patch_positions` 透传(端到端训练才需要)
- `3rdparty/sglang/.../parser/conversation.py` & `lang/chat_template.py` 注册 OV2 chat template matcher

## 6. sglang OV2 模型文件骨架

```
class OneVisionEncoderEmbeddings(nn.Module):
    # patch_dim → hidden_size 的单 Linear
    ...

class VisionRotaryEmbedding(nn.Module):
    # 3D 4:6:6 切分;forward_from_positions(patch_positions) → [L, head_dim/2]
    ...

class OneVisionEncoderMLP(nn.Module):
    # ColumnParallelLinear + GELU + RowParallelLinear
    ...

class OneVisionEncoderBlock(nn.Module):
    # pre_norm + VisionAttention(use_qkv_parallel=True, rotary_embed="normal", proj_bias=True,
    #                            qkv_backend="fa3"|"triton_attn"|"sdpa")
    # + post_norm + OneVisionEncoderMLP
    ...

class OneVisionEncoderTransformer(nn.Module):
    # embeddings → layernorm_pre → blocks(cu_seqlens by frame_windows_size) → [optional layernorm_post]
    # → LlavaOnevision2VisionPatchMerger(x, patch_positions)
    def forward(self, pixel_values, grid_thw, patch_positions): ...

class LlavaOnevision2ForConditionalGeneration(nn.Module):
    def __init__(self, config, quant_config, prefix):
        self.visual = OneVisionEncoderTransformer(config.vision_config, ...)
        self.model = Qwen3Model(config.text_config, ...)
        self.lm_head = ParallelLMHead(...)

    def get_image_feature(self, items):
        # 从 items 拼 pixel_values + image_grid_thw + patch_positions
        return self.visual(pv, grid_thw, patch_positions)

    def forward(self, input_ids, positions, forward_batch, get_embedding=False):
        if self.is_mrope_enabled: positions = forward_batch.mrope_positions
        hidden = general_mm_embed_routine(input_ids, forward_batch,
                                          language_model=self.model,
                                          multimodal_model=self, positions=positions)
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def load_weights(self, weights):
        # qkv: HF 是 "qkv.weight/bias" 单个 Linear,sglang VisionAttention 用 "qkv_proj"
        # mlp: HF 是 SiglipMLP 的 fc1/fc2,sglang 用 ColumnParallel/RowParallel
        # text 部分按 OV1.5 现有方式:strip "model.language_model." 前缀,按 stacked_params_mapping 合并 qkv/gate_up
        ...

EntryClass = [LlavaOnevision2ForConditionalGeneration]
```

## 7. sglang OV2 multimodal processor 骨架

```
class LlavaOnevision2_ImageProcessor(SGLangBaseProcessor):
    models = [LlavaOnevision2ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *a, **kw):
        # 沿用 OV1.5 的 mm_tokens (image_token / vision_start / vision_end)
        # 用 hf_config.image_token_id / vision_start_token_id / vision_end_token_id
        ...

    async def process_mm_data_async(self, image_data, input_text, request_obj, *a, **kw):
        base = self.load_mm_data(prompt=input_text, image_data=image_data,
                                 video_data=request_obj.video_data,
                                 multimodal_tokens=self.mm_tokens)
        # OV2 processor 内部用 Qwen2VL image_processor,接受 PIL,无需额外 smart_resize
        items, input_ids, ret = self.process_and_combine_mm_data(base, self.mm_tokens)
        # ret 必须保留 patch_positions(把它附到 MultimodalDataItem.image_grid_thw 旁边)
        # mrope: 沿用 MRotaryEmbedding.get_rope_index(model_type="qwen2_vl", spatial_merge_size=2, ...)
        return {"input_ids": ..., "mm_items": ..., "mrope_positions": ..., "mrope_position_delta": ...,
                "im_start_id": ..., "im_end_id": ..., "im_token_id": ..., "video_token_id": ...}
```

注意点:
- `process_and_combine_mm_data` 默认只识别 `pixel_values` / `image_grid_thw`。`patch_positions` 需要走"塞 `MultimodalDataItem` 额外属性"的口子(参考 sglang 源码 `process_and_combine_mm_data` 的实现,或手动重写组合)
- OV2 processor 期望调用方传 `text=` 而 sglang 这里调的是底层 `image_processor` 直接处理 image。两种路径都行,关键是 input_text 的 `<|image_pad|>` 展开规则匹配:OV2 的展开规则是 `(t*h*w) // (spatial_merge_size**2)` 个 image_token,与 OV1.5 相同

## 8. 风险与未决项

1. **sglang `process_and_combine_mm_data` 是否支持额外字段**:未读源码,如果不支持需要自己 inline 实现 combine
2. **`VisionRotaryEmbedding` 3D 4:6:6 freqs 怎么 wire 进 sglang `VisionAttention`**:OV2 自己实现的是 `apply_rotary_pos_emb(q, k, freqs)` 直接用 `(B, H, L, D)` 上的逐元素乘,sglang `VisionAttention` 的 `rotary_embed="normal"` 期望传 `position_embeddings`,内部用 `cos/sin` 旋转。需要确认 sglang 的实现是否兼容 OV2 的 freqs 形状(已知 OV2 把 `[L, D/2]` 复制成 `[L, D]` 后传入,这与 sglang 期望相符)
3. **window attention `cu_seqlens` 与 sglang `VisionAttention(cu_seqlens=...)` 的对接**:OV2 的 `_build_cu_seqlens` 按 `frame_windows_size=4` 切窗,这本质上就是给 `VisionAttention` 传 `cu_seqlens` 做 varlen flash attention,sglang 已支持
4. **Trainer 侧 OV2 自带 modeling 在 FSDP 下能否正常 forward + backward + gradient_checkpointing**:checkpoint config 写 `transformers_version=5.7.0`,而 AReaL/sglang 内部某些 utility 可能假设 v4 接口。验证步骤放在实现阶段
5. **mrope `model_type="qwen2_vl"`**:OV2 vision 是 3D RoPE,而 Qwen3 text 的 mrope 是 qwen2_vl 风格 — 这是 LLM 侧 mrope,跟 vision 侧的 3D RoPE 是两件事,前者复用应当 OK

## 9. 验收

本次只做 sglang inference 接入,验收 gate 三个:

| 阶段 | 验证 |
|---|---|
| 静态 import | `from sglang.srt.models.llava_onevision_2 import LlavaOnevision2ForConditionalGeneration` 不抛错 |
| sglang server | `python -m sglang.launch_server --model-path <OV2> --trust-remote-code` 起来 + `/v1/chat/completions` 单图单 prompt 返回非空 completion |
| numerical 对齐 | 同一 prompt+image,sglang OV2 与 HF transformers `model.generate(do_sample=False)` 输出 token-level 一致(或近似,允许 fp16 抖动) |
| TP 验证 | sglang server 用 `--tp 2` 和 `--tp 4` 各跑一次单图 prompt,输出与 TP=1 一致(允许 reduce 顺序导致的 fp16 抖动) |

## 10. 不在范围(本 milestone)

- GRPO 端到端训练(等 sglang inference 跑通、numerical 对齐后再开新 milestone)
- Trainer 侧 FSDP forward/backward 验证(同上)
- 视频任务(OV2 processor 把视频拍扁成 multi-image,沿用图像路径,跑通图像后再覆盖)
- OV2 codec video backend
- LoRA / 量化
- 自定义 chat_template 注册到 sglang conversation matcher
