# LLaVA-OneVision-2 sglang 集成 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 LLaVA-OneVision-2-8B-Instruct 接入仓库 `3rdparty/sglang`,使 `sglang.launch_server --model-path <OV2>` 能跑起来并通过单图 inference 的 numerical 对齐,TP=1/2/4 均验证

**Architecture:** 仿 OV1.5 在 sglang 新增一对文件(model + multimodal processor),vision tower 用 sglang `VisionAttention` + `ColumnParallel/RowParallelLinear` 写,文本侧复用 `Qwen3Model`;`patch_positions` 走 `MultimodalDataItem.model_specific_data`(由 OV2 processor 把 `"patch_positions"` 注册到 `ATTR_NAME_TO_MODALITY`,base 类会自动透传);环境侧 transformers 升到 5.7.0(单独 `install_ov2.sh`)。

**Tech Stack:** sglang 0.5.2 (editable @ `3rdparty/sglang`), transformers 5.7.0, torch 2.8.0, flash_attn 2.8.1, Qwen3-8B text backbone

---

## File Structure

新增/修改清单(每个文件单一职责):

| 文件 | 职责 |
|---|---|
| `install_ov2.sh` (new) | `install.sh` 之后的 OV2 增量环境(只升级 transformers) |
| `3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py` (new) | OV2 模型主体:vision tower (`OneVisionEncoder*` + `LlavaOnevision2VisionPatchMerger`) + Qwen3Model + LM head + forward + weight loader |
| `3rdparty/sglang/python/sglang/srt/multimodal/processors/llava_onevision_2.py` (new) | OV2 multimodal processor:image token 展开 + 调用 HF OV2 processor + 把 `patch_positions` 注册到 `ATTR_NAME_TO_MODALITY` + mrope 计算 |
| `3rdparty/sglang/python/sglang/srt/configs/model_config.py` (modify L731) | `multimodal_model_archs` 追加 `"LlavaOnevision2ForConditionalGeneration"` |
| `tests/sglang_ov2/test_vision_rope.py` (new) | 单测 vision RoPE 3D 4:6:6 切分 |
| `tests/sglang_ov2/test_cu_seqlens.py` (new) | 单测 `_build_cu_seqlens` (window by `frame_windows_size=4`) |
| `tests/sglang_ov2/test_processor.py` (new) | 单测 OV2 sglang processor 输出字段完整、`patch_positions` 在 mm_items 里 |
| `tests/sglang_ov2/test_weight_name_map.py` (new) | 单测 HF → sglang 权重名映射函数 |
| `tests/sglang_ov2/test_inference_offline.py` (new) | 集成测试:sglang `Engine` 单图 prompt 与 HF transformers `generate` 对齐 |
| `tests/sglang_ov2/test_tp.py` (new) | 集成测试:server 启动 TP=2/4,单图输出与 TP=1 一致 |

---

## 前置约定

- OV2 checkpoint 路径在所有 task 中固定为 `OV2_PATH=/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct`,在测试中通过环境变量提供
- 所有命令前提:`source /data/v-kaichen/LLaVA-OneVision-1.5-RL/.venv/bin/activate`
- 所有 git 操作的 cwd:`/data/v-kaichen/LLaVA-OneVision-1.5-RL`
- TDD 节奏:每个 Task 内部 红→绿→提交;跨 Task 也独立可提交

---

## Task 1: 环境脚本 install_ov2.sh

**Files:**
- Create: `install_ov2.sh`

- [ ] **Step 1: 写脚本**

```bash
#!/bin/bash
# OV2 incremental env setup. Prereq: `bash install.sh` 已完成.
# Upgrades transformers to 5.7.0 (the version OV2 checkpoint targets).
# transformers 5.7.0 拉走 huggingface_hub>=1.0, tokenizers>=0.22, numpy>=2 等配套依赖。
set -e
uv pip install transformers==5.7.0 openai==2.2.0
echo "OV2 env ready. Verify with:"
echo "  python -c 'from transformers import AutoConfig; AutoConfig.from_pretrained(\"$OV2_PATH\", trust_remote_code=True)'"
```

- [ ] **Step 2: chmod + 跑一次确认幂等**

```bash
chmod +x install_ov2.sh
bash install_ov2.sh
python -c "import transformers; assert transformers.__version__ == '5.7.0'; print('OK', transformers.__version__)"
```

期望:`OK 5.7.0`

- [ ] **Step 3: Commit**

```bash
git add install_ov2.sh
git commit -m "chore: add install_ov2.sh for OV2 transformers v5 upgrade"
```

---

## Task 2: model_config.py 白名单

**Files:**
- Modify: `3rdparty/sglang/python/sglang/srt/configs/model_config.py:731`

- [ ] **Step 1: 写测试**

`tests/sglang_ov2/__init__.py`(空文件)+ `tests/sglang_ov2/test_model_arch_registered.py`:

```python
from sglang.srt.configs.model_config import is_multimodal_model

def test_ov2_arch_recognized_as_multimodal():
    assert is_multimodal_model(["LlavaOnevision2ForConditionalGeneration"]) is True
```

- [ ] **Step 2: 跑测试,确认失败**

```bash
pytest tests/sglang_ov2/test_model_arch_registered.py -v
```

期望:FAIL(返回 False)

- [ ] **Step 3: 改源码**

文件 `3rdparty/sglang/python/sglang/srt/configs/model_config.py` 第 731 行后追加一行:

```python
    "LLaVAOneVision1_5_ForConditionalGeneration",
    "LlavaOnevision2ForConditionalGeneration",
]
```

- [ ] **Step 4: 跑测试,确认通过**

```bash
pytest tests/sglang_ov2/test_model_arch_registered.py -v
```

期望:PASS

- [ ] **Step 5: Commit**

```bash
git add tests/sglang_ov2/__init__.py tests/sglang_ov2/test_model_arch_registered.py 3rdparty/sglang/python/sglang/srt/configs/model_config.py
git commit -m "feat(sglang): register LlavaOnevision2 arch as multimodal"
```

---

## Task 3: VisionRotaryEmbedding (3D 4:6:6)

**Files:**
- Create: `3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py`
- Test: `tests/sglang_ov2/test_vision_rope.py`

OV2 vision RoPE 把 `head_dim/2` 按 4:6:6 切给 T:H:W。`forward_from_positions(patch_positions)` 输入 `[L, 3]`(每行 `[t, h, w]`),输出 `[L, head_dim/2]`;调用方再 `cat([f, f], dim=-1)` 得到 `[L, head_dim]`,拆 cos/sin 给 `VisionAttention`。

- [ ] **Step 1: 写测试**

```python
# tests/sglang_ov2/test_vision_rope.py
import torch
from sglang.srt.models.llava_onevision_2 import VisionRotaryEmbedding

class _Cfg:
    hidden_size = 1024
    num_attention_heads = 16   # head_dim = 64, half = 32, unit = 2 → t=8, h=12, w=12
    rope_theta = 10000.0

def test_split_4_6_6():
    rope = VisionRotaryEmbedding(_Cfg())
    assert rope.t_size == 8 and rope.h_size == 12 and rope.w_size == 12
    assert rope.head_dim == 64 and rope.half == 32

def test_forward_from_positions_shape():
    rope = VisionRotaryEmbedding(_Cfg())
    L = 16
    pp = torch.zeros(L, 3, dtype=torch.long)
    pp[:, 1] = torch.arange(L) % 4   # h
    pp[:, 2] = torch.arange(L) % 4   # w
    freqs = rope.forward_from_positions(pp)
    assert freqs.shape == (L, 32), freqs.shape
```

- [ ] **Step 2: 跑测试,确认失败**

```bash
pytest tests/sglang_ov2/test_vision_rope.py -v
```

期望:FAIL(模块不存在)

- [ ] **Step 3: 创建 `llava_onevision_2.py` 骨架 + VisionRotaryEmbedding**

```python
# 3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py
"""Inference-only LLaVA-OneVision-2 model for sglang."""
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class VisionRotaryEmbedding(nn.Module):
    """3D (T,H,W) RoPE with 4:6:6 split, ported from OV2 checkpoint.

    head_dim must be divisible by 16; half = head_dim // 2;
    unit = half // 16; t_size = 4*unit, h_size = 6*unit, w_size = 6*unit.
    """

    def __init__(self, config):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        assert head_dim % 2 == 0 and head_dim % 16 == 0
        half = head_dim // 2
        assert half % 16 == 0
        unit = half // 16
        self.head_dim = head_dim
        self.half = half
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit
        self.base = base
        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """patch_positions: [L, 3] long → freqs: [L, half] float32."""
        pp = patch_positions.to(self.inv_freq_t.device)
        t, h, w = pp[:, 0].float(), pp[:, 1].float(), pp[:, 2].float()
        freqs_t = t.unsqueeze(-1) * self.inv_freq_t.unsqueeze(0)  # [L, t_size]
        freqs_h = h.unsqueeze(-1) * self.inv_freq_h.unsqueeze(0)
        freqs_w = w.unsqueeze(-1) * self.inv_freq_w.unsqueeze(0)
        return torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)  # [L, half]
```

- [ ] **Step 4: 跑测试,确认通过**

```bash
pytest tests/sglang_ov2/test_vision_rope.py -v
```

期望:PASS

- [ ] **Step 5: 验证与 OV2 自带实现 numerical 对齐**

加测试用例 `test_match_hf_impl`:

```python
def test_match_hf_impl():
    import sys
    sys.path.insert(0, "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct")
    from modeling_llava_onevision2 import VisionRotaryEmbedding as HFRope
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(
        "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct",
        trust_remote_code=True,
    ).vision_config
    ours = VisionRotaryEmbedding(cfg)
    theirs = HFRope(cfg)
    pp = torch.tensor([[0, 0, 0], [0, 1, 2], [1, 0, 3], [2, 3, 1]], dtype=torch.long)
    a = ours.forward_from_positions(pp)
    b = theirs.forward_from_positions(pp)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
```

```bash
pytest tests/sglang_ov2/test_vision_rope.py::test_match_hf_impl -v
```

期望:PASS

- [ ] **Step 6: Commit**

```bash
git add 3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py tests/sglang_ov2/test_vision_rope.py
git commit -m "feat(sglang/ov2): port 3D 4:6:6 vision RoPE with HF parity test"
```

---

## Task 4: `_build_cu_seqlens` (window attention by frame_windows_size)

**Files:**
- Modify: `3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py` (新增模块级函数)
- Test: `tests/sglang_ov2/test_cu_seqlens.py`

OV2 vision encoder 按 `frame_windows_size=4` 把长视频切窗;图像走 `t=1` 的退化情况(单窗 = 单图)。函数纯,易测。

- [ ] **Step 1: 写测试**

```python
# tests/sglang_ov2/test_cu_seqlens.py
import torch
from sglang.srt.models.llava_onevision_2 import build_cu_seqlens

def test_single_image_t1():
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    cu, ms = build_cu_seqlens(grid_thw, total_patches=16, fixed_t=4, device="cpu")
    assert cu.tolist() == [0, 16] and ms == 16

def test_two_images_concat():
    grid_thw = torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.int64)
    cu, ms = build_cu_seqlens(grid_thw, total_patches=20, fixed_t=4, device="cpu")
    assert cu.tolist() == [0, 16, 20] and ms == 16

def test_video_split_by_window():
    grid_thw = torch.tensor([[10, 2, 2]], dtype=torch.int64)
    cu, ms = build_cu_seqlens(grid_thw, total_patches=40, fixed_t=4, device="cpu")
    # 10 = 4 + 4 + 2 frames → chunks of 16, 16, 8 patches
    assert cu.tolist() == [0, 16, 32, 40] and ms == 16

def test_total_mismatch_raises():
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    import pytest
    with pytest.raises(ValueError):
        build_cu_seqlens(grid_thw, total_patches=99, fixed_t=4, device="cpu")
```

- [ ] **Step 2: 跑测试,确认失败**

```bash
pytest tests/sglang_ov2/test_cu_seqlens.py -v
```

期望:FAIL(`build_cu_seqlens` not defined)

- [ ] **Step 3: 实现函数**

在 `llava_onevision_2.py` 文件末尾加:

```python
def build_cu_seqlens(
    grid_thw: torch.Tensor,
    total_patches: int,
    fixed_t: Optional[int] = 4,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    """Port of LlavaOnevision2VisionPretrainedModel._build_cu_seqlens.

    Splits grids whose t > fixed_t into chunks of fixed_t (+ remainder).
    Returns (cu_seqlens int32[N+1], max_seqlen int).
    """
    if grid_thw is None or grid_thw.numel() == 0:
        return (
            torch.tensor([0, total_patches], dtype=torch.int32, device=device),
            total_patches,
        )
    if device is None:
        device = grid_thw.device

    cu_seqlens = [0]
    max_seqlen = 0
    current_len = 0
    for idx in range(grid_thw.shape[0]):
        t_val = int(grid_thw[idx, 0])
        h_val = int(grid_thw[idx, 1])
        w_val = int(grid_thw[idx, 2])
        if fixed_t is not None and fixed_t > 0 and t_val > fixed_t:
            num_full = t_val // fixed_t
            rem = t_val % fixed_t
            for _ in range(num_full):
                chunk = fixed_t * h_val * w_val
                current_len += chunk
                max_seqlen = max(max_seqlen, chunk)
                cu_seqlens.append(current_len)
            if rem > 0:
                chunk = rem * h_val * w_val
                current_len += chunk
                max_seqlen = max(max_seqlen, chunk)
                cu_seqlens.append(current_len)
        else:
            chunk = t_val * h_val * w_val
            current_len += chunk
            max_seqlen = max(max_seqlen, chunk)
            cu_seqlens.append(current_len)

    if cu_seqlens[-1] != total_patches:
        raise ValueError(
            f"cu_seqlens calculation mismatch: total_patches={total_patches}, "
            f"calculated={cu_seqlens[-1]}, grid_thw={grid_thw}"
        )
    return torch.tensor(cu_seqlens, dtype=torch.int32, device=device), max_seqlen
```

- [ ] **Step 4: 跑测试,确认通过**

```bash
pytest tests/sglang_ov2/test_cu_seqlens.py -v
```

期望:4 passed

- [ ] **Step 5: Commit**

```bash
git add 3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py tests/sglang_ov2/test_cu_seqlens.py
git commit -m "feat(sglang/ov2): port build_cu_seqlens with window splitting"
```

---

## Task 5: Vision encoder 子模块 (Embeddings / MLP / Attention / Block / Encoder)

**Files:**
- Modify: `3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py`

策略:把 `OneVisionEncoderEmbeddings` / `LlavaOnevision2VisionPatchMerger` / `Siglip2MultiheadAttentionPoolingHead`(`use_head=False` 不需要) 等 vision 子模块按 sglang 风格复刻。Attention 用 `VisionAttention(use_qkv_parallel=True, rotary_embed="normal", proj_bias=True)`。MLP 用 `ColumnParallelLinear + GELU + RowParallelLinear`。每个 sub-module 一个简短 shape 测。

- [ ] **Step 1: 写 Embeddings 测试**

```python
# tests/sglang_ov2/test_vision_submodules.py
import torch
from sglang.srt.models.llava_onevision_2 import OneVisionEncoderEmbeddings

class _Cfg:
    hidden_size = 1024
    patch_size = 14
    num_channels = 3
    layer_norm_eps = 1e-6
    layer_norm_type = "layer_norm"

def test_embeddings_shape():
    emb = OneVisionEncoderEmbeddings(_Cfg())
    x = torch.randn(8, 3, 14, 14)
    out = emb(x)
    assert out.shape == (8, 1024)
```

- [ ] **Step 2: 跑测试,确认失败**

```bash
pytest tests/sglang_ov2/test_vision_submodules.py::test_embeddings_shape -v
```

- [ ] **Step 3: 实现 `OneVisionEncoderEmbeddings`**

加到 `llava_onevision_2.py`(在 `VisionRotaryEmbedding` 之后):

```python
class OneVisionEncoderEmbeddings(nn.Module):
    """patch tensor [N, C, P, P] → [N, hidden_size]."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.embed_dim = config.hidden_size
        self.patch_embedding = nn.Linear(
            self.num_channels * self.patch_size * self.patch_size,
            self.embed_dim,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [N, C, P, P] or already flattened [N, C*P*P]
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.flatten(1)
        return self.patch_embedding(pixel_values)
```

注:实际权重命名/形状必须与 checkpoint 一致。等 Task 9 weight loader 跑通后,如果对不上回到这里调整 patch_embedding 的具体实现(可能是 `nn.Conv2d` 等)。

- [ ] **Step 4: 跑测试,确认通过**

- [ ] **Step 5: 写 PatchMerger 测试**

```python
from sglang.srt.models.llava_onevision_2 import LlavaOnevision2VisionPatchMerger

def test_patch_merger_shape_no_pos_enc():
    merger = LlavaOnevision2VisionPatchMerger(
        dim=4096, context_dim=1024, spatial_merge_size=2, layer_norm_eps=1e-6,
        use_patch_position_encoding=False,
    )
    L = 16  # multiple of merge^2 = 4
    x = torch.randn(L, 1024)
    out = merger(x, patch_positions=None)
    assert out.shape == (L // 4, 4096)
```

- [ ] **Step 6: 跑测试,确认失败**

- [ ] **Step 7: 实现 `LlavaOnevision2VisionPatchMerger`**(用 TP layers)

```python
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix
from torch.nn import LayerNorm


class LlavaOnevision2VisionPatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-5,
        use_patch_position_encoding: bool = False,
        patch_position_encoding_type: str = "absolute",
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.spatial_merge_size = spatial_merge_size
        self.use_patch_position_encoding = use_patch_position_encoding
        self.ln_q = LayerNorm(context_dim, eps=layer_norm_eps)
        self.mlp = nn.ModuleList([
            ColumnParallelLinear(
                self.hidden_size, self.hidden_size, bias=True,
                quant_config=quant_config, prefix=add_prefix("mlp.0", prefix),
            ),
            nn.GELU(),
            RowParallelLinear(
                self.hidden_size, dim, bias=True,
                quant_config=quant_config, prefix=add_prefix("mlp.2", prefix),
            ),
        ])
        if use_patch_position_encoding:
            if patch_position_encoding_type != "absolute":
                raise ValueError(f"Unknown encoding type: {patch_position_encoding_type}")
            self.pos_emb_h = nn.Embedding(max_position_embeddings, dim)
            self.pos_emb_w = nn.Embedding(max_position_embeddings, dim)

    def forward(self, x: torch.Tensor, patch_positions: Optional[torch.Tensor] = None):
        if patch_positions is not None and patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)
        x = self.ln_q(x).view(-1, self.hidden_size)
        fc1, act, fc2 = self.mlp
        x, _ = fc1(x); x = act(x); x, _ = fc2(x)
        if self.use_patch_position_encoding and patch_positions is not None:
            pp = patch_positions.view(-1, self.spatial_merge_size ** 2, 3)[:, 0, :]
            pp = (pp // self.spatial_merge_size).long()
            x = x + self.pos_emb_h(pp[:, 1]) + self.pos_emb_w(pp[:, 2])
        return x
```

- [ ] **Step 8: 跑测试,确认通过**

- [ ] **Step 9: 写 MLP / Attention / Block 测试(shape + 不抛错即可,数值对齐留给整体 vision tower)**

```python
from sglang.srt.models.llava_onevision_2 import OneVisionEncoderMLP, OneVisionEncoderBlock

class _BlockCfg(_Cfg):
    intermediate_size = 4096
    num_attention_heads = 16
    attention_dropout = 0.0
    hidden_act = "gelu"

def test_mlp_shape():
    mlp = OneVisionEncoderMLP(_BlockCfg())
    out = mlp(torch.randn(8, 1024))
    assert out.shape == (8, 1024)
```

- [ ] **Step 10: 实现 `OneVisionEncoderMLP` 和 `OneVisionEncoderBlock`**

```python
from transformers.activations import ACT2FN
from sglang.srt.layers.attention.vision import VisionAttention


def _get_norm(config):
    if getattr(config, "layer_norm_type", "layer_norm") == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class OneVisionEncoderMLP(nn.Module):
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, bias=True,
            quant_config=quant_config, prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=True,
            quant_config=quant_config, prefix=add_prefix("fc2", prefix),
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x):
        x, _ = self.fc1(x); x = self.act(x); x, _ = self.fc2(x)
        return x


class OneVisionEncoderBlock(nn.Module):
    """Pre-norm + VisionAttention + Pre-norm + MLP."""

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.layer_norm1 = _get_norm(config)
        self.layer_norm2 = _get_norm(config)
        self.attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            rotary_embed="normal",
            proj_bias=True,
            qkv_backend=None,  # let VisionAttention auto-pick (fa3 on Hopper, triton_attn on A100/A6000, sdpa on CPU)
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = OneVisionEncoderMLP(
            config, quant_config=quant_config, prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        h = self.layer_norm1(x)
        h = rearrange(h, "s b ... -> b s ...")
        attn = self.attn(h, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        attn = rearrange(attn, "b s ... -> s b ...")
        x = x + attn
        x = x + self.mlp(self.layer_norm2(x))
        return x
```

- [ ] **Step 11: 跑测试,确认通过**

```bash
pytest tests/sglang_ov2/test_vision_submodules.py -v
```

- [ ] **Step 12: Commit**

```bash
git add 3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py tests/sglang_ov2/test_vision_submodules.py
git commit -m "feat(sglang/ov2): port vision encoder submodules (emb, mlp, attn, block, merger)"
```

---

## Task 6: Vision tower 顶层 `OneVisionEncoderTransformer.forward`

**Files:**
- Modify: `3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py`
- Test: `tests/sglang_ov2/test_vision_tower_forward.py`

把 embeddings + layernorm_pre + encoder + merger 串起来。RoPE freqs `[L, half]` 复制成 `[L, head_dim]` 后拆 `(cos, sin)`。

- [ ] **Step 1: 写测试**

```python
# tests/sglang_ov2/test_vision_tower_forward.py
import torch
from sglang.srt.models.llava_onevision_2 import OneVisionEncoderTransformer

class _Cfg:
    hidden_size = 256       # use small dims for fast test
    num_attention_heads = 4 # head_dim=64 → 16-divisible ✓
    rope_theta = 10000.0
    patch_size = 14
    num_channels = 3
    layer_norm_eps = 1e-6
    layer_norm_type = "layer_norm"
    intermediate_size = 512
    attention_dropout = 0.0
    hidden_act = "gelu"
    num_hidden_layers = 2
    out_hidden_size = 256
    spatial_merge_size = 2
    frame_windows_size = 4
    use_head = False
    use_patch_position_encoding = False

def test_forward_single_image():
    tower = OneVisionEncoderTransformer(_Cfg()).eval()
    grid = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    pv = torch.randn(16, 3, 14, 14)
    pp = torch.zeros(16, 3, dtype=torch.long)
    pp[:, 1] = torch.arange(16) // 4
    pp[:, 2] = torch.arange(16) % 4
    with torch.no_grad():
        out = tower(pv, grid_thw=grid, patch_positions=pp)
    # merger reduces by spatial_merge_size^2 = 4
    assert out.shape == (16 // 4, 256), out.shape
```

- [ ] **Step 2: 跑测试,确认失败**

- [ ] **Step 3: 实现 `OneVisionEncoderTransformer`**

```python
class OneVisionEncoderTransformer(nn.Module):
    """Top-level vision tower, mirrors LlavaOnevision2VisionPretrainedModel."""

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.frame_windows_size = getattr(config, "frame_windows_size", 4)
        self.embeddings = OneVisionEncoderEmbeddings(config)
        self.layernorm_pre = _get_norm(config)
        self.video_rope = VisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList([
            OneVisionEncoderBlock(
                config, quant_config=quant_config,
                prefix=add_prefix(f"encoder.layers.{i}", prefix),
            )
            for i in range(config.num_hidden_layers)
        ])
        self.layernorm_post = (
            _get_norm(config) if getattr(config, "use_head", False) else None
        )
        self.merger = LlavaOnevision2VisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
            use_patch_position_encoding=getattr(config, "use_patch_position_encoding", False),
            patch_position_encoding_type=getattr(config, "patch_position_encoding_type", "absolute"),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            quant_config=quant_config, prefix=add_prefix("merger", prefix),
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        patch_positions: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embeddings(pixel_values)              # [N, D]
        if h.dim() == 2:
            h = h.unsqueeze(0)                          # [1, N, D]
        if patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)
        freqs = self.video_rope.forward_from_positions(patch_positions)  # [N, half]
        freqs = torch.cat([freqs, freqs], dim=-1)       # [N, head_dim]
        cos = freqs.cos().to(h.dtype)
        sin = freqs.sin().to(h.dtype)
        h = self.layernorm_pre(h)
        cu_seqlens, _ = build_cu_seqlens(
            grid_thw=grid_thw, total_patches=h.shape[1],
            fixed_t=self.frame_windows_size, device=h.device,
        )
        # blocks expect [s, b, ...] convention (matches OV1.5)
        h = rearrange(h, "b s d -> s b d")
        for blk in self.blocks:
            h = blk(h, cu_seqlens=cu_seqlens, position_embeddings=(cos, sin))
        h = rearrange(h, "s b d -> b s d")
        if self.layernorm_post is not None:
            h = self.layernorm_post(h)
        h = h.squeeze(0)                                # [N, D]
        return self.merger(h, patch_positions=patch_positions)
```

- [ ] **Step 4: 跑测试,确认通过**

```bash
pytest tests/sglang_ov2/test_vision_tower_forward.py -v
```

- [ ] **Step 5: Commit**

```bash
git add 3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py tests/sglang_ov2/test_vision_tower_forward.py
git commit -m "feat(sglang/ov2): wire OneVisionEncoderTransformer end-to-end"
```

---

## Task 7: 顶层 `LlavaOnevision2ForConditionalGeneration` (forward 协议)

**Files:**
- Modify: `3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py`

把 vision + Qwen3Model + LM head + sglang `general_mm_embed_routine` 串起来。这部分仿 OV1.5 `LLaVAOneVision1_5_ForConditionalGeneration` 写,关键区别在 `get_image_feature`:从 `item.patch_positions` 取出额外字段(通过 `MultimodalDataItem.__getattr__` 透到 `model_specific_data`)。

- [ ] **Step 1: 编写代码(本步无独立单测,与 Task 10 集成测试一起验)**

在 `llava_onevision_2.py` 末尾追加:

```python
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3 import Qwen3Model


class LlavaOnevision2ForConditionalGeneration(nn.Module):
    default_bitsandbytes_target_modules = [
        ".fc2.", ".fc1.", ".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.visual = OneVisionEncoderTransformer(
            config.vision_config, quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )
        self.model = Qwen3Model(
            config.text_config, quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size, config.text_config.hidden_size,
            quant_config=quant_config, prefix=add_prefix("lm_head", prefix),
        )
        self.is_mrope_enabled = (
            hasattr(config.text_config, "rope_scaling")
            and config.text_config.rope_scaling is not None
            and "mrope_section" in config.text_config.rope_scaling
        )
        self.logits_processor = LogitsProcessor(config.text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs,
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            self.visual.dtype,
        )
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)
        patch_positions = torch.cat(
            [item.patch_positions for item in items], dim=0,
        )
        return self.visual(pixel_values, grid_thw=image_grid_thw,
                           patch_positions=patch_positions)

    # OV2 video is folded into image path by the processor; alias.
    def get_video_feature(self, items):
        return self.get_image_feature(items)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids, forward_batch=forward_batch,
            language_model=self.model, multimodal_model=self, positions=positions,
        )
        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch,
            )
        return self.pooler(hidden_states, forward_batch)
```

- [ ] **Step 2: 静态 import 测试**

```python
# tests/sglang_ov2/test_top_level_import.py
def test_class_imports_and_has_entryclass():
    from sglang.srt.models import llava_onevision_2 as m
    assert hasattr(m, "LlavaOnevision2ForConditionalGeneration")
```

(EntryClass 放到 Task 9 一并定义)

- [ ] **Step 3: Commit**

```bash
git add 3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py tests/sglang_ov2/test_top_level_import.py
git commit -m "feat(sglang/ov2): top-level LlavaOnevision2ForConditionalGeneration forward"
```

---

## Task 8: HF → sglang 权重名映射 + weight loader

**Files:**
- Modify: `3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py`
- Test: `tests/sglang_ov2/test_weight_name_map.py`

OV2 checkpoint 的 HF 命名约定要映射到 sglang 命名:
- `model.language_model.*` → `model.*`(去掉 `language_model`)
- `model.visual.*` → `visual.*`(去掉 `model.` 前缀)
- vision attention 的 `attn.qkv` (HF: 单 Linear `[3D, D]`) → sglang `VisionAttention.qkv_proj`(可直接复用,因 `use_qkv_parallel=True` 时 sglang 也用单个 ColumnParallelLinear)
- text qwen3 部分:`q_proj/k_proj/v_proj` → `qkv_proj`、`gate_proj/up_proj` → `gate_up_proj`(由 stacked_params_mapping 合并)

把映射逻辑抽成纯函数 `map_hf_name(name: str) -> str` 方便测试。

- [ ] **Step 1: 写测试**

```python
# tests/sglang_ov2/test_weight_name_map.py
from sglang.srt.models.llava_onevision_2 import map_hf_name

def test_strip_language_model_prefix():
    assert map_hf_name("model.language_model.layers.0.self_attn.q_proj.weight") \
        == "model.layers.0.self_attn.q_proj.weight"

def test_strip_visual_model_prefix():
    assert map_hf_name("model.visual.encoder.layers.0.layer_norm1.weight") \
        == "visual.encoder.layers.0.layer_norm1.weight"

def test_visual_qkv_rename():
    assert map_hf_name("model.visual.encoder.layers.0.attn.qkv.weight") \
        == "visual.encoder.layers.0.attn.qkv_proj.weight"

def test_lm_head_unchanged():
    assert map_hf_name("lm_head.weight") == "lm_head.weight"
```

- [ ] **Step 2: 跑测试,确认失败**

- [ ] **Step 3: 实现 `map_hf_name` 和 `load_weights`**

在 `llava_onevision_2.py` 末尾加:

```python
from sglang.srt.model_loader.weight_utils import default_weight_loader


def map_hf_name(name: str) -> str:
    """Translate HF OV2 weight key → sglang param key."""
    if name.startswith("model.language_model"):
        name = name.replace("model.language_model", "model", 1)
    if name.startswith("model.visual"):
        name = name.replace("model.visual", "visual", 1)
    # HF: attn.qkv (single Linear); sglang VisionAttention: attn.qkv_proj
    name = name.replace(".attn.qkv.", ".attn.qkv_proj.")
    return name


def _load_weights_into(model, weights):
    stacked_params_mapping = [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    params_dict = dict(model.named_parameters(remove_duplicate=False))
    for name, w in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        name = map_hf_name(name)
        matched = False
        for tgt, src, shard in stacked_params_mapping:
            if src in name and "visual" not in name:
                pname = name.replace(src, tgt)
                if pname.endswith(".bias") and pname not in params_dict:
                    matched = True
                    break
                param = params_dict[pname]
                param.weight_loader(param, w, shard)
                matched = True
                break
        if matched:
            continue
        if name.endswith(".bias") and name not in params_dict:
            continue
        if name not in params_dict:
            # Tolerate unused keys to ease checkpoint evolution; log via raise for now.
            raise KeyError(f"OV2 weight key not in model params: {name}")
        param = params_dict[name]
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, w)


# Attach as method
def _load_weights_method(self, weights):
    return _load_weights_into(self, weights)


LlavaOnevision2ForConditionalGeneration.load_weights = _load_weights_method
```

- [ ] **Step 4: 跑测试,确认 4 项通过**

```bash
pytest tests/sglang_ov2/test_weight_name_map.py -v
```

- [ ] **Step 5: 跑一次真权重加载冒烟测**

```python
# tests/sglang_ov2/test_weight_load_smoke.py
import os
import pytest
import torch

OV2 = os.environ.get(
    "OV2_PATH",
    "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct",
)

@pytest.mark.skipif(not os.path.isdir(OV2), reason="OV2 ckpt missing")
def test_load_full_checkpoint_into_model():
    from safetensors.torch import safe_open
    import glob
    from transformers import AutoConfig
    from sglang.srt.models.llava_onevision_2 import (
        LlavaOnevision2ForConditionalGeneration,
    )

    cfg = AutoConfig.from_pretrained(OV2, trust_remote_code=True)
    model = LlavaOnevision2ForConditionalGeneration(cfg).to(torch.bfloat16)

    def iter_weights():
        for f in sorted(glob.glob(os.path.join(OV2, "*.safetensors"))):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    yield k, sf.get_tensor(k)

    model.load_weights(iter_weights())
    # Should not raise.
```

```bash
pytest tests/sglang_ov2/test_weight_load_smoke.py -v
```

期望:PASS。**如果失败,排错优先级:**
1. Missing keys → 检查 vision tower 子模块的属性命名是否与 HF 完全一致
2. Shape mismatch on vision qkv → 检查 `VisionAttention(use_qkv_parallel=True)` 内部 `qkv_proj` 是否真是 `[3*D, D]`
3. Extra keys in checkpoint → 在 loader 里加白名单 skip

- [ ] **Step 6: Commit**

```bash
git add 3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py tests/sglang_ov2/test_weight_name_map.py tests/sglang_ov2/test_weight_load_smoke.py
git commit -m "feat(sglang/ov2): HF→sglang weight name map and load_weights"
```

---

## Task 9: 注册 `EntryClass`

**Files:**
- Modify: `3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py`

- [ ] **Step 1: 写测试**

```python
# tests/sglang_ov2/test_entryclass.py
def test_entryclass_present():
    from sglang.srt.models.llava_onevision_2 import (
        EntryClass, LlavaOnevision2ForConditionalGeneration,
    )
    assert LlavaOnevision2ForConditionalGeneration in EntryClass
```

- [ ] **Step 2: 跑测试,确认失败**

- [ ] **Step 3: 在文件末尾添加**

```python
EntryClass = [LlavaOnevision2ForConditionalGeneration]
```

- [ ] **Step 4: 跑测试,确认通过**

- [ ] **Step 5: Commit**

```bash
git add 3rdparty/sglang/python/sglang/srt/models/llava_onevision_2.py tests/sglang_ov2/test_entryclass.py
git commit -m "feat(sglang/ov2): register EntryClass for model auto-discovery"
```

---

## Task 10: OV2 multimodal processor (sglang)

**Files:**
- Create: `3rdparty/sglang/python/sglang/srt/multimodal/processors/llava_onevision_2.py`
- Test: `tests/sglang_ov2/test_processor.py`

把 OV2 HF processor 包成 sglang 风格,关键点:
1. `ATTR_NAME_TO_MODALITY["patch_positions"] = Modality.IMAGE`(让 base 类自动把 `patch_positions` 透到 `MultimodalDataItem.model_specific_data`)
2. OV2 processor 内置 Qwen2VL image_processor,接受 PIL `Image.Image`,无需手动 smart_resize
3. mrope 用 `MRotaryEmbedding.get_rope_index(model_type="qwen2_vl", spatial_merge_size=2, ...)`,与 OV1.5 完全一致

- [ ] **Step 1: 写测试**

```python
# tests/sglang_ov2/test_processor.py
import os
import io
import pytest
from PIL import Image

OV2 = os.environ.get(
    "OV2_PATH",
    "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct",
)

@pytest.mark.skipif(not os.path.isdir(OV2), reason="OV2 ckpt missing")
@pytest.mark.asyncio
async def test_process_single_image_carries_patch_positions():
    from transformers import AutoConfig, AutoProcessor
    from sglang.srt.multimodal.processors.llava_onevision_2 import (
        LlavaOnevision2_ImageProcessor,
    )
    from sglang.srt.managers.schedule_batch import Modality

    hf_cfg = AutoConfig.from_pretrained(OV2, trust_remote_code=True)
    hf_proc = AutoProcessor.from_pretrained(OV2, trust_remote_code=True)
    # minimal ServerArgs-ish object — use None and pass via kwargs
    sg_proc = LlavaOnevision2_ImageProcessor(
        hf_config=hf_cfg, server_args=None, _processor=hf_proc,
    )

    img = Image.new("RGB", (448, 448), color=(127, 127, 127))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    prompt = "<|vision_start|><|image_pad|><|vision_end|>describe this"

    class _Req:
        video_data = None

    out = await sg_proc.process_mm_data_async(
        image_data=[buf.getvalue()], input_text=prompt, request_obj=_Req(),
    )
    assert "mm_items" in out and len(out["mm_items"]) >= 1
    img_item = next(i for i in out["mm_items"] if i.modality == Modality.IMAGE)
    assert hasattr(img_item, "patch_positions")
    assert img_item.patch_positions.dim() == 2 and img_item.patch_positions.shape[1] == 3
    assert "mrope_positions" in out and out["mrope_positions"].shape[0] == 3
```

- [ ] **Step 2: 跑测试,确认失败**

- [ ] **Step 3: 实现 processor**

```python
# 3rdparty/sglang/python/sglang/srt/multimodal/processors/llava_onevision_2.py
import re
from typing import List, Union

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.llava_onevision_2 import (
    LlavaOnevision2ForConditionalGeneration,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


class LlavaOnevision2_ImageProcessor(SGLangBaseProcessor):
    models = [LlavaOnevision2ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *a, **kw):
        super().__init__(hf_config, server_args, _processor, *a, **kw)
        # OV2 reuses Qwen3 vision-token IDs; pull from config to be safe.
        self.IM_START_TOKEN_ID = getattr(hf_config, "vision_start_token_id", 151652)
        self.IM_END_TOKEN_ID = getattr(hf_config, "vision_end_token_id", 151653)
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id
        self.IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
        self.IMAGE_TOKEN_REGEX = re.compile(
            r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>",
        )
        # Register OV2-specific extra item so base.collect_mm_items pulls it.
        self.ATTR_NAME_TO_MODALITY["patch_positions"] = Modality.IMAGE
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=self.IMAGE_TOKEN_REGEX,
            video_token=self.VIDEO_TOKEN_ID,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *a,
        **kw,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )
        # OV2 processor handles PIL directly via Qwen2VLImageProcessor; no resize.
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens,
        )
        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
            vision_start_token_id=self.IM_START_TOKEN_ID,
            model_type="qwen2_vl",
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None,
            ),
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            second_per_grid_ts=getattr(ret, "second_per_grid_ts", None),
        )
        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.IM_TOKEN_ID,
            "video_token_id": self.VIDEO_TOKEN_ID,
            "mrope_positions": mrope_positions.squeeze(1),
            "mrope_position_delta": mrope_position_delta,
        }
```

- [ ] **Step 4: 跑测试,确认通过**

```bash
pip install pytest-asyncio   # 如未装
pytest tests/sglang_ov2/test_processor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add 3rdparty/sglang/python/sglang/srt/multimodal/processors/llava_onevision_2.py tests/sglang_ov2/test_processor.py
git commit -m "feat(sglang/ov2): multimodal processor with patch_positions passthrough"
```

---

## Task 11: 集成测试 — sglang Engine inference vs HF generate (TP=1)

**Files:**
- Test: `tests/sglang_ov2/test_inference_offline.py`

用 sglang offline `Engine` API 在单进程里启动 OV2,对同一 prompt+image,与 HF transformers `model.generate(do_sample=False, max_new_tokens=16)` 输出对齐。允许 ≤2 token 差异(fp16/bf16 数值容差)。

- [ ] **Step 1: 写测试**

```python
# tests/sglang_ov2/test_inference_offline.py
import os, io, pytest
from PIL import Image

OV2 = os.environ.get(
    "OV2_PATH",
    "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct",
)

@pytest.mark.skipif(not os.path.isdir(OV2), reason="OV2 ckpt missing")
@pytest.mark.gpu
def test_sglang_vs_hf_single_image_greedy():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    img = Image.new("RGB", (448, 448), color=(200, 100, 100))
    prompt = "<|vision_start|><|image_pad|><|vision_end|>What color is this image?"

    # HF reference (bf16, single GPU)
    proc = AutoProcessor.from_pretrained(OV2, trust_remote_code=True)
    hf = AutoModelForImageTextToText.from_pretrained(
        OV2, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).cuda().eval()
    enc = proc(text=[prompt], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        ref = hf.generate(**enc, max_new_tokens=16, do_sample=False)
    ref_tokens = ref[0, enc.input_ids.shape[1]:].tolist()
    del hf; torch.cuda.empty_cache()

    # sglang offline
    import sglang as sgl
    engine = sgl.Engine(
        model_path=OV2, trust_remote_code=True, dtype="bfloat16",
        tp_size=1, mem_fraction_static=0.6, enable_multimodal=True,
        skip_tokenizer_init=False,
    )
    buf = io.BytesIO(); img.save(buf, format="PNG")
    resp = engine.generate(
        prompt=prompt, image_data=[buf.getvalue()],
        sampling_params={"max_new_tokens": 16, "temperature": 0.0},
    )
    out_tokens = resp.get("output_ids") or resp["meta_info"]["output_token_ids"]
    engine.shutdown()

    # Allow ≤2-token divergence (fp16/bf16 + slightly different attention kernels).
    diff = sum(1 for a, b in zip(ref_tokens, out_tokens) if a != b)
    assert diff <= 2, f"sglang vs HF mismatch: ref={ref_tokens} out={out_tokens}"
```

- [ ] **Step 2: 跑测试**

```bash
pytest tests/sglang_ov2/test_inference_offline.py -v -s
```

期望:PASS。**如果失败,debug 顺序:**
1. sglang Engine 启动失败 → 看 stderr,大概率是 weight load shape mismatch、`pad_input_ids` 异常,或 `image_token_id` 不匹配
2. shape OK 但输出乱码 → vision RoPE freqs 喂给 `VisionAttention` 的格式不对(检查 cos/sin 维度);或 stage 顺序不对(`layernorm_pre` 位置)
3. 输出有意义但与 HF 完全不同 → 权重名映射漏了某些 key(检查 `test_weight_load_smoke` 是否有 warning)

- [ ] **Step 3: Commit**

```bash
git add tests/sglang_ov2/test_inference_offline.py
git commit -m "test(sglang/ov2): offline engine vs HF generate parity"
```

---

## Task 12: 集成测试 — TP=2 / TP=4 对齐

**Files:**
- Test: `tests/sglang_ov2/test_tp.py`

用 sglang server (不是 Engine,因 TP 需要多进程) 起 TP=2 / TP=4,跑同一 prompt,与 TP=1 输出对齐。

- [ ] **Step 1: 写脚本(tests + helpers)**

```python
# tests/sglang_ov2/test_tp.py
"""Run with: pytest -m gpu --tp 2/4 ; takes minutes per TP."""
import os, io, subprocess, time, json, signal, pytest
from PIL import Image
import requests

OV2 = os.environ.get(
    "OV2_PATH",
    "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct",
)

def _launch_server(tp, port):
    p = subprocess.Popen(
        ["python", "-m", "sglang.launch_server",
         "--model-path", OV2, "--trust-remote-code",
         "--dtype", "bfloat16", "--tp", str(tp), "--port", str(port),
         "--mem-fraction-static", "0.6"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    # poll /health for up to 5 min
    import urllib.error, urllib.request
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return p
        except Exception:
            time.sleep(2)
    p.kill(); raise RuntimeError("sglang server did not become healthy")

def _query(port, prompt, img_bytes):
    r = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "text": prompt,
            "image_data": [img_bytes.hex()],   # adjust to sglang API: base64 / file path
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 16},
        }, timeout=60,
    )
    r.raise_for_status()
    return r.json()

@pytest.mark.skipif(not os.path.isdir(OV2), reason="OV2 ckpt missing")
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.gpu
def test_tp_parity(tp):
    img = Image.new("RGB", (448, 448), color=(200, 100, 100))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe."
    port = 30000 + tp
    server = _launch_server(tp, port)
    try:
        out = _query(port, prompt, buf.getvalue())
        # store per-TP token ids in a temp file for cross-TP comparison
        tmp = f"/tmp/ov2_tp{tp}_tokens.json"
        with open(tmp, "w") as f: json.dump(out["meta_info"]["output_token_ids"], f)
    finally:
        server.send_signal(signal.SIGINT); server.wait(timeout=30)

def test_tp_outputs_match():
    """Run after the parametrized tests above produce the per-TP token files."""
    a = json.load(open("/tmp/ov2_tp1_tokens.json"))
    b = json.load(open("/tmp/ov2_tp2_tokens.json"))
    c = json.load(open("/tmp/ov2_tp4_tokens.json"))
    diff_ab = sum(1 for x, y in zip(a, b) if x != y)
    diff_ac = sum(1 for x, y in zip(a, c) if x != y)
    assert diff_ab <= 2 and diff_ac <= 2, f"TP mismatch: 1vs2={diff_ab}, 1vs4={diff_ac}"
```

- [ ] **Step 2: 跑测试**

```bash
pytest tests/sglang_ov2/test_tp.py -v -s
```

期望:全部 PASS。**如果 TP>1 输出与 TP=1 大幅偏差:**
1. 检查 vision tower 哪个 Linear 没用 `ColumnParallel/RowParallel`(纯 `nn.Linear` 在 TP 下会广播错误)
2. 检查 vision merger 的 `pos_emb_h/pos_emb_w`(我们用了 `nn.Embedding`,无 TP)是不是必须 replicate — 当前 `use_patch_position_encoding=False` 应该走不到,但若开启需要确认

- [ ] **Step 3: Commit**

```bash
git add tests/sglang_ov2/test_tp.py
git commit -m "test(sglang/ov2): TP=1/2/4 server parity"
```

---

## Task 13: README / docs 更新

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 在 Quick Start 之后追加 OV2 段**

```markdown
### LLaVA-OneVision-2 (8B) 接入(sglang inference)

```bash
# 1. 先按上面跑完 install.sh
bash install_ov2.sh   # 升级 transformers 到 5.7.0

# 2. 启 sglang server
python -m sglang.launch_server \
    --model-path /path/to/LLaVA-OneVision-2-8B-Instruct \
    --trust-remote-code --tp 1 --dtype bfloat16
```

GRPO 端到端训练 OV2 暂不在本里程碑范围。
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add OV2 sglang inference quick start"
```

---

## Self-Review

跑过一遍 spec(`docs/superpowers/specs/2026-05-28-ov2-sglang-integration-design.md`)与本 plan 的覆盖关系:

| Spec 章节 | 覆盖 Task |
|---|---|
| §3 环境 | Task 1 (`install_ov2.sh`) |
| §4 设计决策 1 (TP linear vision) | Task 5, 6 |
| §4 设计决策 2 (patch_positions 走 mm_items) | Task 7 (`get_image_feature`), Task 10 (`ATTR_NAME_TO_MODALITY`) |
| §4 设计决策 3 (trainer 零改动) | 不在本 plan,验收 §9 中也不验 |
| §5.1 new files | install_ov2.sh = Task 1, model.py = Task 3-9, processor.py = Task 10 |
| §5.2 model_config 白名单 | Task 2 |
| §5.3 暂不做 | 已排除 |
| §9 验收 gate 1 (静态 import) | Task 7 (`test_top_level_import`) + Task 9 (`test_entryclass`) |
| §9 验收 gate 2 (sglang server 起来) | 隐含在 Task 11 (`Engine` 路径) 和 Task 12 (subprocess server) |
| §9 验收 gate 3 (vs HF numerical) | Task 11 |
| §9 验收 gate 4 (TP=2/4) | Task 12 |

**类型一致性自查**:
- `OneVisionEncoderTransformer.forward(pixel_values, grid_thw, patch_positions)` — Task 6 定义,Task 7 调用 ✅
- `map_hf_name` 返回 `str` — Task 8 定义并测试 ✅
- `MultimodalDataItem.patch_positions` 透出 — Task 10 注册 `ATTR_NAME_TO_MODALITY`,Task 7 `get_image_feature` 通过 `item.patch_positions` 读取(由 `__getattr__` 走 `model_specific_data`)✅
- `build_cu_seqlens(grid_thw, total_patches, fixed_t, device)` — Task 4 定义,Task 6 调用 ✅

**已知潜在返工点**(列出来让 executor 心里有数):
1. Task 5 的 `OneVisionEncoderEmbeddings.patch_embedding` 假设是 `nn.Linear`;若 HF checkpoint 用 `nn.Conv2d`,Task 8 的 weight load smoke test 会先报 shape mismatch,届时回 Task 5 调整
2. Task 8 的 `_load_weights_into` 直接用 `params_dict[name]`;若 vision tower 命名细节与 HF 不完全一致(如 `encoder.layers.{i}` vs `blocks.{i}`),需要在 Task 6 调整命名或在 `map_hf_name` 加更多 rename 规则
3. Task 11 / 12 的 `engine.generate` / HTTP `/generate` 调用形参以 sglang 0.5.2 实际 API 为准(image_data 是 hex / base64 / file path 因版本而异);若调用报错,改成跑 `/v1/chat/completions` OpenAI 兼容端点

---

Plan complete and saved to `docs/superpowers/plans/2026-05-28-ov2-sglang-integration.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
