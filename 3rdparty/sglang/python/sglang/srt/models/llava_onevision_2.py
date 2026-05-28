"""Inference-only LLaVA-OneVision-2 model for sglang."""
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix


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
        """patch_positions: [L, 3] long -> freqs: [L, half] float32."""
        pp = patch_positions.to(self.inv_freq_t.device)
        t, h, w = pp[:, 0].float(), pp[:, 1].float(), pp[:, 2].float()
        freqs_t = t.unsqueeze(-1) * self.inv_freq_t.unsqueeze(0)  # [L, t_size]
        freqs_h = h.unsqueeze(-1) * self.inv_freq_h.unsqueeze(0)
        freqs_w = w.unsqueeze(-1) * self.inv_freq_w.unsqueeze(0)
        return torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)  # [L, half]


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


def _get_norm(config):
    """Return LayerNorm or RMSNorm based on `config.layer_norm_type`.

    Raises ValueError on unknown values (defensive — typos like 'rmsnorm'
    silently fell through to LayerNorm before).
    """
    norm_type = getattr(config, "layer_norm_type", "layer_norm")
    if norm_type == "layer_norm":
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    if norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    raise ValueError(f"Unknown layer_norm_type: {norm_type!r}")


class OneVisionEncoderEmbeddings(nn.Module):
    """Patch embedding via Conv2d (kernel_size=stride=patch_size, bias=False).

    Mirrors HF ``OneVisionEncoderEmbeddings`` so the checkpoint
    ``patch_embedding.weight`` loads directly.

    Input  : [N, C, P, P] or [N, C*P*P]
    Output : [N, hidden_size]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.in_channels = config.num_channels
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.patch_embedding(
            hidden_states.to(dtype=target_dtype)
        ).view(-1, self.embed_dim)
        return hidden_states


class LlavaOnevision2VisionPatchMerger(nn.Module):
    """LayerNorm + 2x Linear (TP-aware), merges spatial_merge_size^2 patches into one token.

    Optionally adds H/W absolute position embeddings (controlled by
    ``use_patch_position_encoding``).
    """

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
        self.ln_q = nn.LayerNorm(context_dim, eps=layer_norm_eps)
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.0", prefix),
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.2", prefix),
                ),
            ]
        )
        if use_patch_position_encoding:
            if patch_position_encoding_type != "absolute":
                raise ValueError(
                    f"Unknown patch_position_encoding_type: {patch_position_encoding_type}"
                )
            self.pos_emb_h = nn.Embedding(max_position_embeddings, dim)
            self.pos_emb_w = nn.Embedding(max_position_embeddings, dim)

    def forward(
        self,
        x: torch.Tensor,
        patch_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if patch_positions is not None and patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)
        x = self.ln_q(x).view(-1, self.hidden_size)
        fc1, act, fc2 = self.mlp
        x, _ = fc1(x)
        x = act(x)
        x, _ = fc2(x)
        if self.use_patch_position_encoding and patch_positions is not None:
            pp = patch_positions.view(-1, self.spatial_merge_size ** 2, 3)[:, 0, :]
            pp = (pp // self.spatial_merge_size).long()
            x = x + self.pos_emb_h(pp[:, 1]) + self.pos_emb_w(pp[:, 2])
        return x


class OneVisionEncoderMLP(nn.Module):
    """Siglip-style MLP via TP linear (ColumnParallel + RowParallel)."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class OneVisionEncoderBlock(nn.Module):
    """Pre-norm + VisionAttention + pre-norm + MLP (OV2 vision block).

    Convention (matches OV1.5 ``RiceBlock``): ``x`` carries shape ``[s, b, d]``.
    Inside, we rearrange to ``[b, s, d]`` for VisionAttention then back.

    ``qkv_backend`` defaults to ``None`` so ``VisionAttention`` auto-selects
    based on platform (fa3/triton on CUDA Hopper+, sdpa on CPU). Production
    callers can override.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        qkv_backend: Optional[str] = None,
    ):
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
            qkv_backend=qkv_backend,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = OneVisionEncoderMLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        h = self.layer_norm1(x)
        h = rearrange(h, "s b ... -> b s ...")
        attn = self.attn(
            h, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings,
        )
        attn = rearrange(attn, "b s ... -> s b ...")
        x = x + attn
        x = x + self.mlp(self.layer_norm2(x))
        return x
