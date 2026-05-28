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
