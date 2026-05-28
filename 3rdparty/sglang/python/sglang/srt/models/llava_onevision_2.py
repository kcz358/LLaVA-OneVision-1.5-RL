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
