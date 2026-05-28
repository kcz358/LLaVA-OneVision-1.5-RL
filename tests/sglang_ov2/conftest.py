"""Pytest setup for sglang OV2 unit tests on CPU.

- Forces import of ``sglang.srt.layers.quantization`` first to avoid a
  circular-import bug between ``linear`` and ``quantization.awq``.
- Initializes a single-process torch.distributed (gloo) group and sglang's
  model-parallel + dp_attention state so TP-aware layers (``ColumnParallelLinear``,
  ``RowParallelLinear``, ``VisionAttention``) work on CPU with tp_size=1.
"""

import os

import sglang.srt.layers.quantization  # noqa: F401  (bootstrap import order)

import torch.distributed as dist

from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers import dp_attention as _dp


def _init_single_proc_distributed() -> None:
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        init_distributed_environment(
            backend="gloo",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
        )
        try:
            initialize_model_parallel(tensor_model_parallel_size=1)
        except AssertionError:
            pass

    # Minimal dp_attention state for tp=1, no DP.
    _dp._ATTN_TP_RANK = 0
    _dp._ATTN_TP_SIZE = 1
    _dp._ATTN_DP_RANK = 0
    _dp._ATTN_DP_SIZE = 1
    _dp._LOCAL_ATTN_DP_RANK = 0
    _dp._LOCAL_ATTN_DP_SIZE = 1
    _dp._ENABLE_DP_ATTENTION_FLAG = False


_init_single_proc_distributed()
