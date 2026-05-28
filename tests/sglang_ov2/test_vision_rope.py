import torch
from sglang.srt.models.llava_onevision_2 import VisionRotaryEmbedding


class _Cfg:
    hidden_size = 1024
    num_attention_heads = 16   # head_dim = 64, half = 32, unit = 2 -> t=8, h=12, w=12
    rope_theta = 10000.0


def test_split_4_6_6():
    rope = VisionRotaryEmbedding(_Cfg())
    assert rope.t_size == 8 and rope.h_size == 12 and rope.w_size == 12
    assert rope.head_dim == 64 and rope.half == 32


def test_forward_from_positions_shape():
    rope = VisionRotaryEmbedding(_Cfg())
    L = 16
    pp = torch.zeros(L, 3, dtype=torch.long)
    pp[:, 1] = torch.arange(L) % 4
    pp[:, 2] = torch.arange(L) % 4
    freqs = rope.forward_from_positions(pp)
    assert freqs.shape == (L, 32), freqs.shape


def test_match_hf_impl():
    import importlib
    import os
    import pytest
    OV2 = os.environ.get(
        "OV2_PATH",
        "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct",
    )
    if not os.path.isdir(OV2):
        pytest.skip(f"OV2 checkpoint not available at {OV2}; set OV2_PATH to enable")
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(OV2, trust_remote_code=True).vision_config
    # AutoConfig with trust_remote_code registers the bundled module under transformers_modules.*
    hf_mod = importlib.import_module(
        "transformers_modules.LLaVA_hyphen_OneVision_hyphen_2_hyphen_8B_hyphen_Instruct.modeling_llava_onevision2"
    )
    HFRope = hf_mod.VisionRotaryEmbedding
    ours = VisionRotaryEmbedding(cfg)
    theirs = HFRope(cfg)
    pp = torch.tensor([[0, 0, 0], [0, 1, 2], [1, 0, 3], [2, 3, 1]], dtype=torch.long)
    a = ours.forward_from_positions(pp)
    b = theirs.forward_from_positions(pp)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
