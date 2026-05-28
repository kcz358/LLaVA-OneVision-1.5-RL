import torch
from sglang.srt.models.llava_onevision_2 import (
    OneVisionEncoderEmbeddings,
    LlavaOnevision2VisionPatchMerger,
    OneVisionEncoderMLP,
    OneVisionEncoderBlock,
)


class _VisCfg:
    hidden_size = 1024
    patch_size = 14
    num_channels = 3
    image_size = 448
    layer_norm_eps = 1e-6
    layer_norm_type = "layer_norm"
    intermediate_size = 4096
    num_attention_heads = 16
    attention_dropout = 0.0
    hidden_act = "gelu"
    rope_theta = 10000.0
    spatial_merge_size = 2


def test_embeddings_shape():
    emb = OneVisionEncoderEmbeddings(_VisCfg())
    x = torch.randn(8, 3, 14, 14)
    out = emb(x)
    assert out.shape == (8, 1024)


def test_embeddings_accepts_flattened_input():
    emb = OneVisionEncoderEmbeddings(_VisCfg())
    # Flattened patches as HF .view(-1, C, P, P) does internally
    x = torch.randn(8, 3 * 14 * 14)
    out = emb(x)
    assert out.shape == (8, 1024)


def test_patch_merger_shape_no_pos_enc():
    merger = LlavaOnevision2VisionPatchMerger(
        dim=4096, context_dim=1024, spatial_merge_size=2,
        layer_norm_eps=1e-6, use_patch_position_encoding=False,
    )
    L = 16  # multiple of merge^2 = 4
    x = torch.randn(L, 1024)
    out = merger(x, patch_positions=None)
    assert out.shape == (L // 4, 4096)


def test_mlp_shape():
    mlp = OneVisionEncoderMLP(_VisCfg())
    out = mlp(torch.randn(8, 1024))
    assert out.shape == (8, 1024)


def test_block_forward_shape():
    # Force sdpa backend so the test runs on CPU (triton/fa3 require CUDA).
    blk = OneVisionEncoderBlock(_VisCfg(), qkv_backend="sdpa").eval()
    L = 8
    x = torch.randn(L, 1, 1024)  # [s, b, d] convention
    cu_seqlens = torch.tensor([0, L], dtype=torch.int32)
    head_dim = 1024 // 16  # 64
    cos = torch.randn(L, head_dim)
    sin = torch.randn(L, head_dim)
    with torch.no_grad():
        out = blk(x, cu_seqlens=cu_seqlens, position_embeddings=(cos, sin))
    assert out.shape == (L, 1, 1024)
