import torch
from sglang.srt.models.llava_onevision_2 import OneVisionEncoderTransformer


class _Cfg:
    hidden_size = 256        # use small dims for fast test
    num_attention_heads = 4  # head_dim=64 -> 16-divisible
    rope_theta = 10000.0
    patch_size = 14
    num_channels = 3
    image_size = 448
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
    max_position_embeddings = 8192


def test_forward_single_image():
    tower = OneVisionEncoderTransformer(_Cfg(), qkv_backend="sdpa").eval()
    grid = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    pv = torch.randn(16, 3, 14, 14)
    pp = torch.zeros(16, 3, dtype=torch.long)
    pp[:, 1] = torch.arange(16) // 4
    pp[:, 2] = torch.arange(16) % 4
    with torch.no_grad():
        out = tower(pv, grid_thw=grid, patch_positions=pp)
    # merger reduces by spatial_merge_size^2 = 4
    assert out.shape == (16 // 4, 256), out.shape


def test_forward_two_images_concat():
    """Verify multi-image concatenation: total tokens = sum(t*h*w) / merge^2."""
    tower = OneVisionEncoderTransformer(_Cfg(), qkv_backend="sdpa").eval()
    grid = torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.int64)
    total = 16 + 4
    pv = torch.randn(total, 3, 14, 14)
    pp = torch.zeros(total, 3, dtype=torch.long)
    with torch.no_grad():
        out = tower(pv, grid_thw=grid, patch_positions=pp)
    assert out.shape == (total // 4, 256), out.shape
