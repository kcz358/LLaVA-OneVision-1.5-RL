import torch
import pytest
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
    # 10 = 4 + 4 + 2 frames -> chunks of 16, 16, 8 patches
    assert cu.tolist() == [0, 16, 32, 40] and ms == 16


def test_total_mismatch_raises():
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    with pytest.raises(ValueError):
        build_cu_seqlens(grid_thw, total_patches=99, fixed_t=4, device="cpu")


def test_exact_multiple_split_no_remainder():
    grid_thw = torch.tensor([[8, 2, 2]], dtype=torch.int64)
    cu, ms = build_cu_seqlens(grid_thw, total_patches=32, fixed_t=4, device="cpu")
    # 8 = 4 + 4, no remainder -> chunks of 16, 16
    assert cu.tolist() == [0, 16, 32] and ms == 16


def test_fixed_t_none_skips_split():
    grid_thw = torch.tensor([[3, 2, 2]], dtype=torch.int64)
    cu, ms = build_cu_seqlens(grid_thw, total_patches=12, fixed_t=None, device="cpu")
    # No splitting -> single chunk of 12
    assert cu.tolist() == [0, 12] and ms == 12
