import io
import os

import pytest
from PIL import Image

OV2 = os.environ.get(
    "OV2_PATH",
    "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct",
)


@pytest.mark.skipif(not os.path.isdir(OV2), reason="OV2 checkpoint missing")
def test_processor_imports():
    from sglang.srt.multimodal.processors.llava_onevision_2 import (
        LlavaOnevision2_ImageProcessor,
    )
    assert LlavaOnevision2_ImageProcessor is not None


@pytest.mark.skipif(not os.path.isdir(OV2), reason="OV2 checkpoint missing")
@pytest.mark.asyncio
async def test_process_single_image_carries_patch_positions():
    """Verify the OV2 sglang processor produces mm_items containing patch_positions
    and that mrope_positions are computed."""
    from transformers import AutoConfig, AutoProcessor
    from sglang.srt.multimodal.processors.llava_onevision_2 import (
        LlavaOnevision2_ImageProcessor,
    )
    from sglang.srt.managers.schedule_batch import Modality

    hf_cfg = AutoConfig.from_pretrained(OV2, trust_remote_code=True)
    hf_proc = AutoProcessor.from_pretrained(OV2, trust_remote_code=True)

    class _ServerArgs:
        # Test stub: production sglang typically has disable_fast_image_processor=False
        # which would route the fast image processor to CUDA (base_processor.py:232-237).
        # We set True to keep this unit test CPU-only.
        disable_fast_image_processor = True

    sg_proc = LlavaOnevision2_ImageProcessor(
        hf_config=hf_cfg,
        server_args=_ServerArgs(),
        _processor=hf_proc,
        transport_mode="default",
    )

    img = Image.new("RGB", (448, 448), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    prompt = "<|vision_start|><|image_pad|><|vision_end|>describe this"

    class _Req:
        video_data = None

    out = await sg_proc.process_mm_data_async(
        image_data=[buf.getvalue()],
        input_text=prompt,
        request_obj=_Req(),
    )

    # Check output structure.
    assert "mm_items" in out
    assert "input_ids" in out
    assert "mrope_positions" in out
    assert "mrope_position_delta" in out

    # Confirm patch_positions is on the image item.
    img_items = [it for it in out["mm_items"] if it.is_image()]
    assert len(img_items) >= 1
    item = img_items[0]
    assert hasattr(item, "patch_positions")
    pp = item.patch_positions
    assert pp.dim() == 2 and pp.shape[1] == 3, f"unexpected patch_positions shape {pp.shape}"

    # mrope_positions should be shape [3, seq_len] after squeeze(1).
    assert out["mrope_positions"].dim() == 2 and out["mrope_positions"].shape[0] == 3
