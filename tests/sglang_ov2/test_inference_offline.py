"""Smoke test: sglang Engine boots OV2 and produces a sensible color response."""
import io
import os

import pytest
import torch
from PIL import Image

OV2 = os.environ.get(
    "OV2_PATH",
    "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct",
)


@pytest.mark.skipif(not os.path.isdir(OV2), reason="OV2 checkpoint missing")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sglang_engine_serves_ov2():
    """Boot sglang Engine on OV2, ask about a solid-red image, expect 'red' in answer."""
    import sglang as sgl

    img = Image.new("RGB", (448, 448), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    prompt = "<|vision_start|><|image_pad|><|vision_end|>The color of this image is"

    engine = sgl.Engine(
        model_path=OV2,
        trust_remote_code=True,
        dtype="bfloat16",
        tp_size=1,
        mem_fraction_static=0.6,
        enable_multimodal=True,
        skip_tokenizer_init=False,
    )
    try:
        resp = engine.generate(
            prompt=prompt,
            image_data=[buf.getvalue()],
            sampling_params={"max_new_tokens": 16, "temperature": 0.0},
        )
    finally:
        engine.shutdown()

    # sglang returns the completion text in resp["text"].
    text = resp.get("text", "")
    print(f"[sglang] completion: {text!r}")
    assert text, f"Empty completion from sglang Engine: {resp!r}"
    assert "red" in text.lower(), (
        f"Expected 'red' in completion for a red image, got: {text!r}"
    )
