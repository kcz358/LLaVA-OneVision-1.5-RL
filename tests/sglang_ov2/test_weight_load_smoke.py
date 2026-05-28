import glob
import os

import pytest
import torch

OV2 = os.environ.get(
    "OV2_PATH",
    "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct",
)


@pytest.mark.skipif(not os.path.isdir(OV2), reason="OV2 checkpoint missing")
def test_load_full_checkpoint_into_model():
    """End-to-end weight load: every OV2 checkpoint key must find a param,
    every param must receive at least one load call. Catches name-map and
    silent-skip regressions.
    """
    from safetensors import safe_open
    from transformers import AutoConfig
    from sglang.srt.models.llava_onevision_2 import (
        LlavaOnevision2ForConditionalGeneration,
    )
    from sglang.srt.models import llava_onevision_2 as _ov2_mod
    from sglang.srt.model_loader import weight_utils as _wu

    cfg = AutoConfig.from_pretrained(OV2, trust_remote_code=True)
    model = LlavaOnevision2ForConditionalGeneration(cfg).to(torch.bfloat16)

    # Tag each param param.data id -> param name, and intercept weight_loader.
    loaded_ids = set()
    orig_default = _wu.default_weight_loader
    orig_default_in_mod = _ov2_mod.default_weight_loader

    def tracking_default(param, loaded_weight):
        loaded_ids.add(id(param))
        return orig_default(param, loaded_weight)

    # For params with a bound weight_loader (TP linears, qkv_proj, gate_up_proj),
    # wrap the bound method in-place.
    for _, p in model.named_parameters(remove_duplicate=False):
        if hasattr(p, "weight_loader"):
            orig_wl = p.weight_loader

            def make_tracker(orig_loader, param_obj):
                def _wrapped(param, loaded_weight, *args, **kwargs):
                    loaded_ids.add(id(param))
                    return orig_loader(param, loaded_weight, *args, **kwargs)
                return _wrapped

            p.weight_loader = make_tracker(orig_wl, p)

    _wu.default_weight_loader = tracking_default
    _ov2_mod.default_weight_loader = tracking_default
    try:
        def iter_weights():
            for f in sorted(glob.glob(os.path.join(OV2, "*.safetensors"))):
                with safe_open(f, framework="pt") as sf:
                    for k in sf.keys():
                        yield k, sf.get_tensor(k)

        model.load_weights(iter_weights())
    finally:
        _wu.default_weight_loader = orig_default
        _ov2_mod.default_weight_loader = orig_default_in_mod

    # Every model param must have been touched by load (allow lm_head if tied).
    missing = [
        n for n, p in model.named_parameters(remove_duplicate=False)
        if id(p) not in loaded_ids
    ]
    # lm_head may legitimately remain uninitialized if config.tie_word_embeddings=True
    # (OV2 default is False, so on real checkpoint this list should be empty).
    if getattr(cfg, "tie_word_embeddings", False):
        missing = [n for n in missing if not n.startswith("lm_head")]
    assert not missing, f"Params not loaded: {missing[:10]}{'...' if len(missing) > 10 else ''}"
