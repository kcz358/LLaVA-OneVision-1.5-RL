from sglang.srt.models.llava_onevision_2 import map_hf_name


def test_strip_language_model_prefix():
    assert map_hf_name(
        "model.language_model.layers.0.self_attn.q_proj.weight"
    ) == "model.layers.0.self_attn.q_proj.weight"


def test_strip_visual_model_prefix():
    assert map_hf_name(
        "model.visual.encoder.layers.0.layer_norm1.weight"
    ) == "visual.encoder.layers.0.layer_norm1.weight"


def test_visual_qkv_rename_weight():
    assert map_hf_name(
        "model.visual.encoder.layers.0.self_attn.qkv.weight"
    ) == "visual.encoder.layers.0.attn.qkv_proj.weight"


def test_visual_qkv_rename_bias():
    assert map_hf_name(
        "model.visual.encoder.layers.5.self_attn.qkv.bias"
    ) == "visual.encoder.layers.5.attn.qkv_proj.bias"


def test_visual_proj_rename_weight():
    assert map_hf_name(
        "model.visual.encoder.layers.0.self_attn.proj.weight"
    ) == "visual.encoder.layers.0.attn.proj.weight"


def test_visual_embedding_pass_through():
    assert map_hf_name(
        "model.visual.embeddings.patch_embedding.weight"
    ) == "visual.embeddings.patch_embedding.weight"


def test_visual_merger_pass_through():
    assert map_hf_name(
        "model.visual.merger.mlp.0.weight"
    ) == "visual.merger.mlp.0.weight"


def test_text_q_proj_unchanged_by_visual_rename():
    """Text q_proj must NOT be renamed by the visual self_attn.qkv rule."""
    assert map_hf_name(
        "model.language_model.layers.0.self_attn.q_proj.weight"
    ) == "model.layers.0.self_attn.q_proj.weight"


def test_lm_head_unchanged():
    assert map_hf_name("lm_head.weight") == "lm_head.weight"


def test_text_embed_tokens_remap():
    assert map_hf_name(
        "model.language_model.embed_tokens.weight"
    ) == "model.embed_tokens.weight"


def test_text_norm_remap():
    assert map_hf_name(
        "model.language_model.norm.weight"
    ) == "model.norm.weight"
