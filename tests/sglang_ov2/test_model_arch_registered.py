from sglang.srt.configs.model_config import is_multimodal_model


def test_ov2_arch_recognized_as_multimodal():
    assert is_multimodal_model(["LlavaOnevision2ForConditionalGeneration"]) is True
