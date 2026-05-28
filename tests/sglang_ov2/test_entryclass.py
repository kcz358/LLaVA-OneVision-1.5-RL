def test_entryclass_present():
    from sglang.srt.models.llava_onevision_2 import (
        EntryClass,
        LlavaOnevision2ForConditionalGeneration,
    )
    assert LlavaOnevision2ForConditionalGeneration in EntryClass
