"""Runtime monkey-patches for vendored 3rdparty/AReaL.

We don't edit 3rdparty/AReaL directly (it's gitignored to keep the upstream
tree pristine and easy to refresh). Instead, this module applies targeted
runtime patches and is imported once from the training entrypoints.

Currently patched:

- ``FSDPEngine._save_model_to_hf`` — the OV2 processor
  (``LlavaOnevision2Processor``, loaded via ``trust_remote_code``) does not
  inherit ``ProcessorMixin.save_pretrained``, so AReaL's HF-format save
  crashes at ``saver.freq_steps`` with::

      AttributeError: 'LlavaOnevision2Processor' object has no attribute
      'save_pretrained'.

  We wrap the ``processor.save_pretrained`` call in a try/except. On the
  AttributeError fallback path we copy the processor's source files (``.py``
  modules + tokenizer / processor / preprocessor configs) from the original
  model dir into the save path, so the saved checkpoint stays usable for
  HF/sglang inference with ``trust_remote_code=True``.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterable

import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from areal.engine.fsdp_engine import FSDPEngine

# Files we attempt to copy from the source model dir into the save path when
# ``processor.save_pretrained`` fails. Globs are matched non-recursively in the
# model dir's root, which mirrors the layout HF checkpoints ship in.
_PROCESSOR_FILE_GLOBS: tuple[str, ...] = (
    "*.py",  # custom processor / image_processor / video_processor modules
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
)


def _source_model_dir(model_config) -> str | None:
    """Best-effort lookup of the on-disk directory the model was loaded from.

    HF sets ``_name_or_path`` on the config to the path/repo id used in
    ``AutoConfig.from_pretrained``. For OV2 we always pass a local path
    (``actor.path``), so this resolves to a real directory; for remote-only
    runs it would return a repo id string and we just skip the copy.
    """
    path = getattr(model_config, "_name_or_path", None)
    if isinstance(path, str) and os.path.isdir(path):
        return path
    return None


def _copy_processor_files(src_dir: str, dst_dir: str, globs: Iterable[str]) -> list[str]:
    """Copy any files in ``src_dir`` matching ``globs`` into ``dst_dir``.

    Returns the basenames actually copied (for logging). Files that already
    exist at ``dst_dir`` (e.g. ``tokenizer.json`` written by
    ``tokenizer.save_pretrained``) are skipped to avoid surprising overrides.
    """
    import glob

    copied: list[str] = []
    for pattern in globs:
        for src in glob.glob(os.path.join(src_dir, pattern)):
            name = os.path.basename(src)
            dst = os.path.join(dst_dir, name)
            if os.path.exists(dst):
                continue
            try:
                shutil.copy2(src, dst)
                copied.append(name)
            except OSError:
                # Best-effort: ignore individual copy failures rather than
                # break the whole training run.
                continue
    return copied


def _patched_save_model_to_hf(self, path, tokenizer, processor):
    """Drop-in replacement for ``FSDPEngine._save_model_to_hf``.

    Identical to the upstream implementation except that ``processor`` save
    failures fall back to copying processor source files from the model's
    original directory.
    """
    if self.model is None:
        raise RuntimeError("Model not initialized")
    os.makedirs(path, exist_ok=True)

    # FSDP2 checkpoint saving
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(self.model, options=options)

    if dist.get_rank() == 0:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path, state_dict=state_dict)
        self.model_config.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        if processor is not None:
            try:
                processor.save_pretrained(path)
            except AttributeError as exc:
                # OV2's custom processor doesn't inherit ProcessorMixin.
                # Copy its source files from the original model dir so the
                # saved checkpoint stays loadable via trust_remote_code.
                src_dir = _source_model_dir(self.model_config)
                if src_dir is None:
                    self.logger.warning(
                        "processor.save_pretrained failed (%s) and "
                        "model_config._name_or_path is not a local dir; "
                        "the saved checkpoint will be missing processor files.",
                        exc,
                    )
                else:
                    copied = _copy_processor_files(
                        src_dir, path, _PROCESSOR_FILE_GLOBS
                    )
                    self.logger.warning(
                        "processor.save_pretrained failed (%s); copied %d "
                        "processor file(s) from %s instead: %s",
                        exc,
                        len(copied),
                        src_dir,
                        copied,
                    )

    dist.barrier(device_ids=[self.device.index])


def apply_ov2_processor_save_patch() -> None:
    """Idempotently install the FSDP saver patch."""
    if getattr(FSDPEngine._save_model_to_hf, "_ov2_patched", False):
        return
    _patched_save_model_to_hf._ov2_patched = True  # type: ignore[attr-defined]
    FSDPEngine._save_model_to_hf = _patched_save_model_to_hf  # type: ignore[assignment]
