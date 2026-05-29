#!/bin/bash
# OV2 incremental env setup. Prereq: `bash install.sh` completed.
# Upgrades transformers to 5.7.0 (the version OV2 checkpoint targets).
# transformers 5.7.0 pulls in huggingface_hub>=1.0, tokenizers>=0.22, numpy>=2 and related deps.
set -e
uv pip install transformers==5.7.0 openai==2.2.0
echo "OV2 env ready. Verify with:"
echo "  python -c 'from transformers import AutoConfig; AutoConfig.from_pretrained(\"<path-to-ov2-ckpt>\", trust_remote_code=True)'"
