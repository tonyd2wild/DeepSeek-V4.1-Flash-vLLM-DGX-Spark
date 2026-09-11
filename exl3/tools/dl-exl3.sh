#!/bin/bash
# dl-exl3.sh: download ONLY the 40 changed EXL3 body shards (3-42) + small files of
# bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard into /var/tmp/models/DeepSeek-V4.1-Flash-EXL3-Pollard.
# Shards 1,2,43-48 are sha256-identical to the release and get hardlinked from ../DeepSeek-V4.1-Flash later.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
D=/var/tmp/models/DeepSeek-V4.1-Flash-EXL3-Pollard; mkdir -p "$D"
if pgrep -f "[h]f download bot-lab-21" >/dev/null; then echo "already-running"; exit 0; fi
setsid nohup ~/.local/bin/hf download bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard --local-dir "$D" --max-workers 8 \
  --include "model-0000[3-9]-of-00048.safetensors" --include "model-000[1-3][0-9]-of-00048.safetensors" \
  --include "model-0004[0-2]-of-00048.safetensors" --include "*.json" --include "tokenizer*" \
  --include "*.txt" --include "*.md" --include "recipe/*" \
  > ~/dl-exl3.log 2>&1 < /dev/null &
echo "started pid=$!"
