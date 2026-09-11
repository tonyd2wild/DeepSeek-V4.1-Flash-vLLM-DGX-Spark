#!/bin/bash
# exl3tp4b = the TP4 EXL3 lane: boot 10's serving config (4 Sparks, head Reddie, CUDA graphs, DSpark k=5, vision, 300K, gmu 0.80)
#   on the EXL3 3.5bpw checkpoint (DeepSeek-V4.1-Flash-EXL3-Pollard, 64 heads) instead of the release. Image vllm-dsv41:exl3a
#   (overlay5 + cuda-exl3) and patch set dsv41-exl3-tp3e: the cuda-exl3 files and the streaming vl_model.py matter here; the TP3-only
#   parts do nothing at TP4 (no virtual_heads_from in this config, the vocab splits by 4, 384 and 128 experts divide by 4).
#   Engram rows: boot 10's node-local copies (the tables are sha256-identical in both checkpoints).
#   No KV pin: every TP3 boot sized the KV pool from the tightest rank, which covers the wider expert slices on ranks 1-2.
export EXP_NAME=exl3tp4b IMAGE=vllm-dsv41:exl3a PATCH_NAME=dsv41-exl3-tp3e MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-Pollard ENGRAM_LOCAL=1 ENGRAM_LOCAL_HOST=/var/tmp/engram-local/DeepSeek-V4.1-Flash GMU=0.80 MAXLEN=300000 SEQS=8 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=0 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1'
echo "$EXP_NAME go $(date -u +%FT%TZ) head=reddie patch=$PATCH_NAME model=$MODEL_DIR gmu=$GMU maxlen=$MAXLEN eager=$EAGER spec=$SPEC k=$SPEC_K text_only=$TEXT_ONLY"
bash $HOME/boot_dsv41_tp4x.sh
echo "boot_dsv41_tp4x exit=$? $(date -u +%FT%TZ)"
