#!/bin/bash
# exl3tp3a4 = TP3 try 4 (after the 3-reviewer look-over): head on Bluey (NFS client), Reddie rank-2 worker on local disk,
#   PYTORCH_CUDA_ALLOC_CONF gc threshold 0.6, max-num-batched-tokens 4096, EAGER, gmu 0.75, text-only, no DSpark, 300K.
#   Sidecars on every rank: memstat.sh (per-process memory log), flusher2.sh (droppable cache only); memguard2.sh on Asusi.
export EXP_NAME=exl3tp3a4 IMAGE=vllm-dsv41:exl3a PATCH_NAME=dsv41-exl3-tp3 MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-TP3 ENGRAM_LOCAL=1 GMU=0.75 MAXLEN=300000 SEQS=8 MAX_BATCHED=4096 EAGER=1 SPEC=none ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128'
echo "$EXP_NAME go $(date -u +%FT%TZ) head=bluey gmu=$GMU maxlen=$MAXLEN batched=$MAX_BATCHED eager=$EAGER model=$MODEL_DIR"
bash $HOME/boot_dsv41_tp3b.sh
echo "boot_dsv41_tp3b exit=$? $(date -u +%FT%TZ)"
