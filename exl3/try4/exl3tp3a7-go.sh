#!/bin/bash
# exl3tp3a7 = TP3 try 7: try 6 (the first TP3 boot that came up, 2026-09-11 13:40 ET) + ONE change: CUDA graphs
#   (EAGER=0, FULL_AND_PIECEWISE, decode capture sizes 1..SEQS). Everything else exactly as try 6: patch set
#   dsv41-exl3-tp3d (vl_model.py streaming weight load + stack sampler), head Bluey, Asusi rank 1 and Bluey rank 0 on NFS,
#   Reddie rank 2 on local disk, alloc gc 0.6, gmu 0.75, batched 4096, text-only, no DSpark, 300K.
export EXP_NAME=exl3tp3a7 IMAGE=vllm-dsv41:exl3a PATCH_NAME=dsv41-exl3-tp3d MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-TP3 ENGRAM_LOCAL=1 GMU=0.75 MAXLEN=300000 SEQS=8 MAX_BATCHED=4096 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=none ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton -e KAI_STACK_SAMPLER=1"
export VLLM_EXTRA='--block-size 128'
echo "$EXP_NAME go $(date -u +%FT%TZ) head=bluey patch=$PATCH_NAME gmu=$GMU maxlen=$MAXLEN batched=$MAX_BATCHED eager=$EAGER cg=$CUDAGRAPH_MODE"
bash $HOME/boot_dsv41_tp3b.sh
echo "boot_dsv41_tp3b exit=$? $(date -u +%FT%TZ)"
