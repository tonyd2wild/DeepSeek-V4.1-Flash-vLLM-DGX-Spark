#!/bin/bash
# exl3tp3a11 = TP3 try 11: try 10 (CUDA graphs + vision, gmu 0.80, patch set tp3e) + ONE change: DSpark k=5, boot 10's setting
#   (probabilistic draft, block rejection, adaptive verification off, capture sizes every multiple of k and k+1 up to SEQS*(k+1)).
#   Try 8 failed on the drafter's expert-count check (128 experts, TP3); tp3e relaxes it when expert parallel and EPLB are off,
#   where every rank holds all 128 drafter experts with the intermediate split 3 ways.
export EXP_NAME=exl3tp3a11 IMAGE=vllm-dsv41:exl3a PATCH_NAME=dsv41-exl3-tp3e MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-TP3 ENGRAM_LOCAL=1 GMU=0.80 MAXLEN=300000 SEQS=8 MAX_BATCHED=4096 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=0 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton -e KAI_STACK_SAMPLER=1"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1 --mm-encoder-tp-mode data'
echo "$EXP_NAME go $(date -u +%FT%TZ) head=bluey patch=$PATCH_NAME gmu=$GMU maxlen=$MAXLEN batched=$MAX_BATCHED eager=$EAGER cg=$CUDAGRAPH_MODE spec=$SPEC k=$SPEC_K text_only=$TEXT_ONLY extra=$VLLM_EXTRA"
bash $HOME/boot_dsv41_tp3b.sh
echo "boot_dsv41_tp3b exit=$? $(date -u +%FT%TZ)"
