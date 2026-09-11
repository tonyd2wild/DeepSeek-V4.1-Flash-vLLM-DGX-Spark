#!/bin/bash
# exl3tp3a10 = TP3 try 10: try 9 (CUDA graphs + vision) + ONE change: gpu-memory-utilization 0.80 (boot 10's value; was 0.75).
#   Try 9 serves with 18-20 GiB free per Spark but only 1.59 GiB of KV on the tightest rank (417,333 tokens), and DSpark's drafter
#   brings its own weights to every rank (its experts alone are about 1.9 GiB), so the budget goes up first.
#   Patch set dsv41-exl3-tp3e = tp3d + the drafter expert-count relaxation, a no-op here: the main model's 384 experts divide by 3
#   and take the original code path (this boot checks exactly that). Head Bluey, Reddie rank 2 on local disk, batched 4096, 300K.
export EXP_NAME=exl3tp3a10 IMAGE=vllm-dsv41:exl3a PATCH_NAME=dsv41-exl3-tp3e MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-TP3 ENGRAM_LOCAL=1 GMU=0.80 MAXLEN=300000 SEQS=8 MAX_BATCHED=4096 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=none ENGRAM_DISK=1 TEXT_ONLY=0 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton -e KAI_STACK_SAMPLER=1"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1 --mm-encoder-tp-mode data'
echo "$EXP_NAME go $(date -u +%FT%TZ) head=bluey patch=$PATCH_NAME gmu=$GMU maxlen=$MAXLEN batched=$MAX_BATCHED eager=$EAGER cg=$CUDAGRAPH_MODE spec=$SPEC text_only=$TEXT_ONLY extra=$VLLM_EXTRA"
bash $HOME/boot_dsv41_tp3b.sh
echo "boot_dsv41_tp3b exit=$? $(date -u +%FT%TZ)"
