#!/bin/bash
# exl3tp3a5 = TP3 try 5: try 4 + ONE fix: patch set dsv41-exl3-tp3b, whose exl3_moe.py uploads each checkpoint
#   expert tensor to the GPU whole and slices it there (the as-is path queued host-to-GPU copies of strided CPU
#   slices faster than they completed: ~0.6 GiB of host memory per MoE layer in flight, ~25 GiB per rank at the wall).
#   Kept from try 4: head on Bluey (NFS client), Reddie rank-2 worker on local disk, alloc gc 0.6, batched 4096,
#   EAGER, gmu 0.75, text-only, no DSpark, 300K; sidecars memstat + flusher2 on every rank, memguard2 on Asusi.
export EXP_NAME=exl3tp3a5 IMAGE=vllm-dsv41:exl3a PATCH_NAME=dsv41-exl3-tp3b MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-TP3 ENGRAM_LOCAL=1 GMU=0.75 MAXLEN=300000 SEQS=8 MAX_BATCHED=4096 EAGER=1 SPEC=none ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128'
echo "$EXP_NAME go $(date -u +%FT%TZ) head=bluey patch=$PATCH_NAME gmu=$GMU maxlen=$MAXLEN batched=$MAX_BATCHED eager=$EAGER"
bash $HOME/boot_dsv41_tp3b.sh
echo "boot_dsv41_tp3b exit=$? $(date -u +%FT%TZ)"
