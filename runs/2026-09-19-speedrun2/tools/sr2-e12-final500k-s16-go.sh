#!/bin/bash
# exl3tp4b = the TP4 EXL3 lane: boot 10's serving config (4 Sparks, head Reddie, CUDA graphs, DSpark k=5, vision, 300K, gmu 0.80)
#   on the EXL3 3.5bpw checkpoint (DeepSeek-V4.1-Flash-EXL3-Pollard, 64 heads) instead of the release. Image vllm-dsv41:exl3a
#   (overlay5 + cuda-exl3) and patch set dsv41-exl3-tp3e: the cuda-exl3 files and the streaming vl_model.py matter here; the TP3-only
#   parts do nothing at TP4 (no virtual_heads_from in this config, the vocab splits by 4, 384 and 128 experts divide by 4).
#   Engram rows: boot 10's node-local copies (the tables are sha256-identical in both checkpoints).
#   No KV pin: every TP3 boot sized the KV pool from the tightest rank, which covers the wider expert slices on ranks 1-2.
export EXP_NAME=exl3tp4b-ablit IMAGE=vllm-dsv41:exl3a PATCH_NAME=dsv41-exl3-tp3e MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-Pollard-Abliterated ENGRAM_LOCAL=1 ENGRAM_LOCAL_HOST=/var/tmp/engram-local/DeepSeek-V4.1-Flash GMU=0.80 MAXLEN=300000 SEQS=8 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=0 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1'
echo "$EXP_NAME go $(date -u +%FT%TZ) head=reddie patch=$PATCH_NAME model=$MODEL_DIR gmu=$GMU maxlen=$MAXLEN eager=$EAGER spec=$SPEC k=$SPEC_K text_only=$TEXT_ONLY"
# ---- speed run 2026-09-14 overrides for final-best
export IMAGE=vllm-dsv41:exl3b-roce
export PATCH_NAME=dsv41-exl3-sr1roce
export ENGRAM_THREADS=128
export NCCL_EXTRA="$NCCL_EXTRA -e NCCL_MAX_NCHANNELS=8 -e DSV41_ENGRAM_FAST=1 -e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB -e VLLM_ROCE_ALLGATHER_ENABLE=1 -e B12X_ROCE_HCA=rocep1s0f0 -e B12X_ROCE_GID_INDEX=3 -e B12X_ROCE_SPIN_LIMIT=300000000 -e B12X_ROCE_CACHE_DIR=/opt/b12x-roce/cache"
export PATCH_NAME=dsv41-exl3-sr2roce
export NCCL_EXTRA="$NCCL_EXTRA -e NCCL_MAX_NCHANNELS=8 -e DSV41_ENGRAM_FAST=1 -e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB -e VLLM_ROCE_ALLGATHER_ENABLE=1 -e B12X_ROCE_HCA=rocep1s0f0 -e B12X_ROCE_GID_INDEX=3 -e B12X_ROCE_SPIN_LIMIT=300000000 -e B12X_ROCE_CACHE_DIR=/opt/b12x-roce/cache -e DSV41_INDEXER_TP_SPLIT=1"
export NCCL_EXTRA="$NCCL_EXTRA -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
export EXP_NAME=exl3tp4b-ablit   # keep the compile/autotune cache
echo "speedrun final-best overrides applied"
# ---- speed run 2026-09-19 overrides for e06-exl3lim (base exl3tp4b-ablit-best-go.sh)
export IMAGE=vllm-dsv41:exl3b-roce-lim
export EXP_NAME=exl3tp4b-ablit   # keep the compile/autotune cache
echo "speedrun2 e06-exl3lim overrides applied"
# ---- speed run 2026-09-19 overrides for e09-sr4 (base sr2-e06-exl3lim-go.sh)
export PATCH_NAME=dsv41-exl3-sr4
export EXP_NAME=exl3tp4b-ablit   # keep the compile/autotune cache
echo "speedrun2 e09-sr4 overrides applied"
# ---- speed run 2026-09-19 overrides for e11-final500k (base sr2-e09-sr4-go.sh)
export MAXLEN=500000
export EXP_NAME=exl3tp4b-ablit   # keep the compile/autotune cache
echo "speedrun2 e11-final500k overrides applied"
# ---- speed run 2026-09-19 overrides for e12-final500k-s16 (base sr2-e11-final500k-go.sh)
export SEQS=16
export EXP_NAME=exl3tp4b-ablit   # keep the compile/autotune cache
echo "speedrun2 e12-final500k-s16 overrides applied"
bash $HOME/boot_dsv41_tp4x.sh
echo "boot_dsv41_tp4x exit=$? $(date -u +%FT%TZ)"
