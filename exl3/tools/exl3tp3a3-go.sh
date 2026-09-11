#!/bin/bash
# exl3tp3a3 = TP3 try 3 (Tony 09-11): exl3tp3a + memory safety: gmu 0.75 (was 0.80), EAGER (no CUDA graphs / no
#   torch.compile workers), and flusher.sh on all three ranks during startup. Text-only, no DSpark, 300K.
export EXP_NAME=exl3tp3a3 IMAGE=vllm-dsv41:exl3a PATCH_NAME=dsv41-exl3-tp3 MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-TP3 ENGRAM_LOCAL=1 GMU=0.75 MAXLEN=300000 SEQS=8 EAGER=1 SPEC=none ENGRAM_DISK=1 TEXT_ONLY=1 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128'
echo "$EXP_NAME go $(date -u +%FT%TZ) gmu=$GMU maxlen=$MAXLEN eager=$EAGER spec=$SPEC text_only=$TEXT_ONLY model=$MODEL_DIR"
bash $HOME/boot_dsv41_tp3.sh
echo "boot_dsv41_tp3 exit=$? $(date -u +%FT%TZ)"
