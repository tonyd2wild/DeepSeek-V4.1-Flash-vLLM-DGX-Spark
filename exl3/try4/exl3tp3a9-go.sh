#!/bin/bash
# exl3tp3a9 = TP3 try 9: try 7 (CUDA graphs, no speculative decoding) + ONE change: vision on, boot 10's image settings
#   (4 images per request, 1 GB processor cache), with the vision encoder replicated on every rank (--mm-encoder-tp-mode data)
#   so it needs no TP3 split. First real run of the streaming vl_model.py's second loader pass (vision + aligner weights).
#   DSpark is off: try 8 showed its drafter MoE (128 experts) cannot split across 3 ranks as is.
#   Everything else as try 7: patch set dsv41-exl3-tp3d, head Bluey, Reddie rank 2 on local disk, gmu 0.75, batched 4096, 300K.
export EXP_NAME=exl3tp3a9 IMAGE=vllm-dsv41:exl3a PATCH_NAME=dsv41-exl3-tp3d MODEL_DIR=DeepSeek-V4.1-Flash-EXL3-TP3 ENGRAM_LOCAL=1 GMU=0.75 MAXLEN=300000 SEQS=8 MAX_BATCHED=4096 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=none ENGRAM_DISK=1 TEXT_ONLY=0 THINKING=false PARSERS=1 RUST_FE=0
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton -e KAI_STACK_SAMPLER=1"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1 --mm-encoder-tp-mode data'
echo "$EXP_NAME go $(date -u +%FT%TZ) head=bluey patch=$PATCH_NAME gmu=$GMU maxlen=$MAXLEN batched=$MAX_BATCHED eager=$EAGER cg=$CUDAGRAPH_MODE spec=$SPEC text_only=$TEXT_ONLY extra=$VLLM_EXTRA"
bash $HOME/boot_dsv41_tp3b.sh
echo "boot_dsv41_tp3b exit=$? $(date -u +%FT%TZ)"
