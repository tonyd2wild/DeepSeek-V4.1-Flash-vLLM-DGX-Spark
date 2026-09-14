#!/bin/bash
# sg_node.sh <rank> : start one SGLang TP rank of DeepSeek-V4.1-Flash (uncensored overlay) on this node.
# Rank 0 = Reddie (head, serves HTTP on :8000), 1 = Spark4, 2 = Asusi, 3 = Bluey. Upstream image
# lmsysorg/sglang:dev-dsv41 (Apache-2.0) plus our own mounted files from $PATCH_DIR (mounts.txt lines:
# "<file> <path under /sgl-workspace/sglang/python/sglang/>"). Knobs come from the environment (defaults below).
set -e
R=${1:?rank 0-3}
IMAGE=${IMAGE:-lmsysorg/sglang:dev-dsv41}
NAME=${NAME:-sglang_dsv41}
HEAD=${HEAD:-192.168.192.2}
PORT=${PORT:-8000}
DIST_PORT=${DIST_PORT:-29600}
MODEL=/models/DeepSeek-V4.1-Flash-Ablit
if [ "$R" = 0 ]; then HOST_MODEL=/var/tmp/models/DeepSeek-V4.1-Flash-Ablit; ENGRAM_HOST=$HOST_MODEL
else HOST_MODEL=/mnt/reddie-models/DeepSeek-V4.1-Flash-Ablit; ENGRAM_HOST=/var/tmp/engram-local/DeepSeek-V4.1-Flash; fi
PATCH_DIR=${PATCH_DIR:-$HOME/patches/sglang-dsv41-disk}
CTX=${CTX:-300000}
MEMFRAC=${MEMFRAC:-0.80}
CHUNK=${CHUNK:-2048}
SEQS=${SEQS:-8}
SPEC_BLOCK=${SPEC_BLOCK:-5}
IMAGES=${IMAGES:-4}
MAXTOK=${MAXTOK:-}          # empty = let SGLang size the KV pool from MEMFRAC
EXTRA_ARGS=${EXTRA_ARGS:-}
EXTRA_ENV=${EXTRA_ENV:-}    # extra "-e K=V" pairs for docker run
CACHE=${CACHE:-/var/tmp/sglang-cache}
mkdir -p "$CACHE" 2>/dev/null || true

MOUNTS=""
if [ -f "$PATCH_DIR/mounts.txt" ]; then
  while read -r f dst; do
    [ -z "$f" ] && continue; case "$f" in \#*) continue;; esac
    MOUNTS="$MOUNTS -v $PATCH_DIR/$f:/sgl-workspace/sglang/python/sglang/$dst:ro"
  done < "$PATCH_DIR/mounts.txt"
fi

docker rm -f "$NAME" > /dev/null 2>&1 || true
ARGS=(--model-path "$MODEL" --served-model-name deepseek-v4.1-flash
  --tp 4 --ep-size "${EP:-4}" --nnodes 4 --node-rank "$R" --dist-init-addr "$HEAD:$DIST_PORT"
  --host 0.0.0.0 --port "$PORT"
  --context-length "$CTX" --mem-fraction-static "$MEMFRAC"
  --chunked-prefill-size "$CHUNK" --max-running-requests "$SEQS" --cuda-graph-max-bs-decode "$SEQS"
  --min-free-slots-delay 1
  --speculative-algorithm DSPARK --speculative-draft-model-path "$MODEL" --speculative-dspark-block-size "$SPEC_BLOCK"
  --tool-call-parser deepseekv41 --reasoning-parser deepseek-v41
  --default-chat-template-kwargs '{"thinking": false}'
  --enable-multimodal --limit-mm-data-per-request "{\"image\": $IMAGES}")
[ -n "$MAXTOK" ] && ARGS+=(--max-total-tokens "$MAXTOK")
[ -n "${GRAPH_DECODE:-}" ] && ARGS+=(--cuda-graph-backend-decode "$GRAPH_DECODE")   # default: SGLang's full decode graph
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all --network host --ipc host \
  --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  -v "$HOST_MODEL:$MODEL:ro" -v "$ENGRAM_HOST:/engram:ro" -v "$CACHE:/root/.cache" $MOUNTS \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_IB_MERGE_NICS=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_MAX_NCHANNELS=8 -e NCCL_DEBUG=WARN \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e SGLANG_DSV41_ENGRAM_DISK=1 -e SGLANG_DSV41_ENGRAM_DIR=/engram -e SGLANG_DSV41_ENGRAM_THREADS=${ENGRAM_THREADS:-128} \
  -e SGLANG_DSV41_ENGRAM_ROWIO=/sgl-workspace/sglang/python/sglang/srt/layers/libdsv41_rowio.so -e DSV41_ENGRAM_IO_THREADS=${ENGRAM_IO_THREADS:-64} \
  $EXTRA_ENV \
  --entrypoint python3 "$IMAGE" -m sglang.launch_server "${ARGS[@]}" $EXTRA_ARGS
echo "started $NAME rank=$R on $(hostname) model=$HOST_MODEL engram=$ENGRAM_HOST ctx=$CTX memfrac=$MEMFRAC chunk=$CHUNK seqs=$SEQS maxtok=${MAXTOK:-auto} patches=$(wc -l < "$PATCH_DIR/mounts.txt" 2>/dev/null || echo 0)"
