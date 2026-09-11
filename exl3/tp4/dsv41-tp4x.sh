#!/usr/bin/env bash
# dsv41-tp4x.sh (Tech2Wild/Kai 2026-09-11): dsv41-tp4.sh + MODEL_DIR override (default the EXL3 build),
#   CUDA_EXL3_MODEL_PATH for cuda-exl3, absolute mount targets. Same ranks: 0 Reddie, 1 Spark4, 2 Asusi, 3 Bluey.
#   Uneven EXL3 expert split at TP4 (512/640/640/512): pin KV with VLLM_EXTRA --kv-cache-memory-bytes (see exl3tp4a-go.sh).
# dsv41-tp4.sh <rank>  — DeepSeek-V4.1-Flash TP4 across the 4 DGX Sparks (vLLM dsv41-feat branch).
# Tech2Wild/Kai 2026-09-10. Derived from exp-glm53-tp4-b.sh (same NCCL/fabric env, same rank map).
#
# Rank map (worker-first launch: 3 Bluey, 2 Asusi, 1 Spark4, then 0 Reddie head):
#   0 Reddie 192.168.192.2 (head, model LOCAL /var/tmp/models)   1 Spark4 .4   2 Asusi .3   3 Bluey .1  (NFS /mnt/reddie-models)
# Knobs (export before running, SAME on all four):
#   IMAGE        default vllm-dsv41:overlay1 (fallback: vllm/vllm-openai:deepseekv41-flash-0909-arm64)
#   EXP_NAME     label + per-experiment compile cache dir (default boot1)
#   GMU          --gpu-memory-utilization (default 0.80)
#   MAXLEN       --max-model-len (default 131072)
#   SEQS         --max-num-seqs (default 8)
#   MAX_BATCHED  --max-num-batched-tokens (default 8192)
#   EAGER        1 => --enforce-eager (default 1). 0 => CUDA graphs, CUDAGRAPH_MODE (default FULL_AND_PIECEWISE)
#                   EAGER=0 with ENGRAM_DISK=1 NEEDS the prestage patch (engram.py + nvidia/model_state.py)
#   CG_SIZES     capture sizes, comma list. Default: dspark -> every multiple of k and k+1 up to SEQS*(k+1)
#                (each decode batch has an exact FULL graph: no padded rows, FlashInfer #5015); none -> 1..SEQS
#   SPEC_ADAPT   adaptive verification (default false: it forces varlen FULL graphs = padded rows, #5015)
#   SPEC         none (default) | dspark  (dspark = recipe config, k=5)
#   ENGRAM_DISK  1 (default) => bind-mount the disk-backed Engram patch + DSV41_ENGRAM_DISK=1
#   TEXT_ONLY    1 (default) => --language-model-only
#   THINKING     false (default) => --default-chat-template-kwargs '{"thinking": false}'
#   PARSERS      0 (default) | 1 => --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 (needs the Rust parser ext)
#   RUST_FE      0 (default) => VLLM_USE_RUST_FRONTEND=0
#   VLLM_EXTRA   extra vllm serve args;  NCCL_EXTRA extra "-e K=V" docker env pairs
set -euo pipefail
NODE_RANK="${1:?usage: dsv41-tp4x.sh <0|1|2|3>}"

IMAGE="${IMAGE:-vllm-dsv41:overlay1}"
EXP_NAME="${EXP_NAME:-boot1}"
GMU="${GMU:-0.80}"
MAXLEN="${MAXLEN:-131072}"
SEQS="${SEQS:-8}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
EAGER="${EAGER:-1}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
CG_SIZES="${CG_SIZES:-}"
SPEC="${SPEC:-none}"
ENGRAM_DISK="${ENGRAM_DISK:-1}"
TEXT_ONLY="${TEXT_ONLY:-1}"
THINKING="${THINKING:-false}"
PARSERS="${PARSERS:-0}"
RUST_FE="${RUST_FE:-0}"
VLLM_EXTRA="${VLLM_EXTRA:-}"
NCCL_EXTRA="${NCCL_EXTRA:-}"

NAME="vllm_dsv41"
MODEL_DIR="${MODEL_DIR:-DeepSeek-V4.1-Flash-EXL3-Pollard}"
CACHE_HOST_PATH="/var/tmp/dsv41-vllm-cache"
TOK_ARGS=""
case "$IMAGE" in *deepseekv41-flash-0909*) TOK_ARGS="--tokenizer-mode deepseek_v41" ;; esac
# (overlay/branch image: tokenizer-mode auto-resolves to deepseek_v41 from model_type; its CLI choices list lacks the literal)
SITE="/usr/local/lib/python3.12/dist-packages/vllm"
HEAD_IP="192.168.192.2"; MPORT="29541"; PORT="8000"

case "$NODE_RANK" in
  0) HOST_IP=192.168.192.2; HEADLESS="";           MODEL_HOST="/var/tmp/models/$MODEL_DIR" ;;
  1) HOST_IP=192.168.192.4; HEADLESS="--headless"; MODEL_HOST="/mnt/reddie-models/$MODEL_DIR" ;;
  2) HOST_IP=192.168.192.3; HEADLESS="--headless"; MODEL_HOST="/mnt/reddie-models/$MODEL_DIR" ;;
  3) HOST_IP=192.168.192.1; HEADLESS="--headless"; MODEL_HOST="/mnt/reddie-models/$MODEL_DIR" ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

# ---- preflight ----
test -f "$MODEL_HOST/config.json" || { echo "MODEL MISSING at $MODEL_HOST" >&2; exit 3; }
test -f "$MODEL_HOST/model-00048-of-00048.safetensors" || { echo "MODEL INCOMPLETE at $MODEL_HOST (shard 48 missing)" >&2; exit 3; }
# PATCH_NAME is resolved per node (each rank has its own $HOME); boot_dsv41.sh forwards it.
PATCH_DIR="${PATCH_DIR:-$HOME/patches/${PATCH_NAME:-dsv41-boot3}}"
PATCH_MOUNTS=""
if [ -f "$PATCH_DIR/mounts.txt" ]; then
  # manifest: "<file> <site-relative path>" per line; ENGRAM_DISK=0 skips the engram files
  while read -r f rel; do
    [ -z "$f" ] && continue
    if [ "$ENGRAM_DISK" != "1" ] && { [ "$f" = "engram.py" ] || [ "$f" = "weight_utils.py" ] || [ "$f" = "model_state.py" ]; }; then continue; fi
    test -f "$PATCH_DIR/$f" || { echo "PATCH FILE MISSING: $PATCH_DIR/$f" >&2; exit 3; }
    case "$rel" in /*) tgt="$rel" ;; *) tgt="$SITE/$rel" ;; esac; PATCH_MOUNTS="$PATCH_MOUNTS -v $PATCH_DIR/$f:$tgt:ro"
  done < "$PATCH_DIR/mounts.txt"
else
  echo "no mounts.txt in $PATCH_DIR" >&2; exit 3
fi
if [ "$ENGRAM_DISK" = "1" ]; then
  ENGRAM_ENV="-e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=${ENGRAM_THREADS:-32} -e DSV41_ENGRAM_DISK_CHUNK=${ENGRAM_CHUNK:-16}"
else
  ENGRAM_ENV="-e DSV41_ENGRAM_DISK=0"
fi
# Node-local Engram rows (tools/engram_local.py): mounted only where this node holds a copy.
# engram.py checks the copy's row range against this rank's rows and otherwise reads MODEL_HOST.
ENGRAM_LOCAL_HOST="${ENGRAM_LOCAL_HOST:-/var/tmp/engram-local/$MODEL_DIR}"
ENGRAM_LOCAL_MOUNT=""
if [ "$ENGRAM_DISK" = "1" ] && [ "${ENGRAM_LOCAL:-0}" = "1" ] && [ -f "$ENGRAM_LOCAL_HOST/engram-local.json" ]; then
  ENGRAM_LOCAL_MOUNT="-v $ENGRAM_LOCAL_HOST:/engram-local:ro"
  ENGRAM_ENV="$ENGRAM_ENV -e DSV41_ENGRAM_DIR=/engram-local"
fi
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
AVAIL_GB=$(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1048576 ))
[ "$AVAIL_GB" -ge 100 ] || { echo "MemAvailable ${AVAIL_GB} GiB < 100 GiB, refusing to boot" >&2; exit 4; }

GRAPH_ENV=""
if [ "$EAGER" = "1" ]; then
  GRAPH_ARGS=(--enforce-eager)
else
  if [ "$ENGRAM_DISK" = "1" ] && ! grep -q '^model_state.py ' "$PATCH_DIR/mounts.txt"; then
    echo "EAGER=0 + ENGRAM_DISK=1 needs the Engram prestage patch (model_state.py in mounts.txt)" >&2; exit 3
  fi
  if [ -z "$CG_SIZES" ]; then
    if [ "$SPEC" = "dspark" ]; then
      K="${SPEC_K:-5}"   # target decode = k+1 tokens/req, DSpark draft = k tokens/req
      CG_SIZES=$( { seq "$K" "$K" $((K * SEQS)); seq $((K + 1)) $((K + 1)) $(((K + 1) * SEQS)); } | sort -n -u | paste -sd, - )
    else
      CG_SIZES=$(seq 1 "$SEQS" | paste -sd, -)
    fi
  fi
  # Array: the JSON holds '[' ']' and must not be word-split or globbed.
  GRAPH_ARGS=(--compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"cudagraph_capture_sizes\":[$CG_SIZES]}")
  # Explicit on every node: eager_break_during_capture binds at model import.
  GRAPH_ENV="-e VLLM_USE_BREAKABLE_CUDAGRAPH=1"
fi
# Adaptive verification needs CUDA graphs AND forces varlen FULL decode graphs
# (model_runner.py:640) = padded rows on SM12x sparse MLA (FlashInfer #5015): opt-in.
if [ "$EAGER" = "1" ]; then SPEC_ADAPT=false; else SPEC_ADAPT="${SPEC_ADAPT:-false}"; fi
if [ "$SPEC" = "dspark" ]; then
  SPEC_ARGS="--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_K:-5},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":${SPEC_ADAPT}}"
else SPEC_ARGS=""; fi
if [ "$TEXT_ONLY" = "1" ]; then TEXT_ARGS="--language-model-only"; else TEXT_ARGS=""; fi
if [ "$PARSERS" = "1" ]; then PARSER_ARGS="--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41"; else PARSER_ARGS=""; fi

# shellcheck disable=SC2086
docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  --oom-score-adj 500 \
  -v "$MODEL_HOST:/models/$MODEL_DIR:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  $PATCH_MOUNTS $ENGRAM_LOCAL_MOUNT \
  -e VLLM_HOST_IP=$HOST_IP -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_CACHE_ROOT="/cache/vllm-$EXP_NAME" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_EXL3_MODEL_PATH=/models/$MODEL_DIR -e VLLM_USE_RUST_FRONTEND=$RUST_FE -e VLLM_HAS_FLASHINFER_CUBIN=1 \
  $ENGRAM_ENV $GRAPH_ENV \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  $NCCL_EXTRA \
  "$IMAGE" \
    "/models/$MODEL_DIR" \
    --served-model-name deepseek-v4.1-flash --host 0.0.0.0 --port "$PORT" \
    $TOK_ARGS \
    --tensor-parallel-size 4 --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
    --max-num-seqs "$SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
    --engram-config '{"cpu_offload": false}' \
    --default-chat-template-kwargs "{\"thinking\": $THINKING}" \
    $TEXT_ARGS $PARSER_ARGS $SPEC_ARGS "${GRAPH_ARGS[@]}" \
    --distributed-executor-backend mp --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" $HEADLESS $VLLM_EXTRA

echo "launched $NAME rank=$NODE_RANK exp=$EXP_NAME image=$IMAGE patches=$PATCH_DIR gmu=$GMU maxlen=$MAXLEN seqs=$SEQS eager=$EAGER cg=${CUDAGRAPH_MODE}[${CG_SIZES}] spec=$SPEC adapt=$SPEC_ADAPT engram_disk=$ENGRAM_DISK engram_local=${ENGRAM_LOCAL_MOUNT:+yes} text_only=$TEXT_ONLY avail=${AVAIL_GB}GiB"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited" >&2; docker logs --tail 40 "$NAME" >&2; exit 1; }
