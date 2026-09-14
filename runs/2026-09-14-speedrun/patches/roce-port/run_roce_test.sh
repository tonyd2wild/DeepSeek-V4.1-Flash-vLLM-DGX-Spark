#!/usr/bin/env bash
# 4-rank RoCEnante vs NCCL latency + correctness test, one container per node.
# vLLM MUST be stopped first: on Reddie/Spark4/Asusi a second CUDA context fails while vLLM is up.
# Usage on each node (same moment, rank 0 = head 192.168.192.2):  ./run_roce_test.sh <NODE_RANK> [preflight]
# Rank/IP map mirrors launch/dsv41-tp4.sh: 0=.2 1=.4 2=.3 3=.1
set -euo pipefail
R=${1:?node rank 0-3}; MODE=${2:-full}
IMAGE=${IMAGE:-vllm-dsv41:exl3a-roce}
DIR=$(cd "$(dirname "$0")" && pwd)
HCA=${B12X_ROCE_HCA:-rocep1s0f0}          # add ,roceP2p1s0f0 only if that function is ACTIVE with a GID at index 3
SPIN=${B12X_ROCE_SPIN_LIMIT:-20000000}    # ~1 us per poll

COMMON=(--rm --gpus all --network host --ipc host --shm-size 8g --memory 16g
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband
  -v "$DIR:/w:ro"
  -e B12X_ROCE_HCA="$HCA" -e B12X_ROCE_GID_INDEX=3 -e B12X_ROCE_SPIN_LIMIT="$SPIN"
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=192.168.192.0/24
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0
  -e NCCL_DEBUG=WARN)

if [ "$MODE" = "preflight" ]; then
  # Single node, no network peers: import, API version, HCAs, proxy .so, integrated-GPU check.
  exec docker run "${COMMON[@]}" --entrypoint python3 "$IMAGE" /w/test_roce_latency.py --preflight
fi

exec timeout 900 docker run "${COMMON[@]}" --entrypoint torchrun "$IMAGE" \
  --nnodes 4 --nproc-per-node 1 --node-rank "$R" \
  --master-addr 192.168.192.2 --master-port 29651 \
  /w/test_roce_latency.py --out /tmp/roce_result_rank${R}.json
