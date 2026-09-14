# Source before the boot script on every node (the launcher injects $NCCL_EXTRA into docker run).
# Append the lines of mounts.roce.txt to the patch dir's mounts.txt and copy the 5 files from
# ./vllm/... (flattened basenames) into that patch dir. Image: vllm-dsv41:exl3a-roce (Dockerfile.roce).
#
# Spin limit = failure-detection latency (about 1 us per poll). Boot has lazy JIT (DeepGEMM ~100 s)
# between small collectives, so a rank can legitimately lag for many seconds; 20M (~20 s default)
# could poison a healthy boot. 300M is about 5 min: safe for boot, slow to notice a real wedge.
export NCCL_EXTRA="${NCCL_EXTRA:-} \
 -e VLLM_ENABLE_ROCE_ALLREDUCE=1 \
 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB \
 -e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB \
 -e VLLM_ROCE_ALLGATHER_ENABLE=1 \
 -e B12X_ROCE_HCA=rocep1s0f0 \
 -e B12X_ROCE_GID_INDEX=3 \
 -e B12X_ROCE_SPIN_LIMIT=300000000 \
 -e B12X_ROCE_CACHE_DIR=/opt/b12x-roce/cache"
# Rollback without rebuilding or unmounting: VLLM_ENABLE_ROCE_ALLREDUCE=0 (NCCL path is unchanged).
