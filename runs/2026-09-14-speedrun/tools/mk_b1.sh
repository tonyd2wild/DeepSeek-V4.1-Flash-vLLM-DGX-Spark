#!/bin/bash
# mk_b1.sh (root on Reddie): the "b1" ladder. Base b1 = E02 (ch8 + Engram fast) + E09 (128 read threads) + E19 (RoCE).
# Each b1-* label = b1 plus ONE change. Also builds patch set dsv41-exl3-sr2roce (sr1roce + the indexer TP-split file)
# on all 4 nodes, then replaces the rest of the dynamic queue (e14-idxsplit, already running, is left alone).
set -e
D=/var/tmp/boot-results/speedrun
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
ROCE='-e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB -e VLLM_ROCE_ALLGATHER_ENABLE=1 -e B12X_ROCE_HCA=rocep1s0f0 -e B12X_ROCE_GID_INDEX=3 -e B12X_ROCE_SPIN_LIMIT=300000000 -e B12X_ROCE_CACHE_DIR=/opt/b12x-roce/cache'
mk() {  # mk <label> <patch-set> <extra docker -e pairs> [export lines...]
  local L=$1 P=$2 X=$3; shift 3
  { echo "export IMAGE=vllm-dsv41:exl3b-roce"; echo "export PATCH_NAME=$P"; echo "export ENGRAM_THREADS=128"
    for l in "$@"; do echo "$l"; done
    echo "export NCCL_EXTRA=\"\$NCCL_EXTRA -e NCCL_MAX_NCHANNELS=8 -e DSV41_ENGRAM_FAST=1 $ROCE$X\""; } | bash /root/sr_mkgo.sh $L | tail -1
}
mk b1-best       dsv41-exl3-sr1roce ""
mk b1-idxsplit   dsv41-exl3-sr2roce " -e DSV41_INDEXER_TP_SPLIT=1"
mk b1-k4         dsv41-exl3-sr1roce "" "export SPEC_K=4"
mk b1-mb16k      dsv41-exl3-sr1roce "" "export MAX_BATCHED=16384"
mk b1-idxlogits  dsv41-exl3-sr1roce " -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1024"
mk b1-ctx1m      dsv41-exl3-sr1roce "" "export MAXLEN=1048576"
mk b1-noexpseg   dsv41-exl3-sr1roce " -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
mk b1-gmu85      dsv41-exl3-sr1roce "" "export GMU=0.85"
mk b1-shexp      dsv41-exl3-sr1roce " -e VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=8192"
# patch set sr2roce = sr1roce + indexer TP-split (md5 f87bc894) on all 4 nodes
MK='cp -a ~/patches/dsv41-exl3-sr1roce ~/patches/dsv41-exl3-sr2roce && cp ~/patches/dsv41-exl3-sr2/sparse_attn_indexer.py ~/patches/dsv41-exl3-sr2roce/sparse_attn_indexer.py && echo $(hostname) $(md5sum ~/patches/dsv41-exl3-sr2roce/sparse_attn_indexer.py | cut -c1-8) $(wc -l < ~/patches/dsv41-exl3-sr2roce/mounts.txt) mounts'
[ -d /home/tonyspark2/patches/dsv41-exl3-sr2roce ] || sudo -u tonyspark2 bash -c "$MK"
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do $J -n $h "[ -d ~/patches/dsv41-exl3-sr2roce ] || { $MK; }"; done
$J -n tonyspark3@192.168.192.3 'for f in sr-b1-best-go.sh sr-b1-idxsplit-go.sh sr-b1-k4-go.sh; do echo == $f; sed -n "/# ---- speed run/,/export EXP_NAME/p" $f | grep -v "^#"; bash -n $f && echo ok; done; ls sr-b1-*-go.sh | wc -l'
cat > $D/queue.txt.new <<'EOF'
b1-best
b1-k4
b1-mb16k
b1-idxlogits
b1-ctx1m
b1-noexpseg
b1-gmu85
b1-shexp
EOF
mv $D/queue.txt.new $D/queue.txt
echo "queue now:"; cat $D/queue.txt | tr '\n' ' '; echo
