#!/bin/bash
# boot_dsv41_tp3b.sh (run on Asusi, try 4) - worker-first TP3 boot: rank 2 Reddie, rank 1 Asusi, then rank 0 Bluey (head).
J='ssh -i ~/.ssh/id_ed25519_shared -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o BatchMode=yes'
KNOBS=""
for k in EXP_NAME GMU MAXLEN SEQS MAX_BATCHED EAGER CUDAGRAPH_MODE SPEC SPEC_K IMAGE MODEL_DIR ENGRAM_DISK ENGRAM_THREADS ENGRAM_CHUNK ENGRAM_LOCAL PATCH_NAME TEXT_ONLY THINKING PARSERS RUST_FE VLLM_EXTRA NCCL_EXTRA; do
  v="${!k:-}"; [ -n "$v" ] && KNOBS="$KNOBS $k='$v'"
done
echo "knobs:$KNOBS"
run() { if [ "$1" = "local" ]; then bash -c "export $KNOBS; bash ~/dsv41-tp3b.sh $2"; else $J "$1" "export $KNOBS; bash ~/dsv41-tp3b.sh $2"; fi; }
set -e
echo "== rank 2 Reddie =="; run tonyspark2@192.168.192.2 2
echo "== rank 1 Asusi ==";  run local 1
sleep 5
echo "== rank 0 Bluey (head) =="; run tonyspark1@192.168.192.1 0
echo "all three launched $(date)"
