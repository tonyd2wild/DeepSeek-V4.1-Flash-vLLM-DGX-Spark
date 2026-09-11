#!/bin/bash
# boot_dsv41_tp4x.sh  (run on Asusi, EXL3 variant) — worker-first TP4 boot of DeepSeek-V4.1-Flash on the 4 Sparks.
# Passes the same knob env (EXP_NAME GMU MAXLEN SEQS EAGER SPEC IMAGE MODEL_DIR ENGRAM_LOCAL_HOST ENGRAM_DISK TEXT_ONLY THINKING PARSERS RUST_FE VLLM_EXTRA) to every rank.
J='ssh -i ~/.ssh/id_ed25519_shared -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o BatchMode=yes'
KNOBS=""
for k in EXP_NAME GMU MAXLEN SEQS MAX_BATCHED EAGER CUDAGRAPH_MODE SPEC SPEC_K IMAGE MODEL_DIR ENGRAM_LOCAL_HOST ENGRAM_DISK ENGRAM_THREADS ENGRAM_CHUNK ENGRAM_LOCAL PATCH_NAME TEXT_ONLY THINKING PARSERS RUST_FE VLLM_EXTRA NCCL_EXTRA; do
  v="${!k:-}"; [ -n "$v" ] && KNOBS="$KNOBS $k='$v'"
done
echo "knobs:$KNOBS"
run() { # host rank
  if [ "$1" = "local" ]; then bash -c "export $KNOBS; bash ~/dsv41-tp4x.sh $2"; else $J "$1" "export $KNOBS; bash ~/dsv41-tp4x.sh $2"; fi
}
set -e
echo "== rank 3 Bluey ==";  run tonyspark1@192.168.192.1 3
echo "== rank 2 Asusi ==";  run local 2
echo "== rank 1 Spark4 =="; run tonyspark4@192.168.192.4 1
sleep 5
echo "== rank 0 Reddie (head) =="; run tonyspark2@192.168.192.2 0
echo "all four launched $(date)"
