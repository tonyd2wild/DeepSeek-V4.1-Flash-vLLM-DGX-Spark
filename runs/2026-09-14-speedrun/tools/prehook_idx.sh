#!/bin/bash
# prehook: indexer prefill TP-split exactness test on Bluey (one GB10, serving image exl3a) while the previous
# experiment's server is idle. Output -> /var/tmp/boot-results/speedrun/test-idxsplit2.txt (on Reddie).
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
H=tonyspark1@192.168.192.1
O=/var/tmp/boot-results/speedrun/test-idxsplit2.txt
echo "=== idxsplit test on Bluey $(date -u +%T)" > $O
tar cf - -C /root idxsplit | $J $H 'rm -rf /tmp/idxsplit; tar xf - -C /tmp && ls /tmp/idxsplit' >> $O 2>&1
$J -n $H 'm=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); echo "MemAvailable ${m}G"; [ "$m" -ge 20 ] || { echo "SKIP: under 20G free"; exit 0; }
V=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py
A="--heads 32 --topk 512 --cand-k 2048 --cand-block 8"
for args in "--mode sim" "--mode sim --ties" "--mode spawn --tp 4" "--mode sim --long"; do
  echo "--- $args $(date -u +%T)"
  timeout 600 docker run --rm --gpus all --network none -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -v /tmp/idxsplit/sparse_attn_indexer.py:$V:ro -v /tmp/idxsplit/test_idxsplit.py:/t.py:ro \
    --entrypoint python3 vllm-dsv41:exl3a /t.py $args $A 2>&1 | grep -vE "^WARNING|UserWarning|warnings.warn" | tail -40
  echo "rc=${PIPESTATUS[0]}"
done' >> $O 2>&1
echo "=== done $(date -u +%T)" >> $O
