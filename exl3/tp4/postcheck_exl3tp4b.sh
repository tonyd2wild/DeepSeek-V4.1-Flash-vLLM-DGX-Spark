#!/bin/bash
# postcheck_exl3tp4b.sh (root on Reddie) after the TP4 EXL3 default serves: where each rank reads Engram from, the KV pool
# (expect about 3,304,863 tokens), then verify9.py (count-to-100 x2, code x2, one tool call, one tiny image).
J="sudo -u tonyspark2 ssh -n -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
O=/var/tmp/boot-results/exl3tp4b-restore-$(date -u +%m%d%H%M); mkdir -p $O
echo "=== Engram read path per rank"
echo "REDDIE: $(docker logs vllm_dsv41 2>&1 | grep -oE 'Engram (DISK mode: layer [0-9]+ rows \[[0-9]+, [0-9]+\) read from [^ ]+|layer [0-9]+ rows \[[0-9]+, [0-9]+\) read from node-local [^ ]+)' | sort -u | tr '\n' ';')"
for p in SPARK4:tonyspark4@192.168.192.4 ASUSI:tonyspark3@192.168.192.3 BLUEY:tonyspark1@192.168.192.1; do
  echo "${p%%:*}: $($J ${p#*:} "docker logs vllm_dsv41 2>&1 | grep -E 'node-local|DSV41_ENGRAM_DIR|Engram DISK mode' | sed -E 's/^.*(Engram|DSV41)/\1/' | cut -c1-150 | sort -u | tr '\n' ';'")"
done
echo "=== KV pool"; docker logs vllm_dsv41 2>&1 | grep -E "GPU KV cache size|Available KV cache memory|Graph capturing finished|Model loading took" | sed -E 's/^.*\] //' | tail -5
echo "=== checks (verify9.py)"; python3 /root/verify9.py 2>&1 | tee $O/verify.txt
echo "head: $(docker ps --filter name=vllm_dsv41 --format '{{.Status}}')"
