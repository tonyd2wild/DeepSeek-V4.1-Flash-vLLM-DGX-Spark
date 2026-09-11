#!/bin/bash
# switch_tp3.sh (Asusi): boot 10 -> exl3tp3a. Save logs, stop (head first), Reddie prep, burn check, launch.
set -u
J='ssh -i ~/.ssh/id_ed25519_shared -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o BatchMode=yes'
TS=$(date +%Y%m%d-%H%M); echo "[$(date +%T)] switch start $TS"
ss(){ $J $1 "mkdir -p /var/tmp/prelaunch-logs && docker logs vllm_dsv41 > /var/tmp/prelaunch-logs/boot10-$TS-$2.log 2>&1; docker stop -t 30 vllm_dsv41 >/dev/null 2>&1; echo \"$2: \$(docker inspect -f '{{.State.Status}}' vllm_dsv41 2>/dev/null)\""; }
ss tonyspark2@192.168.192.2 reddie
ss tonyspark4@192.168.192.4 spark4
ss tonyspark1@192.168.192.1 bluey
mkdir -p /var/tmp/prelaunch-logs && docker logs vllm_dsv41 > /var/tmp/prelaunch-logs/boot10-$TS-asusi.log 2>&1; docker stop -t 30 vllm_dsv41 >/dev/null 2>&1; echo "asusi: $(docker inspect -f '{{.State.Status}}' vllm_dsv41 2>/dev/null)"
echo "[$(date +%T)] boot 10 stopped on all four (containers kept for restore_boot10.sh)"
$J tonyspark2@192.168.192.2 "bash /tmp/reddie_prep_tp3.sh" || { echo "REDDIE PREP FAILED, not launching"; exit 1; }
echo "[$(date +%T)] 10 s fp16 burn on the 3 TP3 nodes (healthy 75-90 TFLOPS, 2.2-2.4 GHz)"
for p in reddie:tonyspark2@192.168.192.2 bluey:tonyspark1@192.168.192.1; do ( echo "${p%%:*} $($J ${p#*:} 'bash -s' < /tmp/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-${p%%:*}.txt ) & done
( echo "asusi $(bash /tmp/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-asusi.txt ) &
wait; fail=0
for n in reddie asusi bluey; do l=$(cat /tmp/burn-$n.txt); echo "  $l"; t=$(echo "$l" | grep -oE 'TFLOPS [0-9.]+' | awk '{print int($2)}'); [ "${t:-0}" -ge 50 ] || { echo "  FAIL $n: burn ${t:-none} TFLOPS"; fail=1; }; done
[ "$fail" = 0 ] || { echo "BURN FAILED, not launching (EC clock latch = Tony power-cycles with the adapter unplugged)"; exit 1; }
echo "[$(date +%T)] launching exl3tp3a"; bash ~/exl3tp3a-go.sh
