#!/bin/bash
# prep_launch_tp4.sh (Asusi): the TP4 EXL3 lane (4 Sparks, head Reddie). Stops vllm_dsv41 on all 4 (the TP3 head Bluey first; logs and
# memstat saved), checks NFS, boot 10's node-local Engram rows, the exl3a image and patch set tp3e on every node, drops caches, burns all 4,
# starts sidecars (memstat + flusher2 on all 4, memguard4 here), then launches via boot_dsv41_tp4x.sh (Bluey r3, Asusi r2, Spark4 r1, Reddie r0).
set -u
J='ssh -i ~/.ssh/id_ed25519_shared -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o BatchMode=yes'
PREV=${PREV:-exl3tp3a11}; GO=${GO:-exl3tp4b-go.sh}
SAVE='mkdir -p ~/boot-logs; touch ~/memstat.stop; cp ~/memstat.log ~/boot-logs/'$PREV'-memstat-$(hostname).log 2>/dev/null; docker logs vllm_dsv41 > ~/boot-logs/'$PREV'-$(hostname).log 2>&1; docker stop -t 15 vllm_dsv41 >/dev/null 2>&1; echo "  $(hostname) $(docker inspect -f "{{.State.Status}} exit={{.State.ExitCode}}" vllm_dsv41 2>/dev/null)"'
echo "== stop $PREV on all 4 (TP3 head Bluey first) $(date +%T)"
$J -n tonyspark1@192.168.192.1 "$SAVE"; bash -c "$SAVE"; $J -n tonyspark2@192.168.192.2 "$SAVE"; $J -n tonyspark4@192.168.192.4 "$SAVE"
for h in tonyspark1@192.168.192.1 tonyspark2@192.168.192.2 tonyspark4@192.168.192.4; do [ "$($J -n $h 'docker inspect -f {{.State.Running}} vllm_dsv41 2>/dev/null')" != true ] || { echo "$h STILL RUNNING, not launching"; exit 1; }; done
[ "$(docker inspect -f '{{.State.Running}}' vllm_dsv41 2>/dev/null)" != true ] || { echo "asusi STILL RUNNING, not launching"; exit 1; }
NODE='echo "$(hostname) up $(uptime -p)"; if [ "$(hostname)" != "spark-9de7" ]; then mountpoint -q /mnt/reddie-models || sudo -n mount -t nfs -o ro,vers=3 192.168.192.2:/var/tmp/models /mnt/reddie-models; test -f /mnt/reddie-models/DeepSeek-V4.1-Flash-EXL3-Pollard/config.json && echo "  NFS model OK" || echo "  NFS MODEL MISSING"; L=/var/tmp/engram-local/DeepSeek-V4.1-Flash/engram-local.json; [ -f $L ] && echo "  engram rows: $(tr -d "\n " < $L | cut -c60-150)" || echo "  ENGRAM ROWS MISSING"; fi; docker image inspect vllm-dsv41:exl3a >/dev/null 2>&1 && echo "  image exl3a OK" || echo "  IMAGE exl3a MISSING"; test -f ~/patches/dsv41-exl3-tp3e/mounts.txt && echo "  patch set tp3e OK" || echo "  PATCH SET tp3e MISSING"; test -f ~/dsv41-tp4x.sh && echo "  launcher OK" || echo "  LAUNCHER MISSING"; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; echo "  MemAvailable $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) GiB"'
echo "== node checks"; out=$( { bash -c "$NODE"; for h in tonyspark2@192.168.192.2 tonyspark1@192.168.192.1 tonyspark4@192.168.192.4; do $J -n $h "$NODE" || echo "$h UNREACHABLE"; done; } 2>&1 ); echo "$out"
echo "$out" | grep -qE "MISSING|UNREACHABLE" && { echo "NODE CHECK FAILED, not launching"; exit 1; }
echo "== burn (healthy 75-90 TFLOPS)"; fail=0
for p in reddie:tonyspark2@192.168.192.2 bluey:tonyspark1@192.168.192.1 spark4:tonyspark4@192.168.192.4; do ( echo "${p%%:*} $($J ${p#*:} 'bash -s' < ~/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-${p%%:*}.txt ) & done
( echo "asusi $(bash ~/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-asusi.txt ) &
wait
for n in reddie asusi bluey spark4; do l=$(cat /tmp/burn-$n.txt); echo "  $l"; t=$(echo "$l" | grep -oE 'TFLOPS [0-9.]+' | awk '{print int($2)}'); [ "${t:-0}" -ge 50 ] || { echo "  FAIL $n burn ${t:-none}"; fail=1; }; done
[ "$fail" = 0 ] || { echo "BURN FAILED, not launching"; exit 1; }
echo "== sidecars"
for h in tonyspark2@192.168.192.2 tonyspark1@192.168.192.1 tonyspark4@192.168.192.4; do $J $h 'setsid nohup bash ~/memstat.sh > /dev/null 2>&1 < /dev/null & setsid nohup bash ~/flusher2.sh > ~/flusher2.log 2>&1 < /dev/null & sleep 1; head -1 ~/flusher2.log' < /dev/null; done
setsid nohup bash ~/memstat.sh > /dev/null 2>&1 < /dev/null & setsid nohup bash ~/flusher2.sh > ~/flusher2.log 2>&1 < /dev/null & sleep 1; head -1 ~/flusher2.log
setsid nohup bash ~/memguard4.sh > ~/memguard4.log 2>&1 < /dev/null & sleep 1; head -1 ~/memguard4.log
echo "== launch $(date +%T)"; bash ~/$GO
