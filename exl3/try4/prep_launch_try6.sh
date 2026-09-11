#!/bin/bash
# prep_launch_try6.sh (Asusi): checks, Bluey rank-0 rows gate, burn, sidecars (memstat + flusher2 on all 3, memguard2 here), launch.
set -u
J='ssh -i ~/.ssh/id_ed25519_shared -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o BatchMode=yes'
NODE='echo "$(hostname) up $(uptime -p)"; if [ "$(hostname)" != "spark-9de7" ]; then mountpoint -q /mnt/reddie-models || sudo -n mount -t nfs -o ro,vers=3 192.168.192.2:/var/tmp/models /mnt/reddie-models; test -f /mnt/reddie-models/DeepSeek-V4.1-Flash-EXL3-TP3/config.json && echo "  NFS model OK" || echo "  NFS MODEL MISSING"; fi; L=/var/tmp/engram-local/DeepSeek-V4.1-Flash-EXL3-TP3/engram-local.json; [ -f $L ] && echo "  engram rows: $(tr -d "\n " < $L | cut -c1-100)" || echo "  engram rows: none (reads its model dir)"; nvidia-smi -L | head -1 | cut -c1-40; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; echo "  MemAvailable $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) GiB"'
echo "== node checks"; bash -c "$NODE"; for h in tonyspark2@192.168.192.2 tonyspark1@192.168.192.1; do $J $h "$NODE" || { echo "$h UNREACHABLE"; exit 1; }; done
$J tonyspark1@192.168.192.1 'grep -q " 0 mismatches" ~/engram-tp3-r0.log' || { echo "BLUEY RANK-0 ROWS NOT VERIFIED YET, not launching"; exit 1; }
echo "  bluey rank-0 rows verified: $($J tonyspark1@192.168.192.1 'grep verify ~/engram-tp3-r0.log | tail -1')"
echo "== burn (healthy 75-90 TFLOPS)"; fail=0
for p in reddie:tonyspark2@192.168.192.2 bluey:tonyspark1@192.168.192.1; do ( echo "${p%%:*} $($J ${p#*:} 'bash -s' < ~/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-${p%%:*}.txt ) & done
( echo "asusi $(bash ~/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-asusi.txt ) &
wait
for n in reddie asusi bluey; do l=$(cat /tmp/burn-$n.txt); echo "  $l"; t=$(echo "$l" | grep -oE 'TFLOPS [0-9.]+' | awk '{print int($2)}'); [ "${t:-0}" -ge 50 ] || { echo "  FAIL $n burn ${t:-none}"; fail=1; }; done
[ "$fail" = 0 ] || { echo "BURN FAILED, not launching"; exit 1; }
echo "== sidecars"
for h in tonyspark2@192.168.192.2 tonyspark1@192.168.192.1; do $J $h 'setsid nohup bash ~/memstat.sh > /dev/null 2>&1 < /dev/null & setsid nohup bash ~/flusher2.sh > ~/flusher2.log 2>&1 < /dev/null & sleep 1; head -1 ~/flusher2.log' < /dev/null; done
setsid nohup bash ~/memstat.sh > /dev/null 2>&1 < /dev/null & setsid nohup bash ~/flusher2.sh > ~/flusher2.log 2>&1 < /dev/null & sleep 1; head -1 ~/flusher2.log
setsid nohup bash ~/memguard2.sh > ~/memguard2.log 2>&1 < /dev/null & sleep 1; head -1 ~/memguard2.log
echo "== launch $(date +%T)"; bash ~/exl3tp3a6-go.sh
