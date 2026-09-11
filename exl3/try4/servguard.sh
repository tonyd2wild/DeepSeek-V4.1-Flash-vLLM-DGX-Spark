#!/bin/bash
# servguard.sh (Asusi): serving-phase memory guard. Waits for the head's /health, then for MINUTES stops the TP3 boot on all
# 3 ranks if ANY node's MemAvailable < LIMIT_GIB. Appends to ~/memguard2.log (the boot poller reads GUARD/low: lines there).
# memguard2.sh exits on /health by design; at gmu 0.80 the benchmark and vision check run with ~8 GiB free on the head.
LIM=${LIMIT_GIB:-4}; MIN=${MINUTES:-50}; LOG=~/memguard2.log
J='ssh -i /home/tonyspark3/.ssh/id_ed25519_shared -o ConnectTimeout=4 -o StrictHostKeyChecking=no -o BatchMode=yes'
Q='awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo'
stop_all(){ echo "$(date +%T) SERVING GUARD TRIP: $1 -> docker stop on all 3" >> $LOG; docker stop -t 5 vllm_dsv41 >/dev/null 2>&1 & $J -n tonyspark2@192.168.192.2 "docker stop -t 5 vllm_dsv41" >/dev/null 2>&1 & $J -n tonyspark1@192.168.192.1 "docker stop -t 5 vllm_dsv41" >/dev/null 2>&1 & wait; echo "$(date +%T) GUARD: stop sent" >> $LOG; exit 0; }
for i in $(seq 1 240); do curl -sf -m 3 http://192.168.192.1:8000/health >/dev/null && break; sleep 5; done
echo "$(date +%T) serving guard start limit=${LIM}GiB for ${MIN} min" >> $LOG; t0=$(date +%s); low=""; lowest_seen=99
while [ $(( $(date +%s) - t0 )) -lt $(( MIN * 60 )) ]; do
  a=$(bash -c "$Q"); r=$(timeout 6 $J -n tonyspark2@192.168.192.2 "$Q" 2>/dev/null); b=$(timeout 6 $J -n tonyspark1@192.168.192.1 "$Q" 2>/dev/null)
  m="asusi=${a:-?} reddie=${r:-?} bluey=${b:-?}"
  for v in "asusi:$a" "reddie:$r" "bluey:$b"; do n=${v%%:*}; x=${v#*:}; [ -n "$x" ] && [ "$x" -lt "$LIM" ] && stop_all "$n MemAvailable ${x}GiB ($m)"; done
  lowest=$(printf "%s\n" $a $r $b | grep -E "^[0-9]+$" | sort -n | head -1)
  [ -n "$lowest" ] && [ "$lowest" -lt "$lowest_seen" ] && { lowest_seen=$lowest; echo "$(date +%T) low: $m (serving, new minimum)" >> $LOG; }
  sleep 5
done; echo "$(date +%T) serving guard done after ${MIN} min, lowest ${lowest_seen} GiB" >> $LOG
