#!/bin/bash
# servguard4.sh (Asusi): serving-phase guard for the 4-Spark lane. Waits for the head's /health, then for MINUTES stops vllm_dsv41
# on all 4 if ANY node's MemAvailable < LIMIT_GIB (5). Appends to ~/memguard4.log (the poller reads GUARD/low: lines there).
LIM=${LIMIT_GIB:-5}; MIN=${MINUTES:-60}; LOG=~/memguard4.log
J='ssh -n -i /home/tonyspark3/.ssh/id_ed25519_shared -o ConnectTimeout=4 -o StrictHostKeyChecking=no -o BatchMode=yes'
Q='awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo'; PEERS="reddie:tonyspark2@192.168.192.2 bluey:tonyspark1@192.168.192.1 spark4:tonyspark4@192.168.192.4"
stop_all(){ echo "$(date +%T) SERVING GUARD TRIP: $1 -> docker stop on all 4" >> $LOG; docker stop -t 5 vllm_dsv41 >/dev/null 2>&1 & for p in $PEERS; do $J ${p#*:} "docker stop -t 5 vllm_dsv41" >/dev/null 2>&1 & done; wait; echo "$(date +%T) GUARD: stop sent" >> $LOG; exit 0; }
for i in $(seq 1 240); do curl -sf -m 3 http://192.168.192.2:8000/health >/dev/null && break; sleep 5; done
echo "$(date +%T) serving guard start limit=${LIM}GiB for ${MIN} min" >> $LOG; t0=$(date +%s); lowest_seen=99
while [ $(( $(date +%s) - t0 )) -lt $(( MIN * 60 )) ]; do
  m="asusi=$(bash -c "$Q")"; for p in $PEERS; do m="$m ${p%%:*}=$(timeout 6 $J ${p#*:} "$Q" 2>/dev/null)"; done
  for v in $m; do n=${v%%=*}; x=${v#*=}; [ -n "$x" ] && [ "$x" -lt "$LIM" ] && stop_all "$n MemAvailable ${x}GiB ($m)"; done
  lowest=$(for v in $m; do echo ${v#*=}; done | grep -E "^[0-9]+$" | sort -n | head -1)
  [ -n "$lowest" ] && [ "$lowest" -lt "$lowest_seen" ] && { lowest_seen=$lowest; echo "$(date +%T) low: $m (serving, new minimum)" >> $LOG; }
  sleep 5
done; echo "$(date +%T) serving guard done after ${MIN} min, lowest ${lowest_seen} GiB" >> $LOG
