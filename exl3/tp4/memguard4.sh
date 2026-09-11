#!/bin/bash
# memguard4.sh (Asusi): load-time guard for the 4-Spark lane: stop vllm_dsv41 on all 4 if ANY node's MemAvailable < LIMIT_GIB (5).
# Polls every 3 s; exits when the head (Reddie) answers /health or after 30 min.
LIM=${LIMIT_GIB:-5}; J='ssh -n -i /home/tonyspark3/.ssh/id_ed25519_shared -o ConnectTimeout=4 -o StrictHostKeyChecking=no -o BatchMode=yes'
Q='awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo'; PEERS="reddie:tonyspark2@192.168.192.2 bluey:tonyspark1@192.168.192.1 spark4:tonyspark4@192.168.192.4"
stop_all(){ echo "$(date +%T) GUARD TRIP: $1 -> docker stop on all 4"; docker stop -t 5 vllm_dsv41 >/dev/null 2>&1 & for p in $PEERS; do $J ${p#*:} "docker stop -t 5 vllm_dsv41" >/dev/null 2>&1 & done; wait; echo "$(date +%T) GUARD: stop sent"; exit 0; }
echo "$(date +%T) memguard4 start limit=${LIM}GiB"; t0=$(date +%s); low=""
while [ $(( $(date +%s) - t0 )) -lt 1800 ]; do
  curl -sf -m 3 http://192.168.192.2:8000/health >/dev/null && { echo "$(date +%T) head /health OK -> memguard exit"; exit 0; }
  m="asusi=$(bash -c "$Q")"; for p in $PEERS; do m="$m ${p%%:*}=$(timeout 6 $J ${p#*:} "$Q" 2>/dev/null)"; done
  for v in $m; do n=${v%%=*}; x=${v#*=}; [ -n "$x" ] && [ "$x" -lt "$LIM" ] && stop_all "$n MemAvailable ${x}GiB ($m)"; done
  lowest=$(for v in $m; do echo ${v#*=}; done | grep -E "^[0-9]+$" | sort -n | head -1); [ "${lowest:-99}" -lt 10 ] && [ "$m" != "$low" ] && { echo "$(date +%T) low: $m"; low="$m"; }
  sleep 3
done; echo "$(date +%T) memguard timeout exit"
