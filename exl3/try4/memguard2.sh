#!/bin/bash
# memguard.sh (Asusi): stop the TP3 boot on all 3 ranks if ANY node's MemAvailable < LIMIT_GIB (default 5).
# Polls every 3 s; exits when the head answers /health or after 30 min. A clean stop beats a GB10 hang.
LIM=${LIMIT_GIB:-5}; J='ssh -i /home/tonyspark3/.ssh/id_ed25519_shared -o ConnectTimeout=4 -o StrictHostKeyChecking=no -o BatchMode=yes'
Q='awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo'
stop_all(){ echo "$(date +%T) GUARD TRIP: $1 -> docker stop on all 3"; docker stop -t 5 vllm_dsv41 >/dev/null 2>&1 & $J tonyspark2@192.168.192.2 "docker stop -t 5 vllm_dsv41" >/dev/null 2>&1 & $J tonyspark1@192.168.192.1 "docker stop -t 5 vllm_dsv41" >/dev/null 2>&1 & wait; echo "$(date +%T) GUARD: stop sent"; exit 0; }
echo "$(date +%T) memguard start limit=${LIM}GiB"; t0=$(date +%s); low=""
while [ $(( $(date +%s) - t0 )) -lt 1800 ]; do
  curl -sf -m 3 http://192.168.192.1:8000/health >/dev/null && { echo "$(date +%T) head /health OK -> memguard exit"; exit 0; }
  a=$(bash -c "$Q"); r=$(timeout 6 $J tonyspark2@192.168.192.2 "$Q" 2>/dev/null); b=$(timeout 6 $J tonyspark1@192.168.192.1 "$Q" 2>/dev/null)
  m="asusi=${a:-?} reddie=${r:-?} bluey=${b:-?}"
  for v in "asusi:$a" "reddie:$r" "bluey:$b"; do n=${v%%:*}; x=${v#*:}; [ -n "$x" ] && [ "$x" -lt "$LIM" ] && stop_all "$n MemAvailable ${x}GiB ($m)"; done
  lowest=$(printf "%s\n" $a $r $b | grep -E "^[0-9]+$" | sort -n | head -1); [ "${lowest:-99}" -lt 10 ] && [ "$m" != "$low" ] && { echo "$(date +%T) low: $m"; low="$m"; }
  sleep 3
done; echo "$(date +%T) memguard timeout exit"
