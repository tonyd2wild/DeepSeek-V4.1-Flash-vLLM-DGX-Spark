#!/bin/bash
# flusher.sh: during a TP3 boot, drop clean page cache whenever Cached >= LIMIT_GIB (default 12), every 3 s.
# Stops when ~/flusher.stop exists or after MAX_S (default 2400 s). Unified memory: page cache competes with
# GPU allocations, and on GB10 a starved allocator hangs the box instead of failing cleanly.
LIM=${LIMIT_GIB:-12}; MAX=${MAX_S:-2400}; t0=$(date +%s); n=0; rm -f ~/flusher.stop
echo "$(date +%T) flusher start limit=${LIM}GiB max=${MAX}s"
while [ ! -f ~/flusher.stop ] && [ $(( $(date +%s) - t0 )) -lt "$MAX" ]; do
  c=$(awk '/^Cached:/{print int($2/1048576)}' /proc/meminfo)
  if [ "$c" -ge "$LIM" ]; then sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; n=$((n+1)); echo "$(date +%T) Cached ${c}GiB -> dropped (#$n), MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)GiB"; fi
  sleep 3
done
echo "$(date +%T) flusher exit after $n drops"
