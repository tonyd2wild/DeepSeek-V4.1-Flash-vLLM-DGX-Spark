#!/bin/bash
# flusher2.sh: drop clean page cache only when the DROPPABLE part (Cached - Mapped - Shmem) >= LIMIT_GIB (default 4), every 1 s.
LIM=${LIMIT_GIB:-4}; MAX=${MAX_S:-2400}; t0=$(date +%s); n=0; rm -f ~/flusher.stop
echo "$(date +%T) flusher2 start limit=${LIM}GiB droppable"
while [ ! -f ~/flusher.stop ] && [ $(( $(date +%s) - t0 )) -lt "$MAX" ]; do
  d=$(awk '/^Cached:/{c=$2} /^Mapped:/{m=$2} /^Shmem:/{s=$2} END{print int((c-m-s)/1048576)}' /proc/meminfo)
  if [ "$d" -ge "$LIM" ]; then echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; n=$((n+1)); [ $((n % 10)) -eq 1 ] && echo "$(date +%T) droppable ${d}GiB -> dropped (#$n) MemFree $(awk '/MemFree/{print int($2/1048576)}' /proc/meminfo)GiB"; fi
  sleep 1
done; echo "$(date +%T) flusher2 exit after $n drops"
