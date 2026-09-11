#!/bin/bash
# memfree_flusher.sh (Kai 2026-09-11): drop clean page cache only when MemFree < MIN_FREE_GIB (4) and at least 2 GiB of it is
# reclaimable (MemAvailable - MemFree). Every 2 s for MAX_S (900). Replaces flusher2.sh's Cached-Mapped-Shmem test, which goes
# negative on these boxes (Mapped already counts the shmem mappings) and never fires.
MIN=${MIN_FREE_GIB:-4}; MAX=${MAX_S:-900}; t0=$(date +%s); n=0; LOG=~/memfree-flusher.log
echo "$(date +%T) memfree_flusher start min_free=${MIN}GiB for ${MAX}s" >> $LOG
while [ $(( $(date +%s) - t0 )) -lt "$MAX" ]; do
  read f a < <(awk '/^MemFree:/{f=$2} /^MemAvailable:/{a=$2} END{print int(f/1048576), int(a/1048576)}' /proc/meminfo)
  if [ "$f" -lt "$MIN" ] && [ $((a - f)) -ge 2 ]; then sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; n=$((n+1)); echo "$(date +%T) drop #$n (MemFree ${f} GiB, MemAvailable ${a} GiB)" >> $LOG; fi
  sleep 2
done; echo "$(date +%T) memfree_flusher exit after $n drops" >> $LOG
