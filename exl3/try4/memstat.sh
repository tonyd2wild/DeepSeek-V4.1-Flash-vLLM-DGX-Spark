#!/bin/bash
# memstat.sh: every 5 s log meminfo + per-process memory of the vLLM processes + GPU compute apps. Stops on ~/memstat.stop or 45 min.
t0=$(date +%s); rm -f ~/memstat.stop; L=~/memstat.log; : > $L
while [ ! -f ~/memstat.stop ] && [ $(( $(date +%s) - t0 )) -lt 2700 ]; do
  { echo "=== $(date +%T) $(awk '/^(MemFree|MemAvailable|Cached|Mapped|Shmem|AnonPages):/{printf "%s%.1f ", $1, $2/1048576}' /proc/meminfo)"
    for p in $(pgrep -f "vllm|VLLM|EngineCore|APIServer|Worker" | head -12); do
      [ -r /proc/$p/smaps_rollup ] || continue
      n=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-60)
      awk -v pid=$p -v n="$n" '/^Rss:/{r=$2} /^Anonymous:/{a=$2} /^Shared_Clean:/{sc=$2} /^Shmem|^Pss_Shmem:/{sh=$2} END{if (r>524288) printf "  pid %s rss %.1f anon %.1f shmem %.1f  %s\n", pid, r/1048576, a/1048576, sh/1048576, n}' /proc/$p/smaps_rollup
    done
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'
  } >> $L
  sleep 5
done; echo "=== $(date +%T) memstat exit" >> $L
