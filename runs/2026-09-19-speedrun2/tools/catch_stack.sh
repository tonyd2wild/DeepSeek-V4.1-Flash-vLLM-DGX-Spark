#!/bin/bash
# Wait for the post-load host-memory climb, then dump the worker's Python stack three times, 12 s apart.
# py-spy runs on the host against the container's process (PID namespaces differ, PIDs are visible on the host).
OUT=/var/tmp/nvfp4-stacks.txt
: > $OUT
for i in $(seq 1 300); do
  a=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
  [ "${a:-99}" -lt 22 ] && break
  sleep 3
done
echo "=== climb detected $(date -u +%T), avail ${a} GiB" >> $OUT
pid=$(pgrep -f "VLLM::Worker" | head -1)
[ -z "$pid" ] && pid=$(pgrep -af "python3" | grep -i "worker\|EngineCore" | awk '{print $1}' | head -1)
echo "worker pid=$pid" >> $OUT
for n in 1 2 3; do
  echo "--- dump $n at $(date -u +%T), avail $(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo) GiB" >> $OUT
  /tmp/pyspyenv/bin/py-spy dump --pid "$pid" --nonblocking >> $OUT 2>&1
  sleep 12
done
echo "=== done $(date -u +%T)" >> $OUT
