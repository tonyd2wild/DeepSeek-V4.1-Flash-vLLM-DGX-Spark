#!/bin/bash
# pb_wait.sh CUT LABEL NOTES (Bluey): wait for a container started after CUT (UTC), then run post_boot2 for LABEL.
# Waiting on StartedAt keeps post_boot2 from benchmarking the previous server while it is being stopped.
CUT=${1:?cut}; L=${2:?label}; N=${3:-}
for i in $(seq 1 240); do s=$(docker inspect -f '{{.State.StartedAt}}' vllm_dsv41 2>/dev/null | cut -c1-19); [ "$s" \> "$CUT" ] && break; sleep 10; done
echo "[$(date +%T)] new container started $s; post_boot2 $L waits for /health"
NOTES="$N" exec bash ~/post_boot2.sh "$L"
