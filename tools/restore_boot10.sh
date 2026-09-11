#!/bin/bash
# restore_boot10.sh (root on Reddie): bring the boot 10 serving config back, e.g. after a Spark reboot.
#   1) launch.sh 10 with the quick pre-launch check: stops vllm_dsv41 on all four nodes (head first,
#      logs saved), checks mounts, node-local Engram, staged files and GPU health, re-stages
#      /tmp/boot10-go.sh on Asusi, then does the worker-first fan-out.
#   2) polls until the head logs "Application startup complete" (about 13 min), or stops on a failure.
#   3) postcheck10.sh: Engram read path per rank, KV pool, count, code, tool call, image.
# Everything is logged to /var/tmp/restore-boot10-<time>.log.
L=/var/tmp/restore-boot10-$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$L") 2>&1
echo "=== restore boot 10 $(date -u +%T) (log $L)"
T0=$(date -u +%s)
PRELAUNCH=/root/prelaunch-quick.sh bash /root/launch.sh 10 || { echo "launch.sh failed: not launched (see prelaunch output above)"; exit 1; }
for i in $(seq 1 80); do
  c=$(docker inspect vllm_dsv41 --format '{{.Created}}' 2>/dev/null)
  ct=$([ -n "$c" ] && date -u -d "$c" +%s || echo 0)
  [ "$ct" -gt "$T0" ] || { echo "no boot 10 head container on Reddie: stopping"; exit 2; }
  l=$(bash /root/v41poll.sh 2>&1 | tr '\n' ' ')
  [ $((i % 4)) = 1 ] && echo "$(date -u +%T) $l"
  case "$l" in
    *"Application startup complete"*) echo "SERVING $(date -u +%T)"; break;;
    *HEAD-EXITED*|*WORKER-DOWN*) echo "BOOT FAILED $(date -u +%T): $l"; exit 2;;
  esac
  sleep 30
done
case "$l" in *"Application startup complete"*) ;; *) echo "TIMEOUT: not serving after 40 min"; exit 3;; esac
bash /root/postcheck10.sh
echo "=== restore boot 10 done $(date -u +%T)"
