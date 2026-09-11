#!/bin/bash
# restore_exl3tp4b.sh (root on Reddie): bring back the DEFAULT V4.1 serving config, TP4 EXL3 exl3tp4b (4 Sparks, head Reddie,
#   EXL3 3.5bpw checkpoint, CUDA graphs, DSpark k=5, vision, 300K per request, gmu 0.80, 3.3M-token KV pool), e.g. after a reboot.
#   1) runs ~/prep_launch_tp4.sh on Asusi (as tonyspark3): stops vllm_dsv41 on all four (logs saved), checks NFS, node-local Engram,
#      the exl3a image, patch set and launcher on every node, drops caches, GPU burn on all four, sidecars, worker-first fan-out.
#   2) polls with /root/v41poll.sh until the head logs "Application startup complete" (about 10 min), or stops on a failure.
#   3) once serving: stops the load-time flushers, starts memfree_flusher.sh (24 h) on the two wide-slice ranks (Asusi, Spark4:
#      they run with ~11 GiB free and long prompts fill their page cache), the serving guard for the first hour, then
#      postcheck_exl3tp4b.sh (Engram path per rank, KV pool, verify9.py).
#   Fallback to the release build (boot 10): bash /root/restore_boot10.sh
L=/var/tmp/restore-exl3tp4b-$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$L") 2>&1
echo "=== restore exl3tp4b (default) $(date -u +%T) (log $L)"
T0=$(date -u +%s)
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
$J tonyspark3@192.168.192.3 "PREV=before-restore-$(date -u +%m%d%H%M) bash ~/prep_launch_tp4.sh" < /dev/null || { echo "prep_launch_tp4.sh failed: not launched (see output above)"; exit 1; }
for i in $(seq 1 80); do
  c=$(docker inspect vllm_dsv41 --format '{{.Created}}' 2>/dev/null)
  ct=$([ -n "$c" ] && date -u -d "$c" +%s || echo 0)
  [ "$ct" -gt "$T0" ] || { echo "no new head container on Reddie: stopping"; exit 2; }
  l=$(bash /root/v41poll.sh 2>&1 | tr '\n' ' ')
  [ $((i % 4)) = 1 ] && echo "$(date -u +%T) $l"
  case "$l" in
    *"Application startup complete"*) echo "SERVING $(date -u +%T)"; break;;
    *HEAD-EXITED*|*WORKER-DOWN*) echo "BOOT FAILED $(date -u +%T): $l"; exit 2;;
  esac
  sleep 30
done
case "$l" in *"Application startup complete"*) ;; *) echo "TIMEOUT: not serving after 40 min"; exit 3;; esac
echo "=== serving-time memory tools"
$J tonyspark3@192.168.192.3 'bash -s' <<'EOS'
J="ssh -n -i ~/.ssh/id_ed25519_shared -o ConnectTimeout=6 -o BatchMode=yes"
for h in tonyspark2@192.168.192.2 tonyspark1@192.168.192.1 tonyspark4@192.168.192.4; do $J $h 'touch ~/flusher.stop'; done; touch ~/flusher.stop; sleep 2
MF='pkill -f "^bash /home/tonyspark./memfree_flusher.sh"; sleep 1; MAX_S=86400 setsid nohup bash ~/memfree_flusher.sh > /dev/null 2>&1 < /dev/null & sleep 1; echo "$(hostname) $(tail -1 ~/memfree-flusher.log)"'
bash -c "$MF"; $J tonyspark4@192.168.192.4 "$MF"
LIMIT_GIB=5 MINUTES=60 setsid nohup bash ~/servguard4.sh > /dev/null 2>&1 < /dev/null & sleep 1; echo "serving guard: $(tail -1 ~/memguard4.log)"
EOS
bash /root/postcheck_exl3tp4b.sh
echo "=== restore exl3tp4b done $(date -u +%T)"
