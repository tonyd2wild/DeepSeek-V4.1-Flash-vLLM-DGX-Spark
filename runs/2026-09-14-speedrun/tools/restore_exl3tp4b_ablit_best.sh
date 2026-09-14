#!/bin/bash
# restore_exl3tp4b_ablit_best.sh (root on Reddie): bring back the UNCENSORED DeepSeek-V4.1-Flash on the BEST config found
# in the 2026-09-14 speed run (go script ~tonyspark3/exl3tp4b-ablit-best-go.sh on Asusi; see
# runs/2026-09-14-speedrun/README.md for what is in it and the measured numbers). Same flow as restore_exl3tp4b_ablit.sh:
# prep_launch_tp4.sh on Asusi (stop all 4, node checks, GPU burn, sidecars, worker-first fan-out), poll until serving,
# memfree flushers on the two wide-slice ranks, serving guard for the first hour, then postcheck_exl3tp4b.sh.
# The pre-speed-run config stays available: bash /root/restore_exl3tp4b_ablit.sh
GO=${GO:-exl3tp4b-ablit-best-go.sh}
L=/var/tmp/restore-exl3tp4b-ablit-best-$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$L") 2>&1
echo "=== restore exl3tp4b-ablit BEST ($GO) $(date -u +%T) (log $L)"
T0=$(date -u +%s)
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
$J tonyspark3@192.168.192.3 "GO=$GO PREV=before-restore-best-$(date -u +%m%d%H%M) bash ~/prep_launch_tp4.sh" < /dev/null || { echo "prep_launch_tp4.sh failed: not launched"; exit 1; }
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
echo "=== restore exl3tp4b-ablit BEST done $(date -u +%T)"
