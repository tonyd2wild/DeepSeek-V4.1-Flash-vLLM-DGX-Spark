#!/bin/bash
# sr_boot.sh <go-script on Asusi> <label>  (root on Reddie): one speed-run experiment boot of the EXL3 TP4 lane.
# Same flow as /root/restore_exl3tp4b_ablit.sh (prep_launch_tp4.sh on Asusi: stop all 4, node checks, burn, sidecars,
# worker-first fan-out; poll until serving; memfree flushers on the two wide-slice ranks; serving guard), with the go
# script chosen per experiment. Exit 0 = serving, 1 = not launched, 2 = boot failed, 3 = timeout.
GO=${1:?go script name in ~tonyspark3}; LBL=${2:?label}
mkdir -p /var/tmp/boot-results/speedrun
L=/var/tmp/boot-results/speedrun/boot-$LBL.log
exec > >(tee -a "$L") 2>&1
echo "=== sr_boot $LBL ($GO) $(date -u +%T)"
T0=$(date -u +%s)
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
$J tonyspark3@192.168.192.3 "GO=$GO PREV=before-$LBL bash ~/prep_launch_tp4.sh" < /dev/null || { echo "prep_launch_tp4.sh failed: not launched"; exit 1; }
for i in $(seq 1 80); do
  c=$(docker inspect vllm_dsv41 --format '{{.Created}}' 2>/dev/null)
  ct=$([ -n "$c" ] && date -u -d "$c" +%s || echo 0)
  [ "$ct" -gt "$T0" ] || { echo "no new head container on Reddie: stopping"; exit 2; }
  l=$(bash /root/v41poll.sh 2>&1 | tr '\n' ' ')
  [ $((i % 4)) = 1 ] && echo "$(date -u +%T) $l"
  case "$l" in
    *"Application startup complete"*) echo "SERVING $(date -u +%T) after $(( $(date -u +%s) - T0 ))s"; break;;
    *HEAD-EXITED*|*WORKER-DOWN*) echo "BOOT FAILED $(date -u +%T): $l"; docker logs --tail 30 vllm_dsv41 2>&1 | grep -iE "error|Traceback|raise|Exception" | tail -8; exit 2;;
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
echo "=== milestones"
docker logs vllm_dsv41 2>&1 | grep -E "GPU KV cache size|Available KV cache|Model loading took|Graph capturing finished|non-default args" | sed -E "s/^.*\] //" | cut -c1-400
echo "=== sr_boot $LBL done $(date -u +%T)"
