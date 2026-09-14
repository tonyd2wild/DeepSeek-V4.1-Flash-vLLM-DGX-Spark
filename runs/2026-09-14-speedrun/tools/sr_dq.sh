#!/bin/bash
# sr_dq.sh <wait-label> <fallback-label>  (root on Reddie, run detached with setsid nohup): dynamic queue.
# Waits until run-<wait-label>.status shows SCREEN-DONE (or a failure), then repeatedly pops the first label from
# /var/tmp/boot-results/speedrun/queue.txt and runs it with sr_run.sh, so the order can be edited while it runs
# (replace queue.txt atomically: write queue.txt.new, then mv). Stops when queue.txt is empty or queue.stop exists.
# After 2 consecutive BOOT-FAILs it stops experimenting and boots <fallback-label> (house rule).
D=/var/tmp/boot-results/speedrun
WAIT=${1:?wait label}; FB=${2:?fallback label}
until grep -qE "SCREEN-DONE|SCREEN-FAIL|BOOT-FAIL" $D/run-$WAIT.status 2>/dev/null; do sleep 15; done
echo "dq: $WAIT finished ($(tail -1 $D/run-$WAIT.status)) $(date -u +%T)"
fails=0
while true; do
  [ -f $D/queue.stop ] && { echo "queue.stop found, stopping $(date -u +%T)"; break; }
  L=$(grep -m1 -vE '^\s*(#|$)' $D/queue.txt 2>/dev/null)
  [ -z "$L" ] && { echo "queue.txt empty $(date -u +%T)"; break; }
  grep -vxF "$L" $D/queue.txt > $D/queue.txt.pop; mv $D/queue.txt.pop $D/queue.txt
  echo "$L" >> $D/queue.done
  echo "=== dq: $L $(date -u +%T)"
  bash /root/sr_run.sh $L > $D/run-$L.out 2>&1
  st=$(cat $D/run-$L.status)
  echo "$st" | tr '\n' ' '; echo
  if echo "$st" | grep -q BOOT-FAIL; then
    fails=$((fails+1))
    if [ $fails -ge 2 ]; then
      echo "two consecutive boot failures: stop experimenting, boot fallback $FB $(date -u +%T)"
      bash /root/sr_boot.sh sr-$FB-go.sh $FB-fallback; break
    fi
  else
    fails=0
    grep -q '"all_pass": false' $D/$L/quality.json 2>/dev/null && echo "WARNING: $L failed the quality gate"
  fi
done
echo "dq done $(date -u +%T)"
