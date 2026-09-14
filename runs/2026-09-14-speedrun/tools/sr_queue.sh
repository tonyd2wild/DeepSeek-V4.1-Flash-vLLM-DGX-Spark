#!/bin/bash
# sr_queue.sh <wait-label> <fallback-label> <label>...  (root on Reddie, run detached with setsid nohup)
# Waits until run-<wait-label>.status shows SCREEN-DONE (or a failure), then runs each label with sr_run.sh.
# Stops a label chain early on a failed quality gate. After 2 consecutive BOOT-FAILs it stops experimenting
# and boots <fallback-label> (the last known-good go script) so the endpoint is back up (house rule).
D=/var/tmp/boot-results/speedrun
WAIT=${1:?wait label}; FB=${2:?fallback label}; shift 2
until grep -qE "SCREEN-DONE|SCREEN-FAIL|BOOT-FAIL" $D/run-$WAIT.status 2>/dev/null; do sleep 15; done
echo "queue: $WAIT finished ($(tail -1 $D/run-$WAIT.status)) $(date -u +%T); queue: $*"
# NOBASE=1: the queued labels do not stack on <wait-label> (it is only an ordering point), so skip the base check.
if [ -z "${NOBASE:-}" ] && { grep -q BOOT-FAIL $D/run-$WAIT.status || grep -q '"all_pass": false' $D/$WAIT/quality.json 2>/dev/null; }; then
  echo "base $WAIT failed to boot or failed the quality gate: not stacking on it; booting fallback $FB $(date -u +%T)"
  bash /root/sr_boot.sh sr-$FB-go.sh $FB-fallback; echo "queue done (fallback) $(date -u +%T)"; exit 0
fi
fails=0
for L in "$@"; do
  [ -f $D/queue.stop ] && { echo "queue.stop found, stopping before $L"; break; }
  echo "=== queue: $L $(date -u +%T)"
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
echo "queue done $(date -u +%T)"
