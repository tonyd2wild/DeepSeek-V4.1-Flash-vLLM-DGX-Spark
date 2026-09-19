#!/bin/bash
# sr2_dq.sh [start-at-utc HH:MM]  (root on Reddie, detached): speed run 2 dynamic queue.
# Pops labels from /var/tmp/boot-results/speedrun/s2-queue.txt (editable while running: write .new then mv) and runs
# each with sr2_run.sh (boot asusi:~/sr2-<label>-go.sh, screen, table row). Stops on empty queue or s2-queue.stop.
# Two consecutive boot failures: stop and restore the live 300K best (house rule). Never starts a boot after
# STOP_AT (default 12:50 UTC) so the final config can be booted by 13:25.
D=/var/tmp/boot-results/speedrun
START=${1:-}
STOP_AT=${STOP_AT:-12:50}
S=$D/s2-status.txt
[ -n "$START" ] && { until [ "$(date -u +%H:%M)" \> "$START" ] || [ "$(date -u +%H:%M)" = "$START" ]; do sleep 20; done; }
echo "$(date -u +%T) dq started" >> $S
fails=0
while true; do
  [ -f $D/s2-queue.stop ] && { echo "$(date -u +%T) dq: stop file, exiting" >> $S; break; }
  [ "$(date -u +%H:%M)" \> "$STOP_AT" ] && { echo "$(date -u +%T) dq: past STOP_AT $STOP_AT, exiting" >> $S; break; }
  L=$(grep -m1 -vE '^\s*(#|$)' $D/s2-queue.txt 2>/dev/null)
  [ -z "$L" ] && { echo "$(date -u +%T) dq: queue empty, sleeping 60" >> $S; sleep 60; continue; }
  grep -vxF "$L" $D/s2-queue.txt > $D/s2-queue.txt.pop; mv $D/s2-queue.txt.pop $D/s2-queue.txt
  echo "$L" >> $D/s2-queue.done
  bash /root/sr2_run.sh $L > $D/run-$L.log 2>&1; rc=$?
  if [ $rc != 0 ]; then
    fails=$((fails+1))
    if [ $fails -ge 2 ]; then
      echo "$(date -u +%T) dq: two consecutive boot failures, restoring 300K best" >> $S
      bash /root/restore_exl3tp4b_ablit_best.sh > $D/dq-restore.log 2>&1; break
    fi
  else fails=0; fi
done
echo "$(date -u +%T) dq exited" >> $S
