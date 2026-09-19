#!/bin/bash
# liveness.sh (root on Reddie, run detached): every 120 s send a 2-token greedy completion with a 60 s timeout
# and log the result to /var/tmp/boot-results/speedrun/liveness.log. /health can stay 200 while a rank is stuck
# (b12x#313, reported by @tobymao), so this checks real generation. Log only: it never restarts anything.
# Stop: touch /var/tmp/boot-results/speedrun/liveness.stop
D=/var/tmp/boot-results/speedrun; L=$D/liveness.log; fails=0
rm -f $D/liveness.stop
while [ ! -f $D/liveness.stop ]; do
  t0=$(date +%s.%N)
  out=$(curl -s -m 60 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4.1-flash","prompt":"Count: 1, 2, 3,","max_tokens":2,"temperature":0}' 2>/dev/null)
  dt=$(echo "$(date +%s.%N) - $t0" | bc)
  if echo "$out" | grep -q '"completion_tokens"'; then
    fails=0; echo "$(date -u +%T) ok ${dt}s" >> $L
  else
    fails=$((fails+1)); echo "$(date -u +%T) FAIL #$fails ${dt}s $(echo "$out" | head -c 160)" >> $L
    [ $fails -ge 3 ] && echo "$(date -u +%T) ALERT: 3 consecutive generation failures (possible stuck rank)" >> $L
  fi
  tail -n 2000 $L > $L.tmp && mv $L.tmp $L
  sleep 120
done
