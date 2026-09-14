#!/bin/bash
# Replace the waiting fixed queues 2, 3 and 4 with one dynamic queue (sr_dq.sh) that starts after E05.
# queue-1b (E09 running, then E04, E05) is left alone. Only queues that are still waiting are killed.
D=/var/tmp/boot-results/speedrun
for pat in "sr_queue.sh e05-k10 " "sr_queue.sh e08-noexpseg " "sr_queue.sh e10-idxlogits "; do
  for p in $(ps -eo pid,args | grep -F "bash /root/$pat" | grep -v grep | awk '{print $1}'); do
    echo "kill $p ($pat)"; kill $p
  done
done
cat > $D/queue.txt.new <<'EOF'
e12-ch4
e14-idxsplit
e06-mb16k
e10-idxlogits
e15-tree
e11-ctx1m
e07-shexp
e08-noexpseg
e13-gmu85
e16-ch2
EOF
mv $D/queue.txt.new $D/queue.txt
if ps -eo args | grep -v grep | grep -q "sr_dq.sh"; then echo "dq already running"; else
  setsid nohup bash /root/sr_dq.sh e05-k10 e02-ch8-fast > $D/dq.log 2>&1 < /dev/null &
  sleep 1; echo "dq started"; fi
ps -eo pid,args | grep -E "sr_queue|sr_dq|sr_run" | grep -v grep | cut -c1-120
