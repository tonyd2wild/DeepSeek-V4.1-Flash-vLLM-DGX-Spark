#!/bin/bash
# extra_round.sh (root on Reddie): the 10:15 UTC extra round on top of the verified final config.
#  - stops the liveness probe (boots would log failures), backs up final-best results,
#  - f-gmu82 = final-best + GMU 0.82; f-noexpseg = final-best + expandable_segments:False,
#  - runs them through a fresh dynamic queue (fallback: final-best),
#  - re-arms sr_finalize.sh with STOP_AT=1048, so the best config is restored and re-verified by ~11:00 either way.
set -e
D=/var/tmp/boot-results/speedrun
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
touch $D/liveness.stop
[ -d $D/final-best-run1 ] || cp -a $D/final-best $D/final-best-run1
blk=$($J -n tonyspark3@192.168.192.3 'sed -n "/# ---- speed run/,/export EXP_NAME/p" ~/sr-final-best-go.sh' | grep '^export' | grep -v EXP_NAME)
{ echo "$blk"; echo 'export GMU=0.82'; } | bash /root/sr_mkgo.sh f-gmu82 | tail -1
{ echo "$blk"; echo 'export NCCL_EXTRA="$NCCL_EXTRA -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"'; } | bash /root/sr_mkgo.sh f-noexpseg | tail -1
$J -n tonyspark3@192.168.192.3 'for f in sr-f-gmu82-go.sh sr-f-noexpseg-go.sh; do bash -n $f && echo "$f ok $(grep -cE "^export (GMU=0.82|NCCL_EXTRA=.*expandable_segments:False)" $f)"; done'
rm -f $D/queue.stop
printf "f-gmu82\nf-noexpseg\n" > $D/queue.txt.new && mv $D/queue.txt.new $D/queue.txt
if ps -eo args | grep -q "^bash /root/sr_dq\.sh"; then echo "dq already running"; else
  setsid nohup bash /root/sr_dq.sh b1-ctx1m final-best > $D/dq2.log 2>&1 < /dev/null & sleep 1; echo "dq2 started"; fi
if ps -eo args | grep -q "^bash /root/sr_finalize\.sh"; then echo "finalize already running"; else
  echo "--- re-armed for the extra round $(date -u +%T)" >> $D/finalize.status
  STOP_AT=1048 setsid nohup bash /root/sr_finalize.sh > $D/finalize2.log 2>&1 < /dev/null & sleep 1; echo "finalize re-armed STOP_AT=1048"; fi
ps -eo pid,args | grep -E "^ *[0-9]+ bash /root/sr_(finalize|dq|run)" | cut -c1-100
cat $D/final-labels.txt | tr '\n' ' '; echo
