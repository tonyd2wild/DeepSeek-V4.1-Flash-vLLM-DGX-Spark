#!/bin/bash
# sr_finalize.sh (root on Reddie, run detached): the unattended end of the speed run.
#  1. At STOP_AT (UTC HHMM, default 0955) or when finalize.now exists: touch queue.stop so no new label starts.
#  2. Wait for the running sr_run.sh to finish.
#  3. Build the final go script from final-labels.txt (one label: copy it; several: sr_combo.sh merges them in order),
#     save it as asusi:~/exl3tp4b-ablit-best-go.sh (the old one is kept as .bak-HHMM).
#  4. Boot it with the documented restore path (/root/restore_exl3tp4b_ablit_best.sh), so the restore itself is tested.
#     If that boot fails, restore b1-best (known good). If that fails too (two deaths), restore the pre-speed-run config.
#  5. sr_final.sh final-best (full C1-C6, prefill 2K-64K, vision/tools, quality), then a 131K needle, then liveness.sh.
# Milestones go to finalize.status. Edit final-labels.txt any time before step 3.
D=/var/tmp/boot-results/speedrun; S=$D/finalize.status; STOP_AT=${STOP_AT:-0955}
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
A=tonyspark3@192.168.192.3
st() { echo "$* $(date -u +%T)" | tee -a $S; }
st "ARMED stop_at=$STOP_AT"
until [ "$(date -u +%H%M)" -ge "$STOP_AT" ] || [ -f $D/finalize.now ]; do sleep 30; done
touch $D/queue.stop; st "QUEUE-STOP set"
for i in $(seq 1 120); do ps -eo args | grep -q "^bash /root/sr_run\.sh" || break; sleep 20; done
st "QUEUE-IDLE (last run: $(tail -1 $D/queue.done 2>/dev/null))"
LABELS=$(grep -vE '^\s*(#|$)' $D/final-labels.txt 2>/dev/null | tr '\n' ' '); LABELS=${LABELS:-b1-best}
st "FINAL-LABELS $LABELS"
set -- $LABELS
if [ $# -eq 1 ]; then $J -n $A "cp ~/sr-$1-go.sh ~/sr-final-best-go.sh"; else bash /root/sr_combo.sh final-best "$@" > $D/finalize-combo.log 2>&1; fi
$J -n $A 'f=~/exl3tp4b-ablit-best-go.sh; [ -f $f ] && cp $f $f.bak-$(date -u +%H%M); cp ~/sr-final-best-go.sh $f && bash -n $f && sed -n "/# ---- speed run/,/export EXP_NAME/p" $f' > $D/finalize-go.txt 2>&1
st "BEST-GO written ($(grep -c export $D/finalize-go.txt) export lines)"
if bash /root/restore_exl3tp4b_ablit_best.sh > $D/finalize-restore.log 2>&1; then
  st "SERVING final-best"
else
  st "FINAL BOOT FAILED rc=$? -> restoring b1-best"
  $J -n $A 'cp ~/sr-b1-best-go.sh ~/exl3tp4b-ablit-best-go.sh'
  if bash /root/restore_exl3tp4b_ablit_best.sh > $D/finalize-restore2.log 2>&1; then st "SERVING b1-best (fallback)"
  else st "SECOND BOOT FAILED -> pre-speed-run config"; bash /root/restore_exl3tp4b_ablit.sh > $D/finalize-restore3.log 2>&1; st "restore_exl3tp4b_ablit rc=$?"; exit 1; fi
fi
bash /root/sr_final.sh final-best > $D/finalize-final.log 2>&1; st "SR-FINAL done"
python3 /root/v41needle.py --targets 131072 --depth 0.5 --out $D/final-best/needle.json > $D/final-best/needle.txt 2>&1; st "NEEDLE $(grep -o '"pass": [a-z]*' $D/final-best/needle.txt | tr '\n' ' ')"
setsid nohup bash /root/liveness.sh > /dev/null 2>&1 < /dev/null & st "LIVENESS started"
st "FINALIZE DONE"
