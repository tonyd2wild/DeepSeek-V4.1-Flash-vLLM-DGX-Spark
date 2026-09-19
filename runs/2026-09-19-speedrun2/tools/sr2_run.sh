#!/bin/bash
# sr2_run.sh <label>  (root on Reddie): one speed-run-2 cycle = boot asusi:~/sr2-<label>-go.sh, then SCREEN it,
# then append one row to /var/tmp/boot-results/speedrun/s2-table.txt. Exit codes follow sr_boot.sh.
# A failed boot leaves the fleet DOWN: the caller decides whether to retry or restore.
LBL=${1:?label}
T=/var/tmp/boot-results/speedrun/s2-table.txt
S=/var/tmp/boot-results/speedrun/s2-status.txt
echo "$(date -u +%T) BOOT $LBL" >> $S
bash /root/sr_boot.sh sr2-$LBL-go.sh $LBL; rc=$?
if [ $rc != 0 ]; then echo "$(date -u +%T) BOOT-FAILED $LBL rc=$rc" >> $S; exit $rc; fi
echo "$(date -u +%T) SERVING $LBL" >> $S
bash /root/sr_screen.sh $LBL > /var/tmp/boot-results/speedrun/screen-$LBL.log 2>&1
O=/var/tmp/boot-results/speedrun/$LBL
kv=$(grep -oE "GPU KV cache size: [0-9,]+" $O/kv-context.txt | tail -1 | grep -oE "[0-9,]+")
q=$(grep -oE "PASS|FAIL" $O/quality.txt | head -1)
line=$(python3 /root/sr_table.py /var/tmp/boot-results/speedrun s2-00-baseline 2>/dev/null | grep -E "^\| *$LBL " | head -1)
echo "$LBL | KV $kv | quality $q | $line" >> $T
echo "$(date -u +%T) SCREENED $LBL" >> $S
tail -1 $T
