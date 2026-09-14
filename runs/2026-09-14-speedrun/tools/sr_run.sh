#!/bin/bash
# sr_run.sh <label>  (root on Reddie): one full experiment = sr_boot.sh with asusi:~/sr-<label>-go.sh, then sr_screen.sh.
# Writes /var/tmp/boot-results/speedrun/run-<label>.status with BOOT-OK/BOOT-FAIL and SCREEN-DONE lines.
LBL=${1:?label}; S=/var/tmp/boot-results/speedrun/run-$LBL.status; mkdir -p /var/tmp/boot-results/speedrun
echo "START $(date -u +%T)" > $S
if bash /root/sr_boot.sh sr-$LBL-go.sh $LBL; then
  echo "BOOT-OK $(date -u +%T)" >> $S
  bash /root/sr_screen.sh $LBL && echo "SCREEN-DONE $(date -u +%T)" >> $S || echo "SCREEN-FAIL $(date -u +%T)" >> $S
else
  rc=$?; echo "BOOT-FAIL rc=$rc $(date -u +%T)" >> $S
fi
cat $S
