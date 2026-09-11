#!/bin/bash
# vision_after.sh LABEL (Bluey): after post_boot2 LABEL finishes, run the vision + tools check and append the result to its log.
L=${1:?label}; LOG=~/post_boot_$L.log
for i in $(seq 1 270); do grep -q "POST_BOOT DONE $L" $LOG 2>/dev/null && break; sleep 20; done
if curl -sf -m 5 http://192.168.192.2:8000/health >/dev/null; then
  python3 ~/vision_tools_demo.py http://192.168.192.2:8000/v1 > /var/tmp/boot-results/$L/vision-tools.txt 2>&1; rc=$?
  echo "VISION DONE rc=$rc $(tail -1 /var/tmp/boot-results/$L/vision-tools.txt)" >> $LOG
else echo "VISION DONE skipped: head not healthy" >> $LOG; fi
