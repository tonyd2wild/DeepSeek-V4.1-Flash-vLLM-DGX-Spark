#!/bin/bash
# Final measurement of the serving config (root on Reddie): sr_final vs tonight's baseline, needles, C8/C12.
S=/var/tmp/boot-results/speedrun/s2-status.txt
echo "$(date -u +%T) FINAL-MEASURE start" >> $S
bash /root/sr_final.sh s2-final /var/tmp/boot-results/speedrun/s2-00-baseline/bench-s2-00-baseline.json > /var/tmp/boot-results/speedrun/s2-final.log 2>&1
O=/var/tmp/boot-results/speedrun/s2-final
echo "$(date -u +%T) FINAL-MEASURE bench done" >> $S
python3 /root/v41needle.py --targets 65536,131072 --depth 0.3 --out $O/needle.json > $O/needle.txt 2>&1
echo "$(date -u +%T) FINAL-MEASURE needle done: $(grep -aoE 'PASS|FAIL' $O/needle.txt | tr '\n' ' ')" >> $S
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label s2-final-c8c12 --out $O/hi --levels 8,12 --prefill "" --notes "speed run 2 final C8/C12" > $O/hi.txt 2>&1
echo "$(date -u +%T) FINAL-MEASURE C8/C12 done" >> $S
echo "$(date -u +%T) FINAL-MEASURE DONE" >> $S
