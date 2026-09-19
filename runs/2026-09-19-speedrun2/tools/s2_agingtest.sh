#!/bin/bash
# Aging test on a fresh lane (root on Reddie): probe prefill warm, drop page cache on all four the way the cron does
# (echo 1 > drop_caches), probe again, then a third time. Also idle count before/after. Log only.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=12"
O=/var/tmp/boot-results/speedrun/aging-$(date -u +%H%M); mkdir -p $O
echo "=== aging test $(date -u +%T) on $(docker inspect vllm_dsv41 --format '{{json .Args}}' | grep -oE '"--max-model-len","[0-9]+"' )"
echo "--- probe 1 (as booted)"; bash /root/probe_prefill.sh aging-p1 2>&1 | grep -E "prefill_tok_s|engram-disk|GPU util" | head -3
echo "--- cached before drop: REDDIE $(awk '/^Cached:/{print int($2/1048576)}' /proc/meminfo) GiB"
for p in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do $J -n $p "sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; awk '/^Cached:/{print \"  cached after drop: \" int(\$2/1048576) \" GiB\"}' /proc/meminfo"; done
sync; echo 1 > /proc/sys/vm/drop_caches; echo "--- REDDIE cached after drop: $(awk '/^Cached:/{print int($2/1048576)}' /proc/meminfo) GiB"
echo "--- probe 2 (right after drop_caches on all four)"; bash /root/probe_prefill.sh aging-p2 2>&1 | grep -E "prefill_tok_s|GPU util" | head -3
echo "--- probe 3 (same prompt again, rows now warm)"; bash /root/probe_prefill.sh aging-p3 2>&1 | grep -E "prefill_tok_s|GPU util" | head -3
echo "--- idle test after"; python3 /root/idletest.py 2>&1 | grep -E "tok/s after first" | cut -c1-80
echo "=== aging test done $(date -u +%T)"
