#!/bin/bash
# sg_final.sh <label> (root on Reddie): the same FULL measurement the vLLM final got (tools from the speed run),
# run against the SGLang endpoint on :8000: vision + tools, quality gate, idle test, a discarded warm-up pass,
# v41bench C1-C6 over all categories + cold prefill 2K/8K/32K/64K, bench_report vs the vLLM baseline, and a
# 131K needle. KV/memory lines come from the SGLang head container log.
LBL=${1:?label}; BASE=${2:-/var/tmp/boot-results/speedrun/00-baseline/bench-speedrun-00-baseline.json}
O=/var/tmp/boot-results/sglang/$LBL; mkdir -p $O
echo "=== sg_final $LBL $(date -u +%T)"
docker logs sglang_dsv41 2>&1 | grep -iE "KV Cache is allocated|max_total_num_tokens|#tokens|avail mem|memory calculation|Load weight end|Capture cuda graph end|engram|context_len" | sed -E "s/^.*\] //" > $O/kv-context.txt
docker inspect sglang_dsv41 --format '{{json .Args}}' > $O/args.json
docker inspect sglang_dsv41 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E "NCCL|SGLANG|PYTORCH" > $O/env.txt
python3 /root/vision_tools_demo.py > $O/vision-tools.txt 2>&1; tail -1 $O/vision-tools.txt
python3 /root/sr_quality.py $O > $O/quality.txt 2>&1; cat $O/quality.txt
python3 /root/idletest.py > $O/idletest.txt 2>&1
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label $LBL-warm --out $O/warm --levels 1 --prefill "" > $O/warm.txt 2>&1 || true
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label $LBL --out $O --levels 1,2,3,4,5,6 --prefill 2000,8000,32000,64000 --notes "SGLang lane 2026-09-14 $LBL" > $O/bench.txt 2>&1
echo "bench exit $? $(date -u +%T)"
J=$(ls $O/bench-$LBL.json 2>/dev/null); [ -n "$J" ] && python3 /root/bench_report.py "$J" --vs "$BASE" > $O/report.md 2>&1
python3 /root/v41needle.py --targets 131072 --depth 0.5 --out $O/needle.json > $O/needle.txt 2>&1; tail -1 $O/needle.txt
cat $O/kv-context.txt | tail -8
grep -E "prefill +[0-9]+:" $O/bench.txt
sed -n '/Throughput by concurrency/,/C6/p' $O/bench.txt
echo "=== sg_final $LBL done $(date -u +%T)"
