#!/bin/bash
# sr_final.sh <label> [base-json]  (root on Reddie): the FULL measurement for a finished config, same as the
# baseline: KV/context milestones, vision + tool checks, quality gate, idle test, a discarded warm-up pass, the full
# v41bench C1-C6 over all categories + cold prefill 2K/8K/32K/64K, then bench_report.py against the baseline JSON.
LBL=${1:?label}; BASE=${2:-/var/tmp/boot-results/speedrun/00-baseline/bench-speedrun-00-baseline.json}
O=/var/tmp/boot-results/speedrun/$LBL; mkdir -p $O
echo "=== sr_final $LBL $(date -u +%T)"
docker logs vllm_dsv41 2>&1 | grep -E "GPU KV cache size|Available KV cache|Model loading took|Graph capturing finished|non-default args|Engram FAST|node-local" | sed -E "s/^.*\] //" > $O/kv-context.txt
docker inspect vllm_dsv41 --format '{{json .Args}}' > $O/args.json
docker inspect vllm_dsv41 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E "NCCL|VLLM|PYTORCH|DSV41|CUDA_EXL3|TORCH" > $O/env.txt
python3 /root/vision_tools_demo.py > $O/vision-tools.txt 2>&1; tail -1 $O/vision-tools.txt
python3 /root/sr_quality.py $O > $O/quality.txt 2>&1; cat $O/quality.txt
python3 /root/idletest.py > $O/idletest.txt 2>&1
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label $LBL-warm --out $O/warm --levels 1 --prefill "" > $O/warm.txt 2>&1 || true
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label $LBL --out $O --levels 1,2,3,4,5,6 --prefill 2000,8000,32000,64000 --notes "speed run 2026-09-14 FINAL $LBL" > $O/bench.txt 2>&1
echo "bench exit $? $(date -u +%T)"
J=$(ls $O/bench-$LBL.json 2>/dev/null)
[ -n "$J" ] && python3 /root/bench_report.py "$J" --vs "$BASE" > $O/report.md 2>&1
grep -E "GPU KV cache size" $O/kv-context.txt | tail -1
grep -E "prefill +[0-9]+:" $O/bench.txt
sed -n '/Throughput by concurrency/,/^$/p' $O/bench.txt | head -12
echo "=== sr_final $LBL done $(date -u +%T)"
