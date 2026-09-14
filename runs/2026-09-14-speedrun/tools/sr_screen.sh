#!/bin/bash
# sr_screen.sh <label> [levels] [prefill]  (root on Reddie): the SCREEN protocol against the serving endpoint.
#   1) KV / context milestones from the head log
#   2) warm-up: idletest.py (5 streamed requests, 45 s idle in the middle) + one discarded C1 pass of every category
#   3) screen bench: v41bench.py --levels 1,3,6 --prefill 8000,32000 (overridable)
# Output: /var/tmp/boot-results/speedrun/<label>/
LBL=${1:?label}; LV=${2:-1,3,6}; PF=${3:-8000,32000}
O=/var/tmp/boot-results/speedrun/$LBL; mkdir -p $O
echo "=== sr_screen $LBL $(date -u +%T) levels=$LV prefill=$PF"
docker logs vllm_dsv41 2>&1 | grep -E "GPU KV cache size|Available KV cache|Model loading took|Graph capturing finished|non-default args" | sed -E "s/^.*\] //" > $O/kv-context.txt
docker inspect vllm_dsv41 --format '{{json .Args}}' > $O/args.json
docker inspect vllm_dsv41 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E "NCCL|VLLM|PYTORCH|DSV41|CUDA_EXL3|TORCH" > $O/env.txt
python3 /root/idletest.py > $O/idletest.txt 2>&1
python3 /root/sr_quality.py $O > $O/quality.txt 2>&1; cat $O/quality.txt
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label $LBL-warm --out $O/warm --levels 1 --prefill "" > $O/warm.txt 2>&1 || true
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label $LBL --out $O --levels $LV --prefill $PF --notes "speed run 2026-09-14 SCREEN $LBL" > $O/bench.txt 2>&1
echo "bench exit $? $(date -u +%T)"
bash /root/probe_prefill.sh $LBL > $O/probe.txt 2>&1   # ~40K-token cold prefill: GPU util + worker threads
grep -E "GPU KV cache size" $O/kv-context.txt | tail -1
grep -E "^A |^B |^C |^D |^E " $O/idletest.txt | cut -c1-120
tail -30 $O/bench.txt
