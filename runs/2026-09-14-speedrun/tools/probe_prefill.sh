#!/bin/bash
# probe_prefill.sh (root on Reddie): one ~45K-token cold prefill while sampling, on Reddie (rank 0) and Spark4 (rank 1):
#   per-thread CPU of the vLLM worker (top -H, 2 s) and GPU util/power/clock (nvidia-smi, 0.5 s).
# Tells whether prefill is GPU-bound or held up by host work (e.g. the engram-disk read pool).
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
OUT=/var/tmp/boot-results/speedrun/probe-prefill-${1:-a}; mkdir -p $OUT
S='P=$(pgrep -f "VLLM::Worker" | head -1); ( timeout 50 nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm --format=csv,noheader,nounits -lms 500 > /tmp/probe-gpu.txt 2>&1 & ); top -H -b -d 2 -n 22 -p $P -w 250 | awk "/^top -/{t++} \$1 ~ /^[0-9]+\$/ && \$9+0 >= 3 {print t, \$9, \$12}"; cat /tmp/probe-gpu.txt | sed "s/^/GPU /"'
( bash -c "$S" > $OUT/samp-reddie.txt 2>&1 ) &
( $J -n tonyspark4@192.168.192.4 "$S" > $OUT/samp-spark4.txt 2>&1 ) &
sleep 4
python3 - > $OUT/request.txt 2>&1 <<'PY'
import json, random, time, urllib.request
random.seed(45000)
w = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()
text = " ".join(random.choice(w) for _ in range(24000))
body = {"model": "deepseek-v4.1-flash", "max_tokens": 1, "temperature": 0,
        "messages": [{"role": "user", "content": f"[probe {time.time()}] " + text + "\n\nReply with OK."}]}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(),
                                                            {"Content-Type": "application/json"}), timeout=900))
dt = time.time() - t
print(f"prompt_tokens {r['usage']['prompt_tokens']} ttft_s {dt:.2f} prefill_tok_s {r['usage']['prompt_tokens'] / dt:.0f}")
PY
wait
cat $OUT/request.txt
for n in reddie spark4; do
  echo "--- $n: top worker threads by mean CPU% during the probe (samples with >=3%)"
  grep -v '^GPU' $OUT/samp-$n.txt | awk '{c[$3]+=$2; k[$3]++} END {for (t in c) printf "%-18s mean %6.1f%%  in %d samples\n", t, c[t]/k[t], k[t]}' | sort -k3 -nr | head -12
  echo "  engram-disk threads total CPU (sum of means): $(grep -v '^GPU' $OUT/samp-$n.txt | awk '$3 ~ /engram/ {c[$3]+=$2; k[$3]++} END {s=0; for (t in c) s+=c[t]/k[t]; printf "%.0f%% over %d threads", s, length(c)}')"
  echo "  GPU util/power/clock samples: $(grep '^GPU' $OUT/samp-$n.txt | awk -F'[ ,]+' '{u+=$2; p+=$3; n++} END {if (n) printf "n=%d util %.0f%% power %.1f W", n, u/n, p/n}')"
done
