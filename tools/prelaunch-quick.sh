#!/bin/bash
# prelaunch-quick.sh (root on Reddie): fast health check before relaunching an existing deployment,
# e.g. after a Spark reboot. launch.sh runs it after every vllm_dsv41 container is stopped:
#   PRELAUNCH=/root/prelaunch-quick.sh bash /root/launch.sh 10
# It exits non-zero (launch.sh then does NOT launch) when:
#   - a worker is unreachable or its NFS model mount can't be restored,
#   - a worker has no node-local Engram copy (set ALLOW_NFS_ENGRAM=1 to launch anyway on NFS reads),
#   - a node's staged dsv41-tp4.sh or boot patch differs from Reddie's,
#   - a GPU burns under 50 TFLOPS (GB10 EC clock latch: shut that Spark down, unplug its adapter 30-60 s).
# It also re-stages /tmp/boot$N-go.sh on Asusi from /root/final (a reboot empties /tmp).
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
N=${BOOT:-10}
WORKERS="SPARK4:tonyspark4@192.168.192.4 ASUSI:tonyspark3@192.168.192.3 BLUEY:tonyspark1@192.168.192.1"
A=tonyspark3@192.168.192.3
IMG=vllm-dsv41:overlay5
fail=0
EXP_TP4=$(md5sum < /home/tonyspark2/dsv41-tp4.sh | cut -c1-8)
EXP_ENG=$(md5sum < /home/tonyspark2/patches/dsv41-boot$N/engram.py | cut -c1-8)
echo "=== prelaunch-quick $(date -u +%T): boot $N, expect dsv41-tp4.sh $EXP_TP4, dsv41-boot$N/engram.py $EXP_ENG"

for p in $WORKERS; do
  n=${p%%:*}; h=${p#*:}
  out=$($J -n $h "
    timeout 10 test -f /mnt/reddie-models/DeepSeek-V4.1-Flash/config.json || { sudo -n umount -l /mnt/reddie-models 2>/dev/null; sudo -n mount -t nfs -o ro,vers=3 192.168.192.2:/var/tmp/models /mnt/reddie-models && echo remounted; }
    timeout 10 test -f /mnt/reddie-models/DeepSeek-V4.1-Flash/model-00048-of-00048.safetensors && echo mount=ok || echo mount=FAIL
    test -f /var/tmp/engram-local/DeepSeek-V4.1-Flash/engram-local.json && echo engram_local=ok || echo engram_local=MISSING
    echo tp4=\$(md5sum < ~/dsv41-tp4.sh | cut -c1-8) eng=\$(md5sum < ~/patches/dsv41-boot$N/engram.py 2>/dev/null | cut -c1-8)
    echo up_since=\$(uptime -s | tr ' ' T)" 2>&1 | tr '\n' ' ')
  echo "$n: ${out:-UNREACHABLE}"
  case "$out" in *mount=ok*) ;; *) echo "  FAIL $n: unreachable or model mount missing"; fail=1;; esac
  case "$out" in *engram_local=ok*) ;; *)
    if [ "${ALLOW_NFS_ENGRAM:-0}" = 1 ]; then echo "  WARN $n: no node-local Engram copy, this rank reads NFS";
    else echo "  FAIL $n: no node-local Engram copy (tools/engram_local.py, or ALLOW_NFS_ENGRAM=1)"; fail=1; fi;; esac
  echo "$out" | grep -q "tp4=$EXP_TP4 eng=$EXP_ENG" || { echo "  FAIL $n: staged files differ from Reddie"; fail=1; }
done

if [ -f /root/final/boot$N-go.sh ]; then
  $J $A "cat > /tmp/boot$N-go.sh" < /root/final/boot$N-go.sh
  echo "asusi /tmp/boot$N-go.sh $($J -n $A "md5sum < /tmp/boot$N-go.sh | cut -c1-8") (from /root/final)"
else
  echo "  FAIL: /root/final/boot$N-go.sh missing"; fail=1
fi

# GPU health: wait for memory to come back after the stop, then a 10 s fp16 burn on all four at once.
gib() { awk '/^MemAvailable:/{print int($2/1048576)}'; }
for i in $(seq 1 24); do
  ok=1; [ "$(gib < /proc/meminfo)" -ge 90 ] || ok=0
  for p in $WORKERS; do w=$($J -n ${p#*:} 'cat /proc/meminfo' | gib); [ "${w:-0}" -ge 90 ] || ok=0; done
  [ $ok = 1 ] && break; sleep 5
done
cat > /tmp/burn-quick.sh <<'B'
( docker run --rm --gpus all --network none --entrypoint python3 vllm-dsv41:overlay5 -c "
import torch, time
a = torch.randn(4096, 4096, dtype=torch.float16, device='cuda'); b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
for _ in range(10): c = a @ b
torch.cuda.synchronize(); t0 = time.time(); n = 0
while time.time() - t0 < 10:
    c = a @ b; n += 1
torch.cuda.synchronize(); print(f'TFLOPS {2*4096**3*n/(time.time()-t0)/1e12:.1f}')
" 2>&1 | grep TFLOPS ) &
sleep 8; s=$(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader | tr '\n' ' '); wait
echo "under load: $s"
B
echo "--- 10 s fp16 burn (healthy about 75-90 TFLOPS, 2.2-2.4 GHz, 80 W+; latched about 700-950 MHz)"
( echo "REDDIE $(bash /tmp/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-REDDIE.txt ) &
for p in $WORKERS; do ( echo "${p%%:*} $($J ${p#*:} 'bash -s' < /tmp/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-${p%%:*}.txt ) & done
wait
for n in REDDIE SPARK4 ASUSI BLUEY; do
  l=$(cat /tmp/burn-$n.txt); echo "$l"
  t=$(echo "$l" | grep -oE 'TFLOPS [0-9.]+' | awk '{print int($2)}')
  [ "${t:-0}" -ge 50 ] || { echo "  FAIL $n: GPU burn ${t:-none} TFLOPS (latched or no GPU): power-cycle with the adapter unplugged 30-60 s"; fail=1; }
done
echo "=== prelaunch-quick done $(date -u +%T): $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
