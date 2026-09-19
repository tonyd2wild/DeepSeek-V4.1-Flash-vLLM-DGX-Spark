#!/bin/bash
# read-only GPU health check: 10 s fp16 burn on all four (copied from prelaunch-quick.sh)
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
WORKERS="SPARK4:tonyspark4@192.168.192.4 ASUSI:tonyspark3@192.168.192.3 BLUEY:tonyspark1@192.168.192.1"
cat > /tmp/burn-quick.sh <<'B'
( docker run --rm --gpus all --network none --entrypoint python3 vllm-dsv41:overlay5 -c "
import torch, time
a = torch.randn(4096, 4096, dtype=torch.float16, device='cuda'); b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
for _ in range(10): c = a @ b
torch.cuda.synchronize(); t0 = time.time(); n = 0
while time.time() - t0 < 10:
    c = a @ b; n += 1
torch.cuda.synchronize(); dt = time.time() - t0
print('TFLOPS %.1f' % (n*2*4096**3/dt/1e12))
" 2>/dev/null ) &
sleep 6; nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader | tr '\n' ' '
wait
B
( echo "REDDIE $(bash /tmp/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-REDDIE.txt ) &
for p in $WORKERS; do ( echo "${p%%:*} $($J ${p#*:} 'bash -s' < /tmp/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-${p%%:*}.txt ) & done
wait
for n in REDDIE SPARK4 ASUSI BLUEY; do cat /tmp/burn-$n.txt; done
