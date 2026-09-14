#!/bin/bash
# sr_mkgo.sh <label>  (root on Reddie): write ~tonyspark3/sr-<label>-go.sh on Asusi = the serving go script
# (exl3tp4b-ablit-go.sh) plus the override lines read from stdin, inserted right before the launch line.
# Overrides that reach every rank: exported knobs forwarded by boot_dsv41_tp4x.sh (MAX_BATCHED, SPEC_K, CUDAGRAPH_MODE,
# GMU, SEQS, ENGRAM_THREADS, ENGRAM_CHUNK, ...), NCCL_EXTRA (docker -e pairs, last value wins, so it can override the
# launcher's own -e settings) and VLLM_EXTRA (appended last to `vllm serve`, so a repeated flag overrides; no spaces in JSON).
# Example: printf 'export NCCL_EXTRA="$NCCL_EXTRA -e NCCL_PROTO=LL"\n' | bash /root/sr_mkgo.sh e01-nccl-ll
LBL=${1:?label}
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
$J -n tonyspark3@192.168.192.3 'cat ~/exl3tp4b-ablit-go.sh' > /tmp/sr-base-go.sh || { echo "cannot read base go script"; exit 1; }
{ echo "# ---- speed run 2026-09-14 overrides for $LBL"; cat; echo "export EXP_NAME=exl3tp4b-ablit   # keep the compile/autotune cache"; echo "echo \"speedrun $LBL overrides applied\""; } > /tmp/sr-ov-$LBL.txt
awk 'FNR==NR{ov=ov $0 "\n"; next} /^bash \$HOME\/boot_dsv41_tp4x.sh/{printf "%s", ov} {print}' /tmp/sr-ov-$LBL.txt /tmp/sr-base-go.sh > /tmp/sr-$LBL-go.sh
grep -q "speedrun $LBL overrides applied" /tmp/sr-$LBL-go.sh || { echo "override insertion failed"; exit 1; }
bash -n /tmp/sr-$LBL-go.sh || { echo "syntax error in generated go script"; exit 1; }
$J tonyspark3@192.168.192.3 "cat > ~/sr-$LBL-go.sh && chmod +x ~/sr-$LBL-go.sh" < /tmp/sr-$LBL-go.sh
echo "wrote asusi:~/sr-$LBL-go.sh; overrides:"; cat /tmp/sr-ov-$LBL.txt
