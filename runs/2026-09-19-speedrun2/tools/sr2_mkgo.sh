#!/bin/bash
# sr2_mkgo.sh <label>  (root on Reddie): speed run 2 (2026-09-19). Same as sr_mkgo.sh but the base is the LIVE
# 500K config (asusi:~/exl3tp4b-ablit-best500k-go.sh), so every experiment compares to what is served.
# Overrides come on stdin and are inserted right before the launch line; NCCL_EXTRA -e pairs override the launcher's
# (last wins), VLLM_EXTRA flags are appended last to `vllm serve`.
LBL=${1:?label}
BASE=${BASE:-exl3tp4b-ablit-best500k-go.sh}
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
$J -n tonyspark3@192.168.192.3 "cat ~/$BASE" > /tmp/sr2-base-go.sh || { echo "cannot read base go script $BASE"; exit 1; }
grep -q "^bash \$HOME/boot_dsv41_tp4x.sh" /tmp/sr2-base-go.sh || { echo "base has no launch line"; exit 1; }
{ echo "# ---- speed run 2026-09-19 overrides for $LBL (base $BASE)"; cat; echo "export EXP_NAME=exl3tp4b-ablit   # keep the compile/autotune cache"; echo "echo \"speedrun2 $LBL overrides applied\""; } > /tmp/sr2-ov-$LBL.txt
awk 'FNR==NR{ov=ov $0 "\n"; next} /^bash \$HOME\/boot_dsv41_tp4x.sh/{printf "%s", ov} {print}' /tmp/sr2-ov-$LBL.txt /tmp/sr2-base-go.sh > /tmp/sr2-$LBL-go.sh
grep -q "speedrun2 $LBL overrides applied" /tmp/sr2-$LBL-go.sh || { echo "override insertion failed"; exit 1; }
bash -n /tmp/sr2-$LBL-go.sh || { echo "syntax error in generated go script"; exit 1; }
$J tonyspark3@192.168.192.3 "cat > ~/sr2-$LBL-go.sh && chmod +x ~/sr2-$LBL-go.sh" < /tmp/sr2-$LBL-go.sh
echo "wrote asusi:~/sr2-$LBL-go.sh; overrides:"; cat /tmp/sr2-ov-$LBL.txt
