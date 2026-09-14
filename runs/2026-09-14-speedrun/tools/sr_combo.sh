#!/bin/bash
# sr_combo.sh <new-label> <label>...  (root on Reddie): build asusi:~/sr-<new-label>-go.sh by merging the override
# blocks of asusi:~/sr-<label>-go.sh (the lines sr_mkgo.sh inserted, between "# ---- speed run" and "export EXP_NAME"),
# identical lines once, in the given order (a later export of the same knob wins; NCCL_EXTRA / VLLM_EXTRA additions
# accumulate, and a repeated docker -e with the same value is harmless). Then sr_mkgo.sh writes the combined script.
NEW=${1:?new label}; shift
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
: > /tmp/sr-combo-$NEW.txt
for L in "$@"; do
  $J -n tonyspark3@192.168.192.3 "cat ~/sr-$L-go.sh" | awk '/^# ---- speed run/{f=1; next} /^export EXP_NAME/{f=0} f' >> /tmp/sr-combo-$NEW.txt
done
awk '!seen[$0]++' /tmp/sr-combo-$NEW.txt | grep -v '^echo "speedrun' > /tmp/sr-combo-$NEW.uniq
echo "combined overrides for $NEW (from: $*):"; cat /tmp/sr-combo-$NEW.uniq
bash /root/sr_mkgo.sh $NEW < /tmp/sr-combo-$NEW.uniq | tail -1
