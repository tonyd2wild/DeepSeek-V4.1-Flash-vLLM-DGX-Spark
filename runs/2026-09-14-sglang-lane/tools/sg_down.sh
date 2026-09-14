#!/bin/bash
# sg_down.sh (root on Reddie): stop the SGLang lane on all 4 nodes (containers are removed; no files are touched).
# Then the vLLM lane can come back with: bash /root/restore_exl3tp4b_ablit_best.sh
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
docker rm -f sglang_dsv41 > /dev/null 2>&1; echo "reddie sglang stopped"
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  $J -n $h 'docker rm -f sglang_dsv41 > /dev/null 2>&1; echo "$(hostname) sglang stopped"' &
done; wait
