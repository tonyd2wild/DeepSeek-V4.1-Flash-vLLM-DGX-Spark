#!/bin/bash
# Roll vllm-dsv41:exl3b-roce-lim from Reddie to the three workers (docker save | ssh | docker load), one at a time,
# with a size check. Only the image moves; nothing running is touched.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
IMG=vllm-dsv41:exl3b-roce-lim
echo "=== roll $IMG $(date -u +%T)"
src=$(docker image inspect $IMG --format '{{.Id}}')
for p in SPARK4:tonyspark4@192.168.192.4 ASUSI:tonyspark3@192.168.192.3 BLUEY:tonyspark1@192.168.192.1; do
  n=${p%%:*}; h=${p#*:}
  have=$($J -n $h "docker image inspect $IMG --format '{{.Id}}' 2>/dev/null")
  if [ "$have" = "$src" ]; then echo "$n already has $IMG"; continue; fi
  t0=$(date +%s)
  docker save $IMG | $J $h "docker load" | tail -1 | sed "s/^/$n /"
  echo "$n done in $(( $(date +%s) - t0 ))s, id=$($J -n $h "docker image inspect $IMG --format '{{.Id}}' 2>/dev/null" | cut -c8-19)"
done
echo "=== roll done $(date -u +%T)"
