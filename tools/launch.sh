#!/bin/bash
# usage: launch.sh <N>   (root on Reddie) stop any vllm_dsv41 on all 4 nodes (head first, logs saved),
# then run /tmp/bootN-go.sh on Asusi and wait for the fan-out to finish.
# Stop everything first: a worker that starts while the previous head is still up joins that head's
# TCPStore on the same port and hangs the new boot at distributed init.
N=${1:?boot number}
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
A=tonyspark3@192.168.192.3
P=/var/tmp/prelaunch-logs/$(date -u +%Y%m%dT%H%M%SZ)-before-boot$N; mkdir -p $P
if docker ps -a --format '{{.Names}}' | grep -q '^vllm_dsv41$'; then
  docker logs -t vllm_dsv41 > $P/head.log 2>&1; docker rm -f vllm_dsv41 >/dev/null && echo "stopped head (log saved $P/head.log)"
fi
for p in spark4:tonyspark4@192.168.192.4 asusi:$A bluey:tonyspark1@192.168.192.1; do
  h=${p#*:}; k=${p%%:*}
  if $J $h "docker ps -a --format '{{.Names}}' | grep -q '^vllm_dsv41\$'"; then
    $J $h "docker logs -t vllm_dsv41 2>&1" > $P/worker-$k.log 2>&1; $J $h "docker rm -f vllm_dsv41 >/dev/null" && echo "stopped $k"
  fi
done
chmod 644 $P/*.log 2>/dev/null
# PRELAUNCH overrides the pre-launch step (e.g. /root/prelaunch-quick.sh for a relaunch); PRELAUNCH=none skips it.
PRE=${PRELAUNCH:-/root/prelaunch-$N.sh}
if [ "$PRE" != none ] && [ -f "$PRE" ]; then bash "$PRE" || { echo "prelaunch ($PRE) failed: NOT launching"; exit 1; }; fi
$J $A "nohup bash /tmp/boot$N-go.sh > /tmp/boot$N-go.log 2>&1 < /dev/null & echo launched boot$N pid \$!"
for i in $(seq 1 48); do sleep 5; $J $A "grep -q 'boot_dsv41 exit=' /tmp/boot$N-go.log" && break; done
echo "--- /tmp/boot$N-go.log"; $J $A "tail -14 /tmp/boot$N-go.log" | cut -c1-200
echo "--- containers"; echo "REDDIE $(docker ps -a --filter name=vllm_dsv41 --format '{{.State}} {{.Status}} {{.Image}}')"
for p in SPARK4:tonyspark4@192.168.192.4 ASUSI:$A BLUEY:tonyspark1@192.168.192.1; do echo "${p%%:*} $($J ${p#*:} "docker ps -a --filter name=vllm_dsv41 --format '{{.State}} {{.Status}} {{.Image}}'")"; done
docker inspect vllm_dsv41 --format '{{json .Args}}' 2>/dev/null | python3 -c "import json,sys; a=json.load(sys.stdin); g=lambda k: a[a.index(k)+1] if k in a else None; print('maxlen', g('--max-model-len'), '| block', g('--block-size'), '| spec', g('--speculative-config'))"
