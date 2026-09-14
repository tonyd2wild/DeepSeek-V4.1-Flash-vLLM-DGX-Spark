#!/bin/bash
# sg_up.sh <label> (root on Reddie): stop the vLLM lane on all 4 nodes, start the SGLang lane (workers first,
# head last), poll until /v1/models answers (max 50 min), log to /var/tmp/boot-results/sglang/boot-<label>.log.
# Knob env vars (CTX MEMFRAC CHUNK SEQS MAXTOK EXTRA_ARGS EXTRA_ENV PATCH_NAME) are forwarded to sg_node.sh.
# Exit 0 = serving, 2 = a container died, 3 = timeout. Restore vLLM: bash /root/sg_down.sh; bash /root/restore_exl3tp4b_ablit_best.sh
LBL=${1:?label}; D=/var/tmp/boot-results/sglang; mkdir -p $D; L=$D/boot-$LBL.log
exec > >(tee -a "$L") 2>&1
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
P=${PATCH_NAME:-sglang-dsv41-disk}
FWD="CTX=${CTX:-} MEMFRAC=${MEMFRAC:-} CHUNK=${CHUNK:-} SEQS=${SEQS:-} MAXTOK=${MAXTOK:-} EXTRA_ARGS='${EXTRA_ARGS:-}' EXTRA_ENV='${EXTRA_ENV:-}'"
echo "=== sg_up $LBL $(date -u +%T) patches=$P $FWD"
touch /var/tmp/boot-results/speedrun/liveness.stop
echo "--- stop vLLM and any SGLang on all 4"
docker stop -t 60 vllm_dsv41 > /dev/null 2>&1; docker rm -f sglang_dsv41 > /dev/null 2>&1
pids=()
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  $J -n $h 'docker stop -t 60 vllm_dsv41 > /dev/null 2>&1; docker rm -f sglang_dsv41 > /dev/null 2>&1; echo "$(hostname) stopped; MemAvailable $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) GiB"' &
  pids+=($!)
done
wait "${pids[@]}"   # a bare wait would also wait on the tee process substitution and never return
echo "reddie MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB"
T0=$(date -u +%s)
echo "--- start workers (3, 2, 1), then head (0)"
R=3; for h in tonyspark1@192.168.192.1 tonyspark3@192.168.192.3 tonyspark4@192.168.192.4; do
  $J -n $h "$FWD PATCH_DIR=\$HOME/patches/$P bash ~/sg_node.sh $R"; R=$((R-1))
done
eval "$FWD PATCH_DIR=/home/tonyspark2/patches/$P bash /home/tonyspark2/sg_node.sh 0"
for i in $(seq 1 100); do
  sleep 30
  if curl -s -m 5 http://127.0.0.1:8000/v1/models | grep -q deepseek; then echo "SERVING $(date -u +%T) after $(( $(date -u +%s) - T0 ))s"; break; fi
  dead=""
  docker ps --format '{{.Names}}' | grep -q '^sglang_dsv41$' || dead="$dead reddie"
  for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
    $J -n $h "docker ps --format '{{.Names}}' | grep -q '^sglang_dsv41$'" || dead="$dead ${h%%@*}"
  done
  [ $((i % 4)) = 1 ] && echo "$(date -u +%T) $(docker logs --tail 1 sglang_dsv41 2>&1 | cut -c1-160)"
  if [ -n "$dead" ]; then
    echo "BOOT FAILED $(date -u +%T): container gone on:$dead"
    docker logs --tail 40 sglang_dsv41 2>&1 | grep -iE "error|Traceback|raise|Exception|OOM|killed" | tail -15
    for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
      echo "== ${h%%@*}"; $J -n $h "docker logs --tail 30 sglang_dsv41 2>&1 | grep -iE 'error|Traceback|raise|Exception|OOM|killed' | tail -8"
    done
    exit 2
  fi
done
curl -s -m 5 http://127.0.0.1:8000/v1/models | grep -q deepseek || { echo "TIMEOUT $(date -u +%T)"; exit 3; }
docker logs sglang_dsv41 2>&1 | grep -iE "KV Cache is allocated|max_total_num_tokens|#tokens|avail mem|memory calculation|Load weight end|Capture cuda graph end|engram" | sed -E 's/^.*\] //' | tail -12
rm -f /var/tmp/boot-results/speedrun/liveness.stop; setsid nohup bash /root/liveness.sh > /dev/null 2>&1 < /dev/null &
echo "=== sg_up $LBL done $(date -u +%T)"
