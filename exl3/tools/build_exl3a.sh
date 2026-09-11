#!/bin/bash
# build_exl3a.sh <sha> : runs on Bluey. Memory-capped build of Zeuss5/cuda-exl3 into vllm-dsv41:exl3a (FROM overlay5).
# cgroup cap keeps the compile from touching the live boot-10 worker; no --gpus, so no CUDA context.
set -u
SHA="$1"; W=~/exl3build; mkdir -p $W; cd $W; LOG=$W/build.log; : > $LOG
log(){ echo "[$(date +%T)] $*" | tee -a $LOG; }
log "fetch cuda-exl3 $SHA"
curl -sSL -o src.tgz "https://codeload.github.com/Zeuss5/cuda-exl3/tar.gz/$SHA" || { log "FETCH FAILED"; exit 1; }
rm -rf src && mkdir src && tar xzf src.tgz -C src --strip-components=1 || { log "UNTAR FAILED"; exit 1; }
EP=$(docker inspect vllm-dsv41:overlay5 --format '{{json .Config.Entrypoint}}'); CMD=$(docker inspect vllm-dsv41:overlay5 --format '{{json .Config.Cmd}}')
log "base entrypoint=$EP cmd=$CMD memavail=$(awk '/MemAvailable/{print int($2/1048576)"GiB"}' /proc/meminfo)"
docker rm -f exl3build >/dev/null 2>&1
log "build start (cgroup 3g, MAX_JOBS=1)"
docker run --name exl3build --memory 3g --memory-swap 3g -v $W/src:/src:ro --entrypoint bash vllm-dsv41:overlay5 -c '
  set -eo pipefail; cp -r /src /opt/cuda-exl3; cd /opt/cuda-exl3
  export TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=1
  pip install --no-deps --no-build-isolation . 2>&1 | tail -25
  python3 -c "import cuda_exl3, cuda_exl3._C, torch; print(\"cuda_exl3 import ok\", cuda_exl3.__file__)"
' >> $LOG 2>&1
RC=$?; log "build exit=$RC oom=$(docker inspect exl3build --format '{{.State.OOMKilled}}')"
if [ $RC -ne 0 ]; then log "BUILD FAILED"; exit 1; fi
CH=(--change "ENTRYPOINT $EP"); [ "$CMD" != "null" ] && CH+=(--change "CMD $CMD")
docker commit "${CH[@]}" --change "LABEL kai.exl3a=cuda-exl3-$SHA" exl3build vllm-dsv41:exl3a >> $LOG 2>&1 && log "COMMITTED vllm-dsv41:exl3a $(docker images vllm-dsv41:exl3a --format '{{.ID}} {{.Size}}')"
docker rm exl3build >/dev/null; log "DONE"
