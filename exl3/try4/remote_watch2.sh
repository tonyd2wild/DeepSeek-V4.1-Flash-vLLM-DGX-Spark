#!/bin/bash
# remote_watch2.sh (Asusi): key lines from the 3 try-4 ranks (head = Bluey), replayed from container start, line-flushed, spam-filtered.
J='ssh -i /home/tonyspark3/.ssh/id_ed25519_shared -o BatchMode=yes -o ServerAliveInterval=30'
PAT='Application startup complete|Traceback|Error|error:|CUDA out of memory|OutOfMemory|KV cache|Available KV|Loading weights took|Model loading took|node-local|NCCL WARN|Killed|Failed core|refusing|RuntimeError|ValueError|AssertionError|CONTAINER-EXIT|checkpoint shards: 100%'
SKIP='should dump|frame #|DistNetworkError|c10::'
follow(){ L=$1; shift; "$@" "docker logs -f vllm_dsv41 2>&1; echo CONTAINER-EXIT code=\$(docker inspect -f '{{.State.ExitCode}}' vllm_dsv41)" 2>&1 | grep --line-buffered -E "$PAT" | grep --line-buffered -vE "$SKIP" | awk -v l="$L" '{print "[" l "] " substr($0,1,240); fflush()}'; }
follow reddie $J tonyspark2@192.168.192.2 &
follow asusi bash -c &
follow HEAD-bluey $J tonyspark1@192.168.192.1
