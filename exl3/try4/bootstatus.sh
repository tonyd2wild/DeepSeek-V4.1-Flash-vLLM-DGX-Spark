#!/bin/bash
# bootstatus.sh CUT PREPLOG (Asusi): compact snapshot of a TP3 boot since CUT (UTC, docker --since format). Kai 2026-09-11.
CUT=${1:?cut}; PL=${2:-$HOME/prep_try7.log}
J="ssh -n -i $HOME/.ssh/id_ed25519_shared -o ConnectTimeout=6 -o BatchMode=yes"
P='Application startup complete|Traceback|Error|error:|out of memory|OutOfMemory|Loading weights took|Model loading took|GPU KV cache size|Available KV cache|Capturing CUDA graph|Graph capturing finished|NCCL WARN|RuntimeError|AssertionError|ValueError|Killed'
echo "prep: $(grep -aE '^==|FAIL|not launching|STILL RUNNING|exit=|launched' $PL 2>/dev/null | tail -1 | cut -c1-140)"
for h in local tonyspark2@192.168.192.2 tonyspark1@192.168.192.1; do
  case $h in local) c="bash -c"; n=asusi;; *192.168.192.2) c="timeout 15 $J $h"; n=reddie;; *) c="timeout 15 $J $h"; n=bluey;; esac
  $c "docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} started={{.State.StartedAt}}' vllm_dsv41 2>/dev/null | cut -c1-58; docker logs --since $CUT vllm_dsv41 2>&1 | grep -aE '$P' | grep -avE 'Engine 000|loggers.py|frame #|should dump' | tail -4 | cut -c1-200" 2>/dev/null | sed "s/^/[$n] /"
done
grep -aE 'GUARD|low:' $HOME/memguard2.log 2>/dev/null | tail -2 | sed 's/^/[guard] /'
