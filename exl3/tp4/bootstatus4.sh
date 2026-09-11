#!/bin/bash
# bootstatus4.sh CUT PREPLOG (Asusi): compact snapshot of a 4-Spark boot since CUT (UTC, docker --since format).
CUT=${1:?cut}; PL=${2:-$HOME/prep_tp4.log}
J="ssh -n -i $HOME/.ssh/id_ed25519_shared -o ConnectTimeout=6 -o BatchMode=yes"
P='Application startup complete|Traceback|Error|error:|out of memory|OutOfMemory|Loading weights took|Model loading took|GPU KV cache size|Available KV cache|Graph capturing finished|NCCL WARN|RuntimeError|AssertionError|ValueError|Killed|node-local'
echo "prep: $(grep -aE '^==|FAIL|not launching|STILL RUNNING|MISSING|exit=|launched' $PL 2>/dev/null | tail -1 | cut -c1-140)"
for h in local tonyspark2@192.168.192.2 tonyspark1@192.168.192.1 tonyspark4@192.168.192.4; do
  case $h in local) c="bash -c"; n=asusi;; *192.168.192.2) c="timeout 15 $J $h"; n=reddie;; *192.168.192.1) c="timeout 15 $J $h"; n=bluey;; *) c="timeout 15 $J $h"; n=spark4;; esac
  $c "docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} started={{.State.StartedAt}}' vllm_dsv41 2>/dev/null | cut -c1-58; docker logs --since $CUT vllm_dsv41 2>&1 | grep -aE '$P' | grep -avE 'Engine 000|loggers.py|frame #|should dump' | tail -4 | cut -c1-200" 2>/dev/null | sed "s/^/[$n] /"
done
grep -aE 'GUARD|low:' $HOME/memguard4.log 2>/dev/null | tail -2 | sed 's/^/[guard] /'
