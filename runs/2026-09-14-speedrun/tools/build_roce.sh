#!/bin/bash
# build_roce.sh (root on Reddie, detached): waits until E05's boot has started (E04 screen and the prehook are
# done, so no benchmark is running), then builds vllm-dsv41:exl3a-roce on all 4 nodes in parallel from
# /root/roce (Dockerfile.roce: b12x comm.roce overlay at b58f34ea, by @original-el8 and @lukealonso).
# On success on all 4 nodes it puts e19-roce at the top of the dynamic queue.
D=/var/tmp/boot-results/speedrun
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
# (rebuild 07:33: no wait; exl3a has no b12x, so the base is exl3b)

echo "build start $(date -u +%T)"
( cd /root/roce && docker build -f Dockerfile.roce --build-arg BASE=vllm-dsv41:exl3b -t vllm-dsv41:exl3b-roce . > $D/build-roce-reddie.log 2>&1; echo "reddie rc=$?" ) &
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  n=${h%%@*}
  ( tar cf - -C /root roce | $J $h 'rm -rf ~/roce-build && mkdir ~/roce-build && tar xf - -C ~/roce-build && cd ~/roce-build/roce && docker build -f Dockerfile.roce --build-arg BASE=vllm-dsv41:exl3b -t vllm-dsv41:exl3b-roce .' > $D/build-roce-$n.log 2>&1; echo "$n rc=$?" ) &
done
wait
echo "build end $(date -u +%T)"
ok=1
for n in reddie tonyspark4 tonyspark3 tonyspark1; do
  if grep -q "roce API_VERSION 1" $D/build-roce-$n.log && grep -qiE "naming to .*exl3b-roce|Successfully tagged" $D/build-roce-$n.log; then echo "$n OK"; else echo "$n FAILED"; ok=0; fi
done
if [ $ok = 1 ]; then
  { echo e19-roce; cat $D/queue.txt; } > $D/queue.txt.new && mv $D/queue.txt.new $D/queue.txt
  echo "e19-roce queued first $(date -u +%T)"; cat $D/queue.txt
else
  echo "not queued: a build failed"
fi
