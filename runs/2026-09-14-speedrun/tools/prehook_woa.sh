#!/bin/bash
# prehook: run the wo_a native-MXFP8 standalone test on Bluey (one GPU, synthetic weights, exl3b image) while the
# previous experiment's server is idle. Output -> /var/tmp/boot-results/speedrun/test-woa.txt (on Reddie).
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
H=tonyspark1@192.168.192.1
O=/var/tmp/boot-results/speedrun/test-woa.txt
echo "=== wo_a test on Bluey $(date -u +%T)" > $O
tar cf - -C /root woa-draft | $J $H 'rm -rf /tmp/woa-draft; tar xf - -C /tmp && ls /tmp/woa-draft' >> $O 2>&1
$J -n $H 'm=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); echo "MemAvailable ${m}G"; [ "$m" -ge 20 ] || { echo "SKIP: under 20G free while the server is up"; exit 0; }; timeout 600 docker run --rm --gpus all --network none --memory 8g -v /tmp/woa-draft:/w -w /w --entrypoint bash vllm-dsv41:exl3b -c "cp woa_sm12x.py \$(python3 -c \"import vllm,os;print(os.path.dirname(vllm.__file__))\")/models/deepseek_v4_1/ && cp new/o_proj.py \$(python3 -c \"import vllm,os;print(os.path.dirname(vllm.__file__))\")/models/deepseek_v4/nvidia/ops/o_proj.py && python3 test_woa_sm12x.py" 2>&1 | grep -vE "^WARNING|UserWarning|warnings.warn" | tail -60; echo "rc=${PIPESTATUS[0]}"' >> $O 2>&1
echo "=== done $(date -u +%T)" >> $O
