#!/bin/bash
# reddie_prep_tp3.sh (Reddie, after the download): sha256-verify the 40 EXL3 body shards against the HF LFS oids,
# then build DeepSeek-V4.1-Flash-EXL3-TP3 = hardlinks of the EXL3 dir + a config.json declaring 72 heads / 9 o_groups.
set -u
X=/var/tmp/models/DeepSeek-V4.1-Flash-EXL3-Pollard; T=/var/tmp/models/DeepSeek-V4.1-Flash-EXL3-TP3
cd $X || exit 1
n=$(ls | grep -cE "^model-000[0-9]+-of-00048.safetensors$"); echo "[$(date +%T)] shards present: $n/48"; [ "$n" = 48 ] || { echo "NOT COMPLETE"; exit 2; }
echo "[$(date +%T)] sha256 of 40 body shards (8 parallel)"
cut -c67- /tmp/exl3_body_sha256.txt | xargs -P 8 -n 1 sha256sum > /tmp/exl3_body_sha256.got
sort -k2 /tmp/exl3_body_sha256.txt > /tmp/a; sort -k2 /tmp/exl3_body_sha256.got > /tmp/b
if diff -q /tmp/a /tmp/b >/dev/null; then echo "[$(date +%T)] SHA256 OK 40/40"; else echo "[$(date +%T)] SHA256 MISMATCH:"; diff /tmp/a /tmp/b | head; exit 3; fi
sync; echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null
rm -rf $T && mkdir -p $T && cd $X && for f in $(ls -A | grep -v "^\.cache$"); do cp -al "$f" "$T/$f"; done
rm -f $T/config.json
python3 - <<PY
import json
c = json.load(open("$X/config.json"))
tc = c["text_config"]; assert tc["num_attention_heads"] == 64 and tc["o_groups"] == 8
tc["num_attention_heads"] = 72; tc["o_groups"] = 9
src = {"num_attention_heads": 64, "o_groups": 8}; tc["virtual_heads_from"] = src; c["virtual_heads_from"] = src
c["kai_tp3_virtual_heads"] = "64 real heads / 8 o_groups padded to 72 / 9 for tensor-parallel 3 (virtual_heads.py)"
json.dump(c, open("$T/config.json", "w"), indent=1)
print("TP3 config.json written: heads", tc["num_attention_heads"], "o_groups", tc["o_groups"])
PY
echo "[$(date +%T)] TP3 dir: $(ls $T | grep -cE '^model-000') shards, links on shard 47: $(stat -c %h $T/model-00047-of-00048.safetensors), quantization_config.json $(stat -c %s $T/quantization_config.json) bytes"
df -h /var/tmp | tail -1; echo "[$(date +%T)] PREP DONE"
