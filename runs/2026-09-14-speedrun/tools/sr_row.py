#!/usr/bin/env python3
"""sr_row.py RUN_DIR [BASE_DIR]: one experiment row from a speed-run SCREEN result folder (local copy of
/var/tmp/boot-results/speedrun/<label>/), optionally with % deltas against a baseline SCREEN folder.
Row: KV pool, mean aggregate tok/s over the 8 categories at each level, per-stream decode for code/json/math/prose
at C1, counting C1, prefill tok/s per target, idle-test count/code tok/s."""
import glob
import json
import os
import re
import statistics as st
import sys


def load(d):
    js = [p for p in glob.glob(os.path.join(d, "bench-*.json")) if "-warm" not in p]
    if not js:
        return None
    b = json.load(open(sorted(js)[-1]))
    out = {"label": b["label"]}
    kv = open(os.path.join(d, "kv-context.txt")).read() if os.path.exists(os.path.join(d, "kv-context.txt")) else ""
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", kv)
    out["kv"] = int(m[-1].replace(",", "")) if m else None
    cats = sorted({x["category"] for x in b["batches"]} - {"ceiling_count"})
    for c in sorted({x["c"] for x in b["batches"]}):
        bs = [x for x in b["batches"] if x["c"] == c and x["category"] != "ceiling_count"]
        out[f"C{c} agg"] = st.mean(x["agg_tok_s"] for x in bs)
    for cat in ("coding", "json", "math", "prose", "narrative", "ceiling_count"):
        x = next((x for x in b["batches"] if x["c"] == 1 and x["category"] == cat), None)
        if x and x["per_stream_tok_s"]:
            out[f"C1 {cat}"] = x["per_stream_tok_s"]
    for p in b.get("prefill", []):
        out[f"prefill {p['target'] // 1000}K"] = p["prefill_tok_s"]
    it = os.path.join(d, "idletest.txt")
    if os.path.exists(it):
        for line in open(it):
            m = re.match(r"([A-E]) (count|code)[^:]*:.*?([\d.]+) tok/s after first token", line)
            if m:
                out[f"idle {m.group(1)} {m.group(2)}"] = float(m.group(3))
    return out


r = load(sys.argv[1])
base = load(sys.argv[2]) if len(sys.argv) > 2 else None
if r is None:
    sys.exit(f"no bench json in {sys.argv[1]}")
for k, v in r.items():
    if k == "label":
        print(f"{'label':16} {v}")
        continue
    s = f"{v:,.1f}" if isinstance(v, float) else (f"{v:,}" if v is not None else "-")
    if base and isinstance(v, (int, float)) and isinstance(base.get(k), (int, float)) and base[k]:
        s += f"   ({(v / base[k] - 1) * 100:+.1f}% vs {base[k]:,.1f})"
    print(f"{k:16} {s}")
