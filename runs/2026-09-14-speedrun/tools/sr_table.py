#!/usr/bin/env python3
"""sr_table.py RESULTS_DIR [BASE_LABEL]: markdown table of every speed-run SCREEN/FINAL result folder in
RESULTS_DIR (each with a bench-*.json), with % deltas against BASE_LABEL (default 00b-baseline-screen).
Columns: KV pool, mean aggregate tok/s over the 8 categories at C1/C3/C6, per-stream C1 code/JSON/math/prose,
cold prefill at the 8K and 32K targets, idle-test back-to-back count and code, quality gate."""
import glob
import json
import os
import re
import statistics as st
import sys

D = sys.argv[1]
BASE = sys.argv[2] if len(sys.argv) > 2 else "00b-baseline-screen"


def load(d):
    js = [p for p in glob.glob(os.path.join(d, "bench-*.json")) if "-warm" not in p]
    if not js:
        return None
    b = json.load(open(sorted(js)[-1]))
    r = {}
    kv = open(os.path.join(d, "kv-context.txt")).read() if os.path.exists(os.path.join(d, "kv-context.txt")) else ""
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", kv)
    r["KV"] = int(m[-1].replace(",", "")) if m else None
    for c in (1, 3, 6):
        bs = [x for x in b["batches"] if x["c"] == c and x["category"] != "ceiling_count"]
        r[f"C{c}"] = st.mean(x["agg_tok_s"] for x in bs) if bs else None
    for cat, k in (("coding", "code"), ("json", "json"), ("math", "math"), ("prose", "prose")):
        x = next((x for x in b["batches"] if x["c"] == 1 and x["category"] == cat), None)
        r[k] = x["per_stream_tok_s"] if x else None
    for p in b.get("prefill", []):
        if p["target"] in (8000, 32000):
            r[f"pf{p['target'] // 1000}K"] = p["prefill_tok_s"]
    it = os.path.join(d, "idletest.txt")
    if os.path.exists(it):
        for line in open(it):
            mm = re.match(r"([BC]) (count|code)[^:]*:.*?([\d.]+) tok/s after first token", line)
            if mm:
                r[f"idle {mm.group(2)}"] = float(mm.group(3))
    q = os.path.join(d, "quality.json")
    r["quality"] = ("PASS" if json.load(open(q)).get("all_pass") else "FAIL") if os.path.exists(q) else "-"
    return r


rows = {os.path.basename(p.rstrip("/")): load(p) for p in sorted(glob.glob(os.path.join(D, "*/")))}
rows = {k: v for k, v in rows.items() if v}
base = rows.get(BASE)
cols = ["KV", "C1", "C3", "C6", "code", "json", "math", "prose", "pf8K", "pf32K", "idle count", "idle code", "quality"]
print("| run | " + " | ".join(cols) + " |")
print("|---|" + "---|" * len(cols))
for name, r in rows.items():
    cells = []
    for c in cols:
        v = r.get(c)
        if isinstance(v, (int, float)):
            s = f"{v:,.0f}" if c == "KV" else f"{v:,.1f}"
            if base and name != BASE and isinstance(base.get(c), (int, float)) and base[c] and not c.startswith("pf"):
                s += f" ({(v / base[c] - 1) * 100:+.1f}%)"
            cells.append(s)
        else:
            cells.append(str(v) if v is not None else "-")
    print(f"| {name} | " + " | ".join(cells) + " |")
print("\n(deltas vs " + BASE + "; prefill cells are cold on each fresh boot except the baseline SCREEN, which hit the prefix cache)")
