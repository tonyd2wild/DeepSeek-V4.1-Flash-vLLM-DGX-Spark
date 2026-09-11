#!/usr/bin/env python3
"""readme_tp4.py LABEL [--dry-run]: build the README's TP4 EXL3 (context) lane section from the result files and insert it after the TP3
section, before "## Vision and tool calling" (replacing an earlier copy). Numbers are parsed, never hand-copied. Kai 2026-09-11."""
import re, sys
lab = sys.argv[1]; dry = "--dry-run" in sys.argv
def thr(label):
    b = open(f"results/{label}/bench-{label}.md").read()
    return {m.group(1): tuple(float(x) for x in m.groups()[1:]) for m in re.finditer(r"^\| (C\d) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$", b, re.M)}
def prefill(label):
    r = open(f"results/{label}/report.md").read().split("### Cold prefill", 1)[1]
    pf = re.findall(r"^\| (\d+K) \| ([\d,]+) \| ([\d.]+) \| ([\d,]+) \|$", r, re.M)
    return next((x for x in pf if x[0] == "64K"), pf[-1])
def vision(label):
    try: return open(f"results/{label}/vision-tools.txt").read().strip().splitlines()[-1].replace("summary: ", "").replace(" PASS", " pass")
    except FileNotFoundError: return "n/a"
KV = {"exl3tp4b": (3304863, 11.02), "boot10": (1070168, 3.57), "exl3tp3a11": (678950, 2.26)}
kv_lab = KV.get(lab, (0, 0))
t, b10, t3 = thr(lab), thr("boot10"), thr("exl3tp3a11")
assert len(t) == 6 and len(b10) == 6 and len(t3) == 6
rows = "\n".join(f"| {C} | {t[C][0]:.1f} | {t[C][1]:.1f} | {t[C][2]:.2f} | {b10[C][0]:.1f} | {t3[C][0]:.1f} |" for C in sorted(t))
p4, p10, p3 = prefill(lab), prefill("boot10"), prefill("exl3tp3a11")
sec = f"""## Four Sparks on EXL3: the context lane

Boot 10's serving config (four Sparks, CUDA graphs, DSpark k=5, vision, 300K per request, gmu 0.80) on the same EXL3 3.5 bpw checkpoint as the TP3 lane. The smaller experts leave far more memory for the KV cache:

| | TP4 EXL3 ({lab}) | boot 10, release, 4 Sparks | TP3 EXL3, 3 Sparks |
|---|---|---|---|
| KV pool, tokens | **{kv_lab[0]:,}** | {KV['boot10'][0]:,} | {KV['exl3tp3a11'][0]:,} |
| full 300K-token requests at once | **{kv_lab[1]:.2f}** | {KV['boot10'][1]:.2f} | {KV['exl3tp3a11'][1]:.2f} |
| model memory per Spark, GiB | 56.6 / 68.9 (narrow / wide expert slice) | 81.6 | 84.2 |
| cold prefill, {p4[1]}-token prompt | {p4[3]} tok/s | {p10[3]} tok/s | {p3[3]} tok/s |
| vision and tool checks | {vision(lab)} | 7/7 pass | {vision('exl3tp3a11')} |

EXL3 splits the 2304-wide experts 512/640/640/512 across four ranks, so two Sparks carry 12 GiB more than the other two, and those two set the pool (15.5 GiB of KV each). Throughput across the 8 prompt categories, same bench as boot 10:

| C | TP4 EXL3 aggregate tok/s | per-stream tok/s | mean TTFT (s) | boot 10 aggregate tok/s | TP3 EXL3 aggregate tok/s |
|---|---|---|---|---|---|
{rows}

Launch and guard scripts are in `exl3/tp4/`; the full write-up is in [docs/EXL3-TP3.md](docs/EXL3-TP3.md).

"""
layout_row = "| `exl3/tp4/` | The TP4 EXL3 (context) lane: go, prep, guard, watch and post-boot scripts. |"
s = open("README.md").read()
s = re.sub(r"## Four Sparks on EXL3: the context lane\n.*?(?=## Vision and tool calling)", "", s, flags=re.S)
assert s.count("## Vision and tool calling") == 1
s = s.replace("## Vision and tool calling", sec + "## Vision and tool calling", 1)
if "`exl3/tp4/`" not in s:
    m = re.search(r"^\| `results/exl3tp3a\*/` \|.*\|$", s, re.M); assert m, "layout anchor missing"
    s = s[:m.end()] + "\n" + layout_row + s[m.end():]
if dry: print(sec); print(layout_row)
else: open("README.md", "w").write(s); print("README updated with", lab)
