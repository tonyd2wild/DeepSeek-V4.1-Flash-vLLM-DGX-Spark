#!/usr/bin/env python3
"""readme_tp3.py SERVING_LABEL [--dry-run]: build the README's TP3 lane section from the result files (no hand-copied numbers)
and insert it before "## Vision and tool calling" (replacing an earlier copy), plus repo-layout rows. Kai 2026-09-11."""
import re, sys
lab = sys.argv[1]; dry = "--dry-run" in sys.argv
def thr(label):
    b = open(f"results/{label}/bench-{label}.md").read()
    return {m.group(1): tuple(float(x) for x in m.groups()[1:]) for m in re.finditer(r"^\| (C\d) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$", b, re.M)}
def prefill(label):
    r = open(f"results/{label}/report.md").read().split("### Cold prefill", 1)[1]
    return re.findall(r"^\| (\d+K) \| ([\d,]+) \| ([\d.]+) \| ([\d,]+) \|$", r, re.M)
def vision(label):
    try: return open(f"results/{label}/vision-tools.txt").read().strip().splitlines()[-1].replace("summary: ", "").replace(" PASS", " pass")
    except FileNotFoundError: return "n/a"
KV = {"exl3tp3a11": "678,950", "exl3tp3a10": "1,995,725", "exl3tp3a9": "417,333", "exl3tp3a7": "759,557", "boot10": "1,070,168"}
t, b10 = thr(lab), thr("boot10")
assert len(t) == 6 and len(b10) == 6, (len(t), len(b10))
rows = "\n".join(f"| {C} | {t[C][0]:.1f} | {t[C][1]:.1f} | {t[C][2]:.2f} | {b10[C][0]:.1f} |" for C in sorted(t))
p_lab, p_b10, p_nods = prefill(lab), prefill("boot10"), prefill("exl3tp3a10")
big = lambda pf: next((x for x in pf if x[0] == "64K"), pf[-1])
sec = f"""## Three Sparks: the EXL3 TP3 lane

The same model on **three** DGX Sparks instead of four, using bot-lab-21's [EXL3 3.5 bpw build](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) of the routed experts (everything else is the release's FP8). V4.1 had no TP3 path in vLLM, so this lane adds one: virtual attention heads (64 padded to 72), a vocabulary split that works at TP3, Engram rows for three ranks, a streaming weight loader, and a fix that lets the DSpark drafter's 128 experts load on three ranks. The full write-up, with every boot and what failed, is in [docs/EXL3-TP3.md](docs/EXL3-TP3.md).

Serving config ({lab}): CUDA graphs, DSpark k=5, vision, 300K context, gmu 0.80. Same prompt set and bench as boot 10; throughput across the 8 prompt categories:

| C | TP3 aggregate tok/s | TP3 per-stream tok/s | TP3 mean TTFT (s) | boot 10 aggregate tok/s |
|---|---|---|---|---|
{rows}

| | TP3, DSpark ({lab}) | TP3, no DSpark (exl3tp3a10) | boot 10, 4 Sparks |
|---|---|---|---|
| KV pool, tokens | {KV[lab]} | {KV['exl3tp3a10']} | {KV['boot10']} |
| cold prefill, {big(p_lab)[1]}-token prompt | {big(p_lab)[3]} tok/s | {big(p_nods)[3]} tok/s | {big(p_b10)[3]} tok/s |
| vision and tool checks | {vision(lab)} | {vision('exl3tp3a10')} | 7/7 pass |

"""
layout = ("| `docs/EXL3-TP3.md`, `exl3/` | The EXL3 TP3 lane: write-up and bring-up log, launch and guard scripts (`exl3/try4/`), prep tools, memory evidence. |\n"
          "| `patch/exl3-tp3/` | The TP3 patch set (mounted over the image like `patch/`), with `MD5SUMS.txt`. |\n"
          "| `results/exl3tp3a*/` | Per-boot bench, report, comparisons and vision/tool checks for the TP3 lane. |\n")
s = open("README.md").read()
s = re.sub(r"## Three Sparks: the EXL3 TP3 lane\n.*?(?=## Vision and tool calling)", "", s, flags=re.S)
assert s.count("## Vision and tool calling") == 1
s = s.replace("## Vision and tool calling", sec + "## Vision and tool calling", 1)
if "`docs/EXL3-TP3.md`" not in s:
    m = re.search(r"^\| `bench/` \|.*\|$", s, re.M); assert m
    s = s[:m.end()] + "\n" + layout.rstrip("\n") + s[m.end():]
if dry: print(sec); print(layout)
else: open("README.md", "w").write(s); print("README updated with", lab)
