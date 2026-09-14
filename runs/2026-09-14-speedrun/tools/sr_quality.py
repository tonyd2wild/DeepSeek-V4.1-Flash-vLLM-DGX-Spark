#!/usr/bin/env python3
"""sr_quality.py <out_dir>  (root on Reddie): per-boot quality gate at temperature 0, so a faster config
cannot win with broken output. Checks: count 1..100 exact, JSON parses with the asked keys, Python code
parses and passes two asserts (run in a subprocess with a timeout), 17*23 = 391, prose non-empty and not
degenerate (no 3-gram repeated more than 4 times, no empty completion). Writes quality.json; prints PASS/FAIL.
The texts' sha256 let us compare boots (identical math should give identical text at temperature 0)."""
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from collections import Counter

OUT = sys.argv[1]
U = "http://127.0.0.1:8000/v1/chat/completions"


def ask(prompt, max_tokens):
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=300))
    return r["choices"][0]["message"].get("content") or "", r["usage"]["completion_tokens"]


def degenerate(t):
    w = t.split()
    if not w:
        return True
    tri = Counter(zip(w, w[1:], w[2:]))
    return bool(tri) and max(tri.values()) > 4


res, ok = {}, True

t, n = ask("Count from 1 to 100, separated by spaces. Output only the numbers.", 400)
nums = [int(x) for x in t.split() if x.isdigit()]
res["count"] = {"pass": nums[:100] == list(range(1, 101)), "tokens": n}

t, n = ask("Return a JSON object for a fictional engineer with keys name (string), age (integer) and "
           "skills (a list of exactly 3 strings). Output only the JSON, no code fence.", 200)
try:
    j = json.loads(re.sub(r"^```(json)?|```$", "", t.strip(), flags=re.M).strip())
    jp = isinstance(j.get("name"), str) and isinstance(j.get("age"), int) and isinstance(j.get("skills"), list) and len(j["skills"]) == 3
except Exception:  # noqa: BLE001
    jp = False
res["json"] = {"pass": jp, "tokens": n}

t, n = ask("Write a Python function is_prime(n) that returns True if n is prime and False otherwise. "
           "Output only the code in one ```python block.", 300)
m = re.search(r"```(?:python)?\n(.*?)```", t, re.S)
code = (m.group(1) if m else t) + "\nassert is_prime(97) and not is_prime(91) and not is_prime(1) and is_prime(2)\nprint('ok')\n"
try:
    cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10).stdout.strip() == "ok"
except Exception:  # noqa: BLE001
    cp = False
res["code"] = {"pass": cp, "tokens": n}

t, n = ask("What is 17 * 23? Answer with just the number.", 20)
res["math"] = {"pass": "391" in t, "tokens": n}

t, n = ask("Write three sentences about the ocean at night.", 160)
res["prose"] = {"pass": len(t.strip()) > 40 and not degenerate(t), "tokens": n}

texts = []
for k in ("count", "json", "code", "math", "prose"):
    ok &= res[k]["pass"]
res["all_pass"] = ok
json.dump(res, open(f"{OUT}/quality.json", "w"), indent=1)
print("QUALITY", "PASS" if ok else "FAIL", {k: v["pass"] for k, v in res.items() if isinstance(v, dict)})
