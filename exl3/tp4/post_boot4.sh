#!/bin/bash
# post_boot.sh <label> (Reddie): wait for /health, count + tool-call smokes, then Tony's v41bench (prompt set v1).
set -u; L="${1:?label}"; OUT=/var/tmp/boot-results/$L; mkdir -p $OUT
until curl -sf -m 5 http://192.168.192.2:8000/health >/dev/null; do sleep 15; done; echo "[$(date +%T)] /health OK"
python3 - <<'PY' | tee $OUT/smoke.txt
import json, time, urllib.request
def chat(msgs, **kw):
    body = {"model": "deepseek-v4.1-flash", "messages": msgs, "temperature": 0, "max_tokens": 300, **kw}
    r = urllib.request.Request("http://192.168.192.2:8000/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(r, timeout=300)); return d, time.time() - t
d, s = chat([{"role": "user", "content": "Count from 1 to 30, comma separated, nothing else."}])
txt = d["choices"][0]["message"]["content"]; ok = txt.replace(" ", "").startswith("1,2,3,4,5") and "30" in txt
print(f"count smoke: {'PASS' if ok else 'FAIL'} | {d['usage']['completion_tokens']} tok in {s:.1f}s | {txt[:120]!r}")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
d, s = chat([{"role": "user", "content": "What's the weather in Atlanta right now?"}], tools=tools, tool_choice="auto")
tc = d["choices"][0]["message"].get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "atlanta" in tc[0]["function"]["arguments"].lower()
print(f"tool smoke: {'PASS' if ok else 'FAIL'} | {tc[0]['function'] if tc else d['choices'][0]['message'].get('content','')[:120]}")
d, s = chat([{"role": "user", "content": "In one sentence: why is the sky blue?"}])
print("coherence:", repr(d["choices"][0]["message"]["content"][:200]))
PY
echo "[$(date +%T)] bench start"; python3 ~/v41bench.py --base http://192.168.192.2:8000/v1 --model deepseek-v4.1-flash --label "$L" --out $OUT --notes "${NOTES:-}" > $OUT/bench.txt 2>&1
echo "[$(date +%T)] bench rc=$? -> $OUT"; ls $OUT; echo "POST_BOOT DONE $L"
