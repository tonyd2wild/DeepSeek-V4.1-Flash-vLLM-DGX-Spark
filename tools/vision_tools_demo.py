#!/usr/bin/env python3
"""Vision and tool-calling checks against the served endpoint (OpenAI-compatible API).
Images are generated in pure Python (no PIL), so the expected answers are known exactly.

  V1  one image: three vertical stripes (red, green, blue), ask for the order left to right
  V2  two images in one message: a red square and a blue square, ask which comes first
  V3  image + text reasoning: a 2x2 grid of colored quadrants, ask for the top-right color
  T1  tool call: the model must call get_weather with the right arguments
  T2  full round trip: send the tool result back, the model must answer from it
  T3  two tools offered, parallel calls: weather in two cities
  T4  tool_choice forcing a named function

usage: vision_tools_demo.py [BASE_URL] (default http://127.0.0.1:8000/v1)"""
import base64
import json
import struct
import sys
import time
import urllib.request
import zlib

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
M = "deepseek-v4.1-flash"


def post(body):
    req = urllib.request.Request(f"{BASE}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r, time.time() - t


def png(w, h, pixel):
    raw = b"".join(b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(w)) for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


RED, GREEN, BLUE, YELLOW = (220, 30, 30), (30, 170, 60), (30, 60, 220), (240, 210, 30)
stripes = png(192, 96, lambda x, y: (RED, GREEN, BLUE)[x * 3 // 192])
red_sq = png(96, 96, lambda x, y: RED)
blue_sq = png(96, 96, lambda x, y: BLUE)
grid = png(128, 128, lambda x, y: ((RED, GREEN), (BLUE, YELLOW))[y // 64][x // 64])


def ask(tag, parts, expect):
    r, dt = post({"model": M, "max_tokens": 40, "temperature": 0, "messages": [{"role": "user", "content": parts}]})
    a = (r["choices"][0]["message"].get("content") or "").strip()
    ok = all(e in a.lower() for e in expect)
    print(f"{tag}: {'PASS' if ok else 'CHECK'} | answer={a!r} | prompt_tokens={r['usage']['prompt_tokens']} ({dt:.1f}s)", flush=True)
    return ok


ok = []
ok.append(ask("V1 stripes", [{"type": "text", "text": "This image has three vertical stripes. Name their colors from left to right, comma separated, lowercase."}, stripes], ["red", "green", "blue"]))
ok.append(ask("V2 two images", [{"type": "text", "text": "Here are two images. What color is the first image and what color is the second? Answer as: first=<color>, second=<color>."}, red_sq, blue_sq], ["first=red", "second=blue"]))
ok.append(ask("V3 grid", [{"type": "text", "text": "The image is a 2x2 grid of colored squares. What color is the top-right square? One word."}, grid], ["green"]))

weather = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                          "required": ["city"]}}}
clock = {"type": "function", "function": {"name": "get_time", "description": "Get the current local time in a city",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def calls(r):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"), c.get("id"))
            for c in (r["choices"][0]["message"].get("tool_calls") or [])]


msgs = [{"role": "user", "content": "What's the weather in Paris right now, in celsius?"}]
r, dt = post({"model": M, "max_tokens": 200, "temperature": 0, "messages": msgs, "tools": [weather], "tool_choice": "auto"})
c = calls(r)
t1 = r["choices"][0]["finish_reason"] == "tool_calls" and c and c[0][0] == "get_weather" and c[0][1].get("city", "").lower() == "paris"
ok.append(bool(t1))
print(f"T1 tool call: {'PASS' if t1 else 'CHECK'} | finish={r['choices'][0]['finish_reason']} calls={[(n, a) for n, a, _ in c]} ({dt:.1f}s)", flush=True)

if c:
    msgs += [r["choices"][0]["message"],
             {"role": "tool", "tool_call_id": c[0][2], "content": json.dumps({"city": "Paris", "temp_c": 18, "sky": "light rain"})}]
    r, dt = post({"model": M, "max_tokens": 120, "temperature": 0, "messages": msgs, "tools": [weather]})
    a = (r["choices"][0]["message"].get("content") or "").strip()
    t2 = "18" in a and "rain" in a.lower()
    ok.append(t2)
    print(f"T2 round trip: {'PASS' if t2 else 'CHECK'} | answer={a!r} ({dt:.1f}s)", flush=True)

r, dt = post({"model": M, "max_tokens": 300, "temperature": 0, "tools": [weather, clock], "tool_choice": "auto",
              "messages": [{"role": "user", "content": "Get me the weather in Tokyo and the weather in Berlin, both in celsius."}]})
c = calls(r)
cities = sorted(a.get("city", "").lower() for n, a, _ in c if n == "get_weather")
t3 = cities == ["berlin", "tokyo"]
ok.append(t3)
print(f"T3 parallel calls: {'PASS' if t3 else 'CHECK'} | calls={[(n, a) for n, a, _ in c]} ({dt:.1f}s)", flush=True)

r, dt = post({"model": M, "max_tokens": 200, "temperature": 0, "tools": [weather, clock],
              "tool_choice": {"type": "function", "function": {"name": "get_time"}},
              "messages": [{"role": "user", "content": "I'm flying to Sydney tomorrow."}]})
c = calls(r)
t4 = bool(c) and c[0][0] == "get_time"
ok.append(t4)
print(f"T4 forced tool_choice: {'PASS' if t4 else 'CHECK'} | calls={[(n, a) for n, a, _ in c]} ({dt:.1f}s)", flush=True)

print(f"summary: {sum(ok)}/{len(ok)} PASS", flush=True)
