## 00b-baseline-screen (2026-09-14T06:15:44Z)

speed run 2026-09-14 SCREEN 00b-baseline-screen

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 54.27 | 59.62 | 0.257 |
| C3 | 114.77 | 44.08 | 0.374 |
| C6 | 166.34 | 32.63 | 0.437 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 80.63 | 58.98 | 49.95 |
| json | 63.64 | 49.89 | 35.32 |
| narrative | 31.22 | 22.4 | 14.94 |
| prose | 37.29 | 26.17 | 18.81 |
| math | 78.2 | 61.24 | 43.86 |
| reasoning | 71.13 | 47.94 | 30.8 |
| summary | 31.86 | 23.4 | 18.69 |
| format | 82.96 | 62.58 | 48.7 |
| ceiling_count | 103.88 | 79.16 | 62.26 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 0.281 | 41227.6 |
| 32000 | 46810 | 0.345 | 135723.7 |
