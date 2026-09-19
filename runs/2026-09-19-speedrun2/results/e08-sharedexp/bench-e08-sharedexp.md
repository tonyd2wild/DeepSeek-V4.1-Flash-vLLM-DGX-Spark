## e08-sharedexp (2026-09-19T07:49:53Z)

speed run 2026-09-14 SCREEN e08-sharedexp

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 61.76 | 67.72 | 0.207 |
| C3 | 124.05 | 47.04 | 0.298 |
| C6 | 201.26 | 39.36 | 0.382 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 91.61 | 67.35 | 57.76 |
| json | 89.08 | 53.93 | 39.72 |
| narrative | 33.26 | 23.37 | 17.92 |
| prose | 38.24 | 29.42 | 22.25 |
| math | 84.61 | 61.34 | 58.98 |
| reasoning | 68.96 | 46.39 | 39.11 |
| summary | 38.63 | 26.16 | 21.94 |
| format | 97.39 | 68.36 | 57.22 |
| ceiling_count | 114.23 | 89.17 | 74.72 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.154 | 1883.7 |
| 32000 | 46810 | 23.6 | 1983.5 |
