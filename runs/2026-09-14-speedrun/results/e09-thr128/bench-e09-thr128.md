## e09-thr128 (2026-09-14T07:10:24Z)

speed run 2026-09-14 SCREEN e09-thr128

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 56.29 | 61.02 | 0.213 |
| C3 | 118.05 | 45.56 | 0.298 |
| C6 | 185.41 | 36.32 | 0.402 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 85.46 | 68.24 | 54.78 |
| json | 71.53 | 50.02 | 35.27 |
| narrative | 32.03 | 21.4 | 18.07 |
| prose | 34.3 | 26.01 | 21.13 |
| math | 78.3 | 59.57 | 50.58 |
| reasoning | 72.37 | 46.89 | 34.98 |
| summary | 33.03 | 27.69 | 21.07 |
| format | 81.15 | 64.68 | 54.67 |
| ceiling_count | 107.79 | 82.28 | 68.86 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.111 | 1896.8 |
| 32000 | 46810 | 23.607 | 1982.9 |
