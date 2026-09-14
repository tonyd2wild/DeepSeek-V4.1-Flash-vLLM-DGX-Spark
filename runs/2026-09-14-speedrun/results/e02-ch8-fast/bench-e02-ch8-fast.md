## e02-ch8-fast (2026-09-14T06:42:21Z)

speed run 2026-09-14 SCREEN e02-ch8-fast

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.84 | 64.44 | 0.213 |
| C3 | 119.08 | 45.38 | 0.307 |
| C6 | 189.53 | 37.17 | 0.403 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 83.47 | 63.1 | 57.47 |
| json | 84.02 | 49.5 | 37.06 |
| narrative | 33.15 | 22.77 | 17.46 |
| prose | 37.66 | 28.25 | 20.44 |
| math | 81.83 | 59.67 | 53.6 |
| reasoning | 70.12 | 48.29 | 36.44 |
| summary | 38.96 | 25.78 | 19.82 |
| format | 86.32 | 65.66 | 55.07 |
| ceiling_count | 106.01 | 82.55 | 71.2 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.964 | 1664.7 |
| 32000 | 46810 | 26.733 | 1751.0 |
