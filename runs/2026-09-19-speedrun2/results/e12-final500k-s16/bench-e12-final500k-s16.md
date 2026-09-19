## e12-final500k-s16 (2026-09-19T08:53:41Z)

speed run 2026-09-14 SCREEN e12-final500k-s16

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 59.92 | 65.06 | 0.202 |
| C3 | 125.81 | 47.86 | 0.295 |
| C6 | 190.2 | 37.76 | 0.393 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 90.7 | 72.11 | 58.67 |
| json | 68.52 | 53.94 | 36.14 |
| narrative | 33.18 | 24.97 | 18.32 |
| prose | 36.69 | 28.29 | 22.65 |
| math | 82.85 | 65.09 | 53.12 |
| reasoning | 74.07 | 46.51 | 36.36 |
| summary | 39.48 | 28.65 | 20.04 |
| format | 95.01 | 63.3 | 56.74 |
| ceiling_count | 113.0 | 88.95 | 74.34 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.173 | 1877.9 |
| 32000 | 46810 | 23.32 | 2007.3 |
