## f-noexpseg (2026-09-14T10:38:59Z)

speed run 2026-09-14 SCREEN f-noexpseg

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 59.95 | 65.48 | 0.207 |
| C3 | 128.07 | 49.03 | 0.315 |
| C6 | 191.54 | 37.8 | 0.406 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 91.01 | 73.37 | 61.07 |
| json | 78.15 | 47.91 | 35.38 |
| narrative | 31.79 | 23.3 | 19.36 |
| prose | 39.16 | 27.87 | 20.83 |
| math | 83.6 | 63.91 | 52.18 |
| reasoning | 73.04 | 52.22 | 33.45 |
| summary | 37.54 | 26.45 | 21.11 |
| format | 89.55 | 77.22 | 59.03 |
| ceiling_count | 113.53 | 87.78 | 70.39 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.063 | 1912.1 |
| 32000 | 46810 | 23.284 | 2010.4 |
