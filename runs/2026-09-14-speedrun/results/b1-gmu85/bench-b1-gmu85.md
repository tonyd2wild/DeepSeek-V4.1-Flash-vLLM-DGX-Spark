## b1-gmu85 (2026-09-14T09:33:07Z)

speed run 2026-09-14 SCREEN b1-gmu85

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 59.57 | 65.09 | 0.213 |
| C3 | 124.45 | 47.41 | 0.3 |
| C6 | 194.86 | 38.1 | 0.397 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 89.31 | 67.34 | 56.93 |
| json | 72.77 | 55.83 | 40.65 |
| narrative | 34.42 | 22.38 | 17.28 |
| prose | 39.5 | 27.27 | 23.04 |
| math | 83.75 | 62.97 | 47.03 |
| reasoning | 76.41 | 49.1 | 37.93 |
| summary | 37.68 | 24.52 | 22.46 |
| format | 86.92 | 69.88 | 59.46 |
| ceiling_count | 114.43 | 87.23 | 71.08 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 8.417 | 1377.3 |
| 32000 | 46810 | 24.906 | 1879.5 |
