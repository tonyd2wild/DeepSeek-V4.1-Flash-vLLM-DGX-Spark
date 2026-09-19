## s2-00-baseline (2026-09-19T05:54:36Z)

speed run 2026-09-14 FINAL s2-00-baseline

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.27 | 64.03 | 0.237 |
| C2 | 90.58 | 50.71 | 0.289 |
| C3 | 110.78 | 42.55 | 0.298 |
| C4 | 146.03 | 42.5 | 0.389 |
| C5 | 164.26 | 38.35 | 0.415 |
| C6 | 184.78 | 36.25 | 0.429 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 90.88 | 78.04 | 65.0 | 61.88 | 59.51 | 56.51 |
| json | 68.03 | 56.05 | 49.27 | 48.85 | 42.53 | 36.73 |
| narrative | 31.39 | 26.67 | 24.61 | 22.53 | 20.31 | 17.5 |
| prose | 39.48 | 33.4 | 27.89 | 21.88 | 22.81 | 17.03 |
| math | 85.04 | 67.47 | 55.99 | 59.87 | 53.78 | 49.84 |
| reasoning | 66.21 | 51.14 | 46.26 | 49.02 | 31.19 | 33.01 |
| summary | 32.57 | 30.64 | 29.03 | 16.56 | 22.7 | 19.73 |
| format | 98.65 | 62.28 | 42.32 | 59.44 | 53.95 | 59.62 |
| ceiling_count | 113.48 | 76.75 | 79.54 | 82.83 | 76.22 | 69.7 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 2.141 | 1377.9 |
| 8000 | 11592 | 8.768 | 1322.1 |
| 32000 | 46810 | 32.991 | 1418.9 |
| 64000 | 93335 | 66.622 | 1401.0 |
