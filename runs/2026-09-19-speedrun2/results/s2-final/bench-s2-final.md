## s2-final (2026-09-19T08:58:50Z)

speed run 2026-09-14 FINAL s2-final

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 61.26 | 67.4 | 0.205 |
| C2 | 102.29 | 57.59 | 0.267 |
| C3 | 128.87 | 49.12 | 0.289 |
| C4 | 153.35 | 44.25 | 0.332 |
| C5 | 174.34 | 41.28 | 0.362 |
| C6 | 189.29 | 37.75 | 0.381 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 90.74 | 82.66 | 74.06 | 64.07 | 65.61 | 59.52 |
| json | 75.79 | 63.44 | 56.73 | 48.67 | 39.75 | 36.36 |
| narrative | 31.4 | 28.32 | 24.19 | 22.01 | 20.47 | 18.48 |
| prose | 39.04 | 31.66 | 28.13 | 25.56 | 23.75 | 22.03 |
| math | 87.48 | 78.88 | 61.79 | 59.85 | 60.86 | 52.69 |
| reasoning | 75.14 | 63.28 | 51.22 | 48.01 | 41.31 | 37.39 |
| summary | 41.09 | 33.23 | 27.22 | 25.45 | 22.83 | 20.09 |
| format | 98.55 | 79.21 | 69.61 | 60.4 | 55.64 | 55.47 |
| ceiling_count | 113.0 | 100.72 | 88.42 | 84.93 | 79.53 | 74.31 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.521 | 1939.3 |
| 8000 | 11592 | 0.183 | 63302.0 |
| 32000 | 46810 | 0.269 | 173710.3 |
| 64000 | 93335 | 45.574 | 2048.0 |
