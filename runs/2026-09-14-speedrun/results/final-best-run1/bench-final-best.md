## final-best (2026-09-14T10:06:39Z)

speed run 2026-09-14 FINAL final-best

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 60.75 | 66.63 | 0.208 |
| C2 | 99.32 | 56.22 | 0.268 |
| C3 | 125.17 | 47.84 | 0.312 |
| C4 | 153.45 | 44.07 | 0.341 |
| C5 | 180.14 | 42.03 | 0.381 |
| C6 | 192.02 | 37.48 | 0.394 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 91.87 | 75.93 | 70.39 | 65.22 | 61.46 | 57.03 |
| json | 76.97 | 66.54 | 49.74 | 46.32 | 44.45 | 38.2 |
| narrative | 29.11 | 27.5 | 24.48 | 21.35 | 20.75 | 18.98 |
| prose | 38.03 | 34.9 | 28.23 | 26.37 | 24.26 | 21.66 |
| math | 84.7 | 74.59 | 64.91 | 62.13 | 57.5 | 51.25 |
| reasoning | 76.63 | 59.12 | 51.45 | 46.02 | 43.36 | 38.79 |
| summary | 39.05 | 32.14 | 25.38 | 23.44 | 22.49 | 20.4 |
| format | 96.67 | 79.01 | 68.1 | 61.7 | 61.96 | 53.55 |
| ceiling_count | 113.9 | 100.55 | 88.55 | 83.76 | 78.53 | 73.54 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.6 | 1844.2 |
| 8000 | 11592 | 5.823 | 1990.7 |
| 32000 | 46810 | 23.304 | 2008.7 |
| 64000 | 93335 | 46.181 | 2021.1 |
