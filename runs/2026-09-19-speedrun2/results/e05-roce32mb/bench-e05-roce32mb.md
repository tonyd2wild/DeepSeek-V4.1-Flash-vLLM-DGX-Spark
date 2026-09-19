## e05-roce32mb (2026-09-19T06:42:08Z)

speed run 2026-09-14 SCREEN e05-roce32mb

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 60.04 | 65.41 | 0.202 |
| C3 | 124.75 | 47.75 | 0.327 |
| C6 | 191.68 | 37.53 | 0.457 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 90.6 | 73.59 | 57.47 |
| json | 71.7 | 47.9 | 39.51 |
| narrative | 35.47 | 23.12 | 17.25 |
| prose | 40.74 | 26.69 | 21.92 |
| math | 84.94 | 63.23 | 52.68 |
| reasoning | 73.48 | 46.41 | 36.03 |
| summary | 37.01 | 26.93 | 20.17 |
| format | 89.32 | 74.13 | 55.19 |
| ceiling_count | 112.77 | 87.18 | 70.16 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.801 | 1704.4 |
| 32000 | 46810 | 23.731 | 1972.5 |
