## e06-exl3lim (2026-09-19T07:08:32Z)

speed run 2026-09-14 SCREEN e06-exl3lim

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 61.32 | 67.02 | 0.202 |
| C3 | 122.31 | 46.82 | 0.297 |
| C6 | 195.23 | 38.21 | 0.391 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 90.18 | 67.56 | 59.53 |
| json | 71.86 | 51.52 | 38.43 |
| narrative | 32.3 | 23.51 | 18.83 |
| prose | 42.33 | 26.25 | 22.55 |
| math | 88.39 | 62.45 | 53.37 |
| reasoning | 74.26 | 52.01 | 37.07 |
| summary | 37.74 | 25.2 | 20.01 |
| format | 99.09 | 66.07 | 55.88 |
| ceiling_count | 114.71 | 89.07 | 72.95 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.03 | 1922.2 |
| 32000 | 46810 | 23.209 | 2016.9 |
