## e05-k10 (2026-09-14T07:39:59Z)

speed run 2026-09-14 SCREEN e05-k10

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 44.75 | 48.67 | 0.247 |
| C3 | 87.62 | 32.33 | 0.325 |
| C6 | 126.22 | 24.53 | 0.427 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 74.33 | 45.36 | 37.56 |
| json | 52.77 | 34.83 | 20.63 |
| narrative | 23.26 | 15.48 | 10.86 |
| prose | 23.89 | 17.01 | 12.74 |
| math | 66.48 | 50.99 | 35.6 |
| reasoning | 49.85 | 35.18 | 24.12 |
| summary | 27.9 | 17.71 | 13.73 |
| format | 70.86 | 42.05 | 40.99 |
| ceiling_count | 119.93 | 87.14 | 65.26 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.727 | 1723.2 |
| 32000 | 46810 | 26.472 | 1768.3 |
