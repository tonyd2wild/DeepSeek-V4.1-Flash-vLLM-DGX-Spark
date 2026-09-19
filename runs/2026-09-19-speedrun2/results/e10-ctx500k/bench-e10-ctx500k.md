## e10-ctx500k (2026-09-19T06:55:20Z)

speed run 2026-09-14 SCREEN e10-ctx500k

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 56.09 | 60.83 | 0.207 |
| C3 | 124.05 | 47.36 | 0.306 |
| C6 | 190.13 | 37.12 | 0.403 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 85.93 | 71.03 | 59.72 |
| json | 65.48 | 50.23 | 35.45 |
| narrative | 31.48 | 22.71 | 18.23 |
| prose | 38.77 | 26.49 | 22.12 |
| math | 84.96 | 61.21 | 50.22 |
| reasoning | 57.13 | 47.1 | 33.25 |
| summary | 35.1 | 24.82 | 19.83 |
| format | 87.77 | 75.3 | 58.13 |
| ceiling_count | 113.38 | 83.11 | 68.58 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.203 | 1868.9 |
| 32000 | 46810 | 23.85 | 1962.7 |
