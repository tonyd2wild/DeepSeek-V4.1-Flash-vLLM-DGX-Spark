## e07-gather4 (2026-09-19T07:37:12Z)

speed run 2026-09-14 SCREEN e07-gather4

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 60.98 | 67.23 | 0.217 |
| C3 | 120.09 | 46.29 | 0.316 |
| C6 | 189.57 | 36.8 | 0.404 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 90.59 | 65.05 | 58.29 |
| json | 80.34 | 43.05 | 35.39 |
| narrative | 32.6 | 22.06 | 19.3 |
| prose | 42.31 | 28.91 | 21.04 |
| math | 82.89 | 62.13 | 49.41 |
| reasoning | 73.31 | 46.18 | 33.72 |
| summary | 37.82 | 24.88 | 22.18 |
| format | 97.95 | 78.05 | 55.11 |
| ceiling_count | 113.67 | 87.77 | 75.35 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 7.086 | 1635.8 |
| 32000 | 46810 | 27.541 | 1699.7 |
