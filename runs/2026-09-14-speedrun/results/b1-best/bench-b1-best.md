## b1-best (2026-09-14T08:22:25Z)

speed run 2026-09-14 SCREEN b1-best

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.89 | 64.56 | 0.21 |
| C3 | 124.78 | 47.77 | 0.307 |
| C6 | 197.91 | 38.69 | 0.408 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 91.85 | 73.41 | 58.42 |
| json | 62.05 | 54.0 | 37.99 |
| narrative | 32.62 | 23.39 | 17.58 |
| prose | 42.52 | 28.55 | 21.34 |
| math | 80.16 | 59.55 | 59.14 |
| reasoning | 74.24 | 48.43 | 36.92 |
| summary | 38.95 | 29.88 | 21.66 |
| format | 94.06 | 64.99 | 56.48 |
| ceiling_count | 114.18 | 88.16 | 70.38 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.045 | 1917.7 |
| 32000 | 46810 | 23.677 | 1977.0 |
