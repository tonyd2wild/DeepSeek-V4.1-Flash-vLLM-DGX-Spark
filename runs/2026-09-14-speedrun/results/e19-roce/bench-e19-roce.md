## e19-roce (2026-09-14T07:54:06Z)

speed run 2026-09-14 SCREEN e19-roce

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 60.13 | 65.81 | 0.213 |
| C3 | 125.31 | 47.58 | 0.306 |
| C6 | 191.69 | 37.58 | 0.41 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 92.53 | 73.89 | 57.61 |
| json | 72.34 | 55.49 | 39.46 |
| narrative | 33.99 | 22.11 | 18.01 |
| prose | 43.93 | 27.85 | 21.6 |
| math | 81.08 | 58.23 | 48.93 |
| reasoning | 71.53 | 46.39 | 35.43 |
| summary | 38.6 | 28.55 | 22.51 |
| format | 92.48 | 68.16 | 57.08 |
| ceiling_count | 115.19 | 88.68 | 70.73 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.893 | 1681.7 |
| 32000 | 46810 | 26.112 | 1792.7 |
