## f-gmu82 (2026-09-14T10:25:50Z)

speed run 2026-09-14 SCREEN f-gmu82

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 60.02 | 65.79 | 0.214 |
| C3 | 122.44 | 47.05 | 0.302 |
| C6 | 191.41 | 37.56 | 0.402 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 93.4 | 66.87 | 55.91 |
| json | 62.52 | 55.61 | 39.07 |
| narrative | 33.83 | 24.26 | 18.44 |
| prose | 38.56 | 28.33 | 22.18 |
| math | 84.45 | 59.55 | 51.56 |
| reasoning | 76.46 | 48.46 | 35.83 |
| summary | 38.82 | 27.49 | 21.73 |
| format | 98.29 | 65.84 | 55.74 |
| ceiling_count | 114.55 | 87.08 | 70.55 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.1 | 1900.2 |
| 32000 | 46810 | 23.491 | 1992.7 |
