## b1-k4 (2026-09-14T08:52:44Z)

speed run 2026-09-14 SCREEN b1-k4

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 59.08 | 64.29 | 0.203 |
| C3 | 130.22 | 49.69 | 0.303 |
| C6 | 197.53 | 38.88 | 0.383 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 86.96 | 71.84 | 56.61 |
| json | 78.44 | 51.74 | 43.5 |
| narrative | 33.78 | 26.36 | 20.46 |
| prose | 37.96 | 33.18 | 23.43 |
| math | 80.31 | 57.83 | 49.49 |
| reasoning | 72.47 | 61.88 | 36.61 |
| summary | 38.73 | 28.14 | 24.19 |
| format | 85.66 | 66.52 | 56.75 |
| ceiling_count | 102.37 | 79.48 | 65.62 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.039 | 1919.7 |
| 32000 | 46810 | 23.703 | 1974.8 |
