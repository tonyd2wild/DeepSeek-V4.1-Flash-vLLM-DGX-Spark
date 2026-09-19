## e11-final500k (2026-09-19T08:28:44Z)

speed run 2026-09-14 SCREEN e11-final500k

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 60.84 | 66.59 | 0.209 |
| C3 | 120.65 | 46.89 | 0.298 |
| C6 | 197.48 | 38.57 | 0.393 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 93.6 | 71.76 | 59.25 |
| json | 67.0 | 50.29 | 37.44 |
| narrative | 35.13 | 23.75 | 19.24 |
| prose | 37.63 | 28.63 | 21.64 |
| math | 85.53 | 64.19 | 48.85 |
| reasoning | 75.84 | 44.46 | 39.27 |
| summary | 44.2 | 29.37 | 22.56 |
| format | 93.77 | 62.7 | 60.28 |
| ceiling_count | 114.39 | 88.16 | 75.82 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.027 | 1923.2 |
| 32000 | 46810 | 23.214 | 2016.5 |
