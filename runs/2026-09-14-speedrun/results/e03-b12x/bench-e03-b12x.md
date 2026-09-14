## e03-b12x (2026-09-14T06:55:43Z)

speed run 2026-09-14 SCREEN e03-b12x

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 54.57 | 59.6 | 0.228 |
| C3 | 118.08 | 45.12 | 0.336 |
| C6 | 188.14 | 37.14 | 0.416 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 82.11 | 66.31 | 57.04 |
| json | 59.04 | 54.52 | 40.21 |
| narrative | 29.94 | 22.87 | 18.06 |
| prose | 37.63 | 27.82 | 19.58 |
| math | 77.82 | 60.29 | 50.34 |
| reasoning | 68.56 | 45.41 | 38.98 |
| summary | 31.2 | 24.67 | 19.34 |
| format | 90.53 | 59.04 | 53.6 |
| ceiling_count | 105.16 | 84.39 | 68.51 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 7.4 | 1566.4 |
| 32000 | 46810 | 27.682 | 1691.0 |
