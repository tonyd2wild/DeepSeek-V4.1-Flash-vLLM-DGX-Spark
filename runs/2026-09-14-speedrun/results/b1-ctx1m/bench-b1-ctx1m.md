## b1-ctx1m (2026-09-14T09:46:21Z)

speed run 2026-09-14 SCREEN b1-ctx1m

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 61.24 | 66.9 | 0.2 |
| C3 | 120.66 | 46.39 | 0.313 |
| C6 | 179.55 | 36.14 | 0.42 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 93.6 | 67.18 | 58.65 |
| json | 82.8 | 53.55 | 36.09 |
| narrative | 32.57 | 22.85 | 17.14 |
| prose | 42.12 | 26.9 | 21.5 |
| math | 82.73 | 64.4 | 48.51 |
| reasoning | 75.58 | 49.36 | 33.91 |
| summary | 34.33 | 23.79 | 21.18 |
| format | 91.5 | 63.09 | 52.18 |
| ceiling_count | 112.22 | 86.75 | 71.03 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.115 | 1895.7 |
| 32000 | 46810 | 23.605 | 1983.1 |
