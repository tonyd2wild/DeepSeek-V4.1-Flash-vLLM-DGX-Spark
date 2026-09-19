## e09-sr4 (2026-09-19T07:24:08Z)

speed run 2026-09-14 SCREEN e09-sr4

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 62.02 | 67.81 | 0.203 |
| C3 | 121.5 | 47.0 | 0.306 |
| C6 | 195.05 | 39.18 | 0.379 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 92.25 | 72.5 | 61.3 |
| json | 75.68 | 48.1 | 40.55 |
| narrative | 33.78 | 23.65 | 19.88 |
| prose | 38.55 | 28.36 | 22.12 |
| math | 88.16 | 65.23 | 52.64 |
| reasoning | 76.87 | 43.92 | 38.64 |
| summary | 38.88 | 26.81 | 21.83 |
| format | 98.34 | 67.45 | 56.49 |
| ceiling_count | 114.8 | 90.21 | 75.79 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.048 | 1916.6 |
| 32000 | 46810 | 23.158 | 2021.3 |
