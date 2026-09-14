## b1-idxsplit (2026-09-14T08:37:22Z)

speed run 2026-09-14 SCREEN b1-idxsplit

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 60.03 | 65.61 | 0.212 |
| C3 | 126.53 | 47.64 | 0.301 |
| C6 | 190.41 | 37.27 | 0.403 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 92.92 | 70.01 | 57.45 |
| json | 64.27 | 53.41 | 39.84 |
| narrative | 33.99 | 22.61 | 18.57 |
| prose | 40.37 | 27.41 | 21.57 |
| math | 84.52 | 62.81 | 48.44 |
| reasoning | 80.88 | 54.16 | 34.31 |
| summary | 37.09 | 28.43 | 21.57 |
| format | 90.81 | 62.27 | 56.41 |
| ceiling_count | 113.28 | 87.02 | 73.58 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.117 | 1895.1 |
| 32000 | 46810 | 23.493 | 1992.5 |
