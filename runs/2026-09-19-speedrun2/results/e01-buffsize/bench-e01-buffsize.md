## e01-buffsize (2026-09-19T08:01:58Z)

speed run 2026-09-14 SCREEN e01-buffsize

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 59.52 | 65.05 | 0.203 |
| C3 | 127.41 | 47.71 | 0.295 |
| C6 | 198.37 | 38.88 | 0.381 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 95.76 | 68.85 | 58.25 |
| json | 78.37 | 53.56 | 41.2 |
| narrative | 31.1 | 23.55 | 18.86 |
| prose | 39.88 | 26.49 | 22.94 |
| math | 84.36 | 59.86 | 53.06 |
| reasoning | 60.2 | 51.8 | 32.29 |
| summary | 43.69 | 28.84 | 22.46 |
| format | 87.03 | 68.72 | 61.98 |
| ceiling_count | 114.03 | 89.23 | 75.43 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.038 | 1919.8 |
| 32000 | 46810 | 22.993 | 2035.9 |
