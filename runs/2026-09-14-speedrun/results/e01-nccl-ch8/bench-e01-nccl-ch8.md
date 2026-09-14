## e01-nccl-ch8 (2026-09-14T06:28:55Z)

speed run 2026-09-14 SCREEN e01-nccl-ch8

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 57.42 | 62.96 | 0.234 |
| C3 | 117.32 | 44.58 | 0.311 |
| C6 | 184.21 | 36.17 | 0.435 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 85.4 | 66.69 | 53.47 |
| json | 72.52 | 54.89 | 34.92 |
| narrative | 31.74 | 21.94 | 15.66 |
| prose | 36.5 | 28.15 | 20.65 |
| math | 82.91 | 57.85 | 50.76 |
| reasoning | 71.53 | 40.54 | 37.32 |
| summary | 34.37 | 25.12 | 20.01 |
| format | 88.69 | 61.48 | 56.56 |
| ceiling_count | 102.43 | 79.31 | 68.59 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 9.067 | 1278.5 |
| 32000 | 46810 | 31.51 | 1485.5 |
