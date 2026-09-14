## e14-idxsplit (2026-09-14T08:07:20Z)

speed run 2026-09-14 SCREEN e14-idxsplit

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 57.5 | 62.7 | 0.213 |
| C3 | 122.48 | 46.3 | 0.306 |
| C6 | 191.81 | 37.22 | 0.396 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 85.96 | 66.95 | 55.41 |
| json | 74.35 | 53.19 | 44.54 |
| narrative | 33.75 | 22.28 | 17.76 |
| prose | 37.37 | 27.64 | 21.64 |
| math | 79.26 | 58.14 | 50.72 |
| reasoning | 68.98 | 50.11 | 34.47 |
| summary | 34.96 | 26.1 | 18.75 |
| format | 86.98 | 66.01 | 54.45 |
| ceiling_count | 107.74 | 83.06 | 69.51 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 7.013 | 1653.0 |
| 32000 | 46810 | 26.522 | 1765.0 |
