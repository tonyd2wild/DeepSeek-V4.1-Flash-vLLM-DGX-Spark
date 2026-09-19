## e02-gmu82 (2026-09-19T08:40:48Z)

speed run 2026-09-14 SCREEN e02-gmu82

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 59.82 | 65.54 | 0.211 |
| C3 | 123.59 | 47.43 | 0.3 |
| C6 | 187.96 | 37.26 | 0.384 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 88.51 | 66.93 | 58.23 |
| json | 84.57 | 49.22 | 35.06 |
| narrative | 33.58 | 23.96 | 18.8 |
| prose | 34.66 | 28.58 | 22.37 |
| math | 80.87 | 63.44 | 52.52 |
| reasoning | 70.21 | 51.96 | 34.53 |
| summary | 44.42 | 26.77 | 21.36 |
| format | 87.49 | 68.58 | 55.23 |
| ceiling_count | 113.66 | 88.29 | 74.67 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.077 | 1907.6 |
| 32000 | 46810 | 23.379 | 2002.2 |
