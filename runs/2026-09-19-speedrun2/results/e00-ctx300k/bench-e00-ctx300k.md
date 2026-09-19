## e00-ctx300k (2026-09-19T06:13:22Z)

speed run 2026-09-14 SCREEN e00-ctx300k

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 59.54 | 65.11 | 0.212 |
| C3 | 121.75 | 46.29 | 0.304 |
| C6 | 187.65 | 36.53 | 0.412 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 88.9 | 66.42 | 57.69 |
| json | 68.06 | 50.75 | 36.71 |
| narrative | 33.95 | 23.03 | 17.58 |
| prose | 40.94 | 26.83 | 21.45 |
| math | 81.26 | 66.04 | 48.24 |
| reasoning | 70.81 | 45.68 | 35.9 |
| summary | 38.28 | 26.27 | 19.86 |
| format | 98.68 | 65.27 | 54.8 |
| ceiling_count | 113.82 | 87.77 | 72.66 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 6.208 | 1867.3 |
| 32000 | 46810 | 23.894 | 1959.0 |
