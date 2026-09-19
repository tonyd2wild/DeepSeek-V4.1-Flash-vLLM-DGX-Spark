## e03-seqs16 (2026-09-19T08:14:31Z)

speed run 2026-09-14 SCREEN e03-seqs16

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.82 | 63.71 | 0.201 |
| C3 | 130.4 | 48.91 | 0.297 |
| C6 | 190.92 | 38.09 | 0.39 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 91.04 | 67.7 | 56.21 |
| json | 70.58 | 55.31 | 39.08 |
| narrative | 31.94 | 24.43 | 18.2 |
| prose | 42.48 | 29.04 | 22.59 |
| math | 81.69 | 70.56 | 49.88 |
| reasoning | 65.89 | 52.23 | 41.3 |
| summary | 37.09 | 26.89 | 20.14 |
| format | 88.99 | 65.09 | 57.3 |
| ceiling_count | 112.97 | 88.22 | 74.47 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 7.052 | 1643.8 |
| 32000 | 46810 | 23.227 | 2015.3 |
