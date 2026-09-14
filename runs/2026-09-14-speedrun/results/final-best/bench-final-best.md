## final-best (2026-09-14T10:53:34Z)

speed run 2026-09-14 FINAL final-best

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.27 | 63.74 | 0.221 |
| C2 | 96.0 | 54.65 | 0.268 |
| C3 | 128.29 | 48.25 | 0.3 |
| C4 | 152.44 | 44.42 | 0.33 |
| C5 | 171.11 | 40.63 | 0.375 |
| C6 | 189.97 | 36.95 | 0.399 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 85.2 | 75.04 | 69.5 | 64.35 | 61.21 | 56.6 |
| json | 65.01 | 56.63 | 51.46 | 45.69 | 43.87 | 38.56 |
| narrative | 36.29 | 28.22 | 22.29 | 21.69 | 20.29 | 18.08 |
| prose | 41.0 | 31.03 | 28.81 | 24.33 | 24.34 | 21.95 |
| math | 87.86 | 76.64 | 66.22 | 58.9 | 56.02 | 51.97 |
| reasoning | 72.77 | 63.6 | 54.19 | 48.41 | 40.08 | 33.82 |
| summary | 35.99 | 31.45 | 27.61 | 26.73 | 22.73 | 21.52 |
| format | 85.81 | 74.55 | 65.96 | 65.25 | 56.52 | 53.14 |
| ceiling_count | 112.87 | 99.82 | 87.7 | 80.1 | 75.27 | 73.12 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.607 | 1836.1 |
| 8000 | 11592 | 5.804 | 1997.3 |
| 32000 | 46810 | 23.238 | 2014.4 |
| 64000 | 93335 | 45.923 | 2032.4 |
