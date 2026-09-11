## exl3tp4b-rep2 (2026-09-11T21:25:33Z)

TP4 EXL3 speed repeat 2 of 2 (C1-C6 only), checking run-to-run variance

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 57.15 | 63.37 | 0.25 |
| C2 | 89.23 | 51.15 | 0.328 |
| C3 | 112.63 | 44.2 | 0.38 |
| C4 | 142.64 | 42.14 | 0.417 |
| C5 | 165.89 | 39.13 | 0.435 |
| C6 | 176.14 | 34.81 | 0.48 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 81.11 | 70.62 | 65.54 | 61.78 | 59.18 | 55.64 |
| json | 76.11 | 62.06 | 49.22 | 52.63 | 48.16 | 39.93 |
| narrative | 33.21 | 26.56 | 21.02 | 20.3 | 17.62 | 14.77 |
| prose | 39.71 | 30.23 | 23.78 | 23.37 | 21.63 | 18.49 |
| math | 85.29 | 67.82 | 59.06 | 55.29 | 51.77 | 46.1 |
| reasoning | 71.4 | 51.76 | 45.02 | 41.17 | 39.31 | 34.51 |
| summary | 35.17 | 28.39 | 26.55 | 23.56 | 19.89 | 20.02 |
| format | 84.94 | 71.78 | 63.43 | 59.05 | 55.49 | 49.0 |
| ceiling_count | 102.4 | 92.27 | 78.97 | 75.37 | 70.33 | 64.93 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
