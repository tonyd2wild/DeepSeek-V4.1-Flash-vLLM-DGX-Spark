## exl3tp4b-rep1 (2026-09-11T21:21:33Z)

TP4 EXL3 speed repeat 1 of 2 (C1-C6 only), checking run-to-run variance

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 54.11 | 60.1 | 0.259 |
| C2 | 90.84 | 52.2 | 0.33 |
| C3 | 115.47 | 44.44 | 0.355 |
| C4 | 137.81 | 40.42 | 0.427 |
| C5 | 167.57 | 38.77 | 0.436 |
| C6 | 170.92 | 33.11 | 0.466 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 81.2 | 78.09 | 64.47 | 62.81 | 59.15 | 52.35 |
| json | 57.5 | 60.25 | 50.94 | 47.42 | 40.17 | 37.7 |
| narrative | 30.41 | 27.14 | 22.12 | 19.92 | 17.13 | 15.39 |
| prose | 39.42 | 28.34 | 24.6 | 21.97 | 20.85 | 18.29 |
| math | 82.95 | 69.32 | 61.17 | 50.1 | 50.16 | 43.52 |
| reasoning | 68.66 | 55.35 | 46.92 | 43.8 | 43.96 | 32.18 |
| summary | 38.16 | 30.4 | 26.81 | 23.04 | 23.19 | 19.43 |
| format | 82.48 | 68.74 | 58.47 | 54.32 | 55.53 | 46.0 |
| ceiling_count | 97.39 | 89.9 | 79.48 | 75.09 | 69.66 | 62.69 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
