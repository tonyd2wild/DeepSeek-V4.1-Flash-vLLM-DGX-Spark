## speedrun-00-baseline (2026-09-14T06:08:21Z)

speed run 2026-09-14 baseline: exl3tp4b ablit (EXL3-Pollard-Abliterated, TP4, DSpark k=5, CUDA graphs FULL_AND_PIECEWISE, max_num_batched_tokens 8192, gmu 0.80, 300K, vision+tools, patch set dsv41-exl3-tp3e), as found; KV 3,274,912

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 56.27 | 63.19 | 0.276 |
| C2 | 86.89 | 49.95 | 0.31 |
| C3 | 114.7 | 43.6 | 0.352 |
| C4 | 136.3 | 39.7 | 0.419 |
| C5 | 156.62 | 36.72 | 0.449 |
| C6 | 167.68 | 32.76 | 0.456 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 83.99 | 69.05 | 60.7 | 56.17 | 53.8 | 51.21 |
| json | 75.42 | 52.68 | 61.54 | 42.24 | 41.74 | 32.18 |
| narrative | 34.93 | 26.9 | 22.66 | 19.77 | 17.48 | 14.99 |
| prose | 36.98 | 29.41 | 25.9 | 23.18 | 21.94 | 19.78 |
| math | 78.45 | 70.09 | 53.93 | 55.17 | 52.09 | 44.94 |
| reasoning | 70.17 | 52.49 | 38.74 | 39.29 | 33.76 | 31.07 |
| summary | 34.46 | 29.15 | 25.51 | 23.48 | 20.64 | 18.54 |
| format | 91.14 | 69.82 | 59.82 | 58.33 | 52.28 | 49.37 |
| ceiling_count | 98.31 | 91.71 | 77.96 | 72.65 | 70.81 | 62.33 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 2.154 | 1369.4 |
| 8000 | 11592 | 8.033 | 1443.1 |
| 32000 | 46810 | 32.186 | 1454.3 |
| 64000 | 93335 | 63.577 | 1468.1 |
