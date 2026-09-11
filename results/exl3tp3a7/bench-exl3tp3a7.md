## exl3tp3a7 (2026-09-11T18:06:02Z)

TP3 EXL3 try7: try 6 + CUDA graphs (FULL_AND_PIECEWISE, sizes 1..8), no spec, text-only, head Bluey

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 25.37 | 26.5 | 0.253 |
| C2 | 44.95 | 23.93 | 0.317 |
| C3 | 61.02 | 21.88 | 0.382 |
| C4 | 73.51 | 20.2 | 0.67 |
| C5 | 90.16 | 19.64 | 0.47 |
| C6 | 108.75 | 19.75 | 0.489 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 26.7 | 25.23 | 24.77 | 21.7 | 21.97 | 21.42 |
| json | 27.05 | 24.46 | 22.29 | 21.08 | 21.16 | 19.84 |
| narrative | 26.87 | 23.25 | 20.43 | 19.11 | 18.07 | 17.13 |
| prose | 26.75 | 23.71 | 20.96 | 20.56 | 19.37 | 19.85 |
| math | 25.94 | 23.52 | 21.26 | 20.75 | 18.63 | 20.85 |
| reasoning | 26.2 | 23.02 | 21.5 | 19.42 | 18.7 | 18.97 |
| summary | 26.44 | 22.07 | 20.46 | 17.63 | 18.72 | 18.14 |
| format | 26.02 | 26.21 | 23.34 | 21.37 | 20.54 | 21.76 |
| ceiling_count | 27.06 | 25.92 | 24.89 | 23.3 | 23.66 | 23.02 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 3.486 | 846.3 |
| 8000 | 11592 | 9.873 | 1174.1 |
| 32000 | 46810 | 38.134 | 1227.5 |
| 64000 | 93335 | 77.422 | 1205.5 |
