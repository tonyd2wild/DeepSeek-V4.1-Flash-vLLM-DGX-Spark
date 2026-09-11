## exl3tp3a9 (2026-09-11T18:35:35Z)

TP3 EXL3 try9: try 7 plus vision (encoder replicated per rank), CUDA graphs, no spec, head Bluey

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 25.69 | 26.79 | 0.25 |
| C2 | 44.2 | 23.59 | 0.332 |
| C3 | 60.45 | 22.12 | 0.586 |
| C4 | 78.09 | 21.1 | 0.424 |
| C5 | 89.36 | 19.47 | 0.485 |
| C6 | 107.47 | 19.62 | 0.489 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 27.71 | 23.76 | 22.96 | 21.93 | 21.71 | 20.33 |
| json | 27.43 | 22.97 | 21.44 | 21.14 | 18.08 | 21.74 |
| narrative | 26.96 | 24.27 | 20.54 | 19.93 | 17.6 | 17.23 |
| prose | 26.88 | 23.36 | 24.65 | 20.91 | 18.84 | 18.5 |
| math | 26.18 | 25.01 | 23.64 | 21.85 | 20.71 | 19.98 |
| reasoning | 26.45 | 23.11 | 22.1 | 21.29 | 18.74 | 19.03 |
| summary | 26.61 | 22.19 | 18.52 | 18.76 | 17.23 | 18.16 |
| format | 26.1 | 24.02 | 23.1 | 22.99 | 22.85 | 21.96 |
| ceiling_count | 27.2 | 26.05 | 25.15 | 24.48 | 22.56 | 23.19 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 4.032 | 731.6 |
| 8000 | 11592 | 10.03 | 1155.7 |
| 32000 | 46810 | 38.871 | 1204.2 |
| 64000 | 93335 | 75.53 | 1235.7 |
