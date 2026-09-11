## exl3tp3a10 (2026-09-11T18:57:31Z)

TP3 EXL3 try10: try 9 plus gmu 0.80 (patch set tp3e), graphs, vision, no spec, head Bluey

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 24.23 | 25.41 | 0.275 |
| C2 | 39.26 | 21.74 | 0.59 |
| C3 | 57.52 | 20.93 | 0.397 |
| C4 | 71.71 | 19.53 | 0.495 |
| C5 | 87.58 | 19.32 | 0.517 |
| C6 | 106.13 | 19.34 | 0.497 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 26.95 | 21.42 | 22.98 | 18.24 | 17.86 | 19.04 |
| json | 27.07 | 18.44 | 21.11 | 19.79 | 19.37 | 19.6 |
| narrative | 24.69 | 16.92 | 20.48 | 19.32 | 17.87 | 17.19 |
| prose | 26.71 | 22.72 | 20.94 | 20.62 | 19.36 | 18.53 |
| math | 20.37 | 24.73 | 18.28 | 18.9 | 17.78 | 21.41 |
| reasoning | 26.51 | 23.25 | 20.4 | 19.67 | 18.59 | 18.92 |
| summary | 26.33 | 25.0 | 19.31 | 18.79 | 19.36 | 18.16 |
| format | 24.67 | 21.43 | 23.96 | 20.91 | 24.34 | 21.9 |
| ceiling_count | 27.3 | 19.4 | 22.67 | 23.72 | 23.04 | 22.99 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 4.142 | 712.2 |
| 8000 | 11592 | 9.815 | 1181.0 |
| 32000 | 46810 | 40.82 | 1146.7 |
| 64000 | 93335 | 80.837 | 1154.6 |
