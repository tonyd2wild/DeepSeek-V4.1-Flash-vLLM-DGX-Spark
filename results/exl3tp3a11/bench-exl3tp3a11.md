## exl3tp3a11 (2026-09-11T19:22:28Z)

TP3 EXL3 try11: try 10 plus DSpark k=5 (patch set tp3e), graphs, vision, gmu 0.80, head Bluey

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 46.0 | 51.46 | 0.305 |
| C2 | 73.37 | 43.58 | 0.634 |
| C3 | 100.14 | 38.52 | 0.412 |
| C4 | 118.41 | 35.16 | 0.441 |
| C5 | 134.44 | 31.67 | 0.46 |
| C6 | 152.89 | 29.62 | 0.5 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 70.05 | 62.57 | 52.18 | 50.66 | 46.41 | 40.86 |
| json | 42.03 | 41.34 | 31.2 | 35.02 | 28.58 | 28.29 |
| narrative | 28.08 | 21.98 | 18.57 | 16.46 | 14.26 | 12.97 |
| prose | 29.55 | 26.21 | 22.47 | 20.07 | 17.35 | 17.33 |
| math | 76.46 | 59.54 | 51.72 | 46.6 | 44.05 | 38.31 |
| reasoning | 50.67 | 41.42 | 37.39 | 32.62 | 30.58 | 28.56 |
| summary | 36.16 | 30.21 | 26.84 | 20.99 | 18.6 | 19.89 |
| format | 78.65 | 65.38 | 67.83 | 58.87 | 53.55 | 50.78 |
| ceiling_count | 83.08 | 77.73 | 71.61 | 61.92 | 59.57 | 55.69 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 2.635 | 1119.4 |
| 8000 | 11592 | 9.774 | 1186.0 |
| 32000 | 46810 | 39.41 | 1187.8 |
| 64000 | 93335 | 77.865 | 1198.7 |
