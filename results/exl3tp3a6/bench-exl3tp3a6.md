## exl3tp3a6 (2026-09-11T17:40:45Z)

TP3 EXL3 try6: vl_model streaming load fix, eager, no spec, text-only, head Bluey

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 16.92 | 17.38 | 0.262 |
| C2 | 32.65 | 17.09 | 0.337 |
| C3 | 46.7 | 16.64 | 0.399 |
| C4 | 59.13 | 16.19 | 0.682 |
| C5 | 75.66 | 16.37 | 0.48 |
| C6 | 90.55 | 16.35 | 0.511 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 17.86 | 17.71 | 17.48 | 16.26 | 16.77 | 16.94 |
| json | 17.37 | 16.75 | 16.3 | 16.42 | 17.18 | 16.51 |
| narrative | 17.44 | 16.82 | 16.16 | 16.03 | 15.88 | 15.57 |
| prose | 17.1 | 16.43 | 16.26 | 16.33 | 16.08 | 15.7 |
| math | 16.98 | 17.55 | 16.32 | 16.26 | 16.43 | 16.58 |
| reasoning | 17.3 | 17.05 | 16.57 | 16.35 | 16.12 | 16.03 |
| summary | 17.6 | 16.87 | 16.78 | 15.12 | 15.91 | 16.16 |
| format | 17.41 | 17.52 | 17.27 | 16.72 | 16.57 | 17.29 |
| ceiling_count | 17.56 | 17.56 | 16.92 | 17.23 | 17.29 | 17.45 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 3.491 | 845.1 |
| 8000 | 11592 | 9.828 | 1179.5 |
| 32000 | 46810 | 39.579 | 1182.7 |
| 64000 | 93335 | 75.589 | 1234.8 |
