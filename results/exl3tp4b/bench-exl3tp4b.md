## exl3tp4b (2026-09-11T19:54:57Z)

TP4 EXL3 exl3tp4b: boot 10 config on the EXL3 3.5bpw checkpoint, 4 Sparks, head Reddie, graphs + DSpark k=5 + vision, gmu 0.80, no KV pin; bench run from Bluey against 192.168.192.2

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 41.62 | 46.36 | 0.37 |
| C2 | 62.18 | 37.55 | 0.768 |
| C3 | 101.14 | 38.17 | 0.442 |
| C4 | 106.72 | 30.88 | 0.507 |
| C5 | 129.15 | 30.4 | 0.563 |
| C6 | 141.19 | 27.7 | 0.579 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 81.54 | 48.29 | 64.32 | 35.81 | 32.17 | 53.65 |
| json | 58.75 | 38.53 | 62.5 | 28.34 | 24.84 | 30.22 |
| narrative | 22.75 | 16.48 | 21.1 | 14.73 | 9.98 | 9.54 |
| prose | 26.75 | 18.28 | 25.41 | 22.6 | 13.85 | 11.6 |
| math | 52.85 | 44.16 | 49.14 | 51.57 | 50.91 | 27.53 |
| reasoning | 46.17 | 34.32 | 29.81 | 39.86 | 37.45 | 26.28 |
| summary | 25.06 | 30.76 | 16.0 | 18.65 | 21.72 | 19.07 |
| format | 56.99 | 69.62 | 37.11 | 35.44 | 52.29 | 43.74 |
| ceiling_count | 65.57 | 88.75 | 46.42 | 41.8 | 68.88 | 62.9 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 2.224 | 1326.2 |
| 8000 | 11592 | 8.27 | 1401.7 |
| 32000 | 46810 | 48.899 | 957.3 |
| 64000 | 93335 | 91.012 | 1025.5 |
