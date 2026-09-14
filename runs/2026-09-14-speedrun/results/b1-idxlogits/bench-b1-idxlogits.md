## b1-idxlogits (2026-09-14T09:20:01Z)

speed run 2026-09-14 SCREEN b1-idxlogits

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.77 | 64.35 | 0.21 |
| C3 | 121.87 | 47.16 | 0.314 |
| C6 | 193.48 | 37.88 | 0.409 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 88.23 | 70.75 | 56.49 |
| json | 66.53 | 49.53 | 40.85 |
| narrative | 32.78 | 24.2 | 18.39 |
| prose | 41.46 | 27.69 | 22.6 |
| math | 85.23 | 59.72 | 53.15 |
| reasoning | 75.88 | 44.87 | 35.82 |
| summary | 33.55 | 25.4 | 19.92 |
| format | 91.14 | 75.08 | 55.85 |
| ceiling_count | 113.72 | 88.05 | 70.8 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 5.97 | 1941.8 |
| 32000 | 46810 | 23.367 | 2003.2 |
