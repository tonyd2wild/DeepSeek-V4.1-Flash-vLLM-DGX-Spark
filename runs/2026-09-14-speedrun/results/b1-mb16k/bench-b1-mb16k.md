## b1-mb16k (2026-09-14T09:06:52Z)

speed run 2026-09-14 SCREEN b1-mb16k

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 60.92 | 67.17 | 0.216 |
| C3 | 127.52 | 48.19 | 0.305 |
| C6 | 185.49 | 36.71 | 0.405 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 86.09 | 68.12 | 55.03 |
| json | 84.2 | 60.09 | 37.74 |
| narrative | 32.08 | 22.07 | 17.8 |
| prose | 39.85 | 29.88 | 23.02 |
| math | 85.33 | 61.23 | 52.65 |
| reasoning | 77.81 | 49.87 | 35.63 |
| summary | 39.5 | 25.62 | 19.62 |
| format | 92.5 | 68.67 | 52.16 |
| ceiling_count | 114.32 | 87.9 | 70.1 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 5.902 | 1964.1 |
| 32000 | 46810 | 24.237 | 1931.3 |
