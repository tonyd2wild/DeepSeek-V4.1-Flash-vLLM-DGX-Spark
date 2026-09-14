## e04-bss (2026-09-14T07:24:56Z)

speed run 2026-09-14 SCREEN e04-bss

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 58.13 | 63.7 | 0.218 |
| C3 | 110.12 | 43.3 | 0.335 |
| C6 | 186.52 | 35.98 | 0.407 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 88.06 | 57.11 | 47.94 |
| json | 76.41 | 51.02 | 39.34 |
| narrative | 32.26 | 21.6 | 16.46 |
| prose | 38.44 | 27.24 | 21.77 |
| math | 81.08 | 59.38 | 50.38 |
| reasoning | 67.92 | 42.49 | 36.56 |
| summary | 34.14 | 27.14 | 20.39 |
| format | 91.29 | 60.43 | 54.97 |
| ceiling_count | 107.39 | 83.84 | 71.66 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 7.135 | 1624.7 |
| 32000 | 46810 | 26.65 | 1756.5 |
