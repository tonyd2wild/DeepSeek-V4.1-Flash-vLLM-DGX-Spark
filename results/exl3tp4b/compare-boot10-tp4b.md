### Throughput by concurrency (8 categories; counting ceiling excluded)

| C | boot10 agg tok/s | boot10 per-stream | boot10 TTFT s | exl3tp4b agg tok/s | exl3tp4b per-stream | exl3tp4b TTFT s |
|---|---|---|---|---|---|---|
| C1 | 37.95 | 43.12 | 0.441 | 41.62 | 46.36 | 0.37 |
| C2 | 64.3 | 37.43 | 0.441 | 62.18 | 37.55 | 0.768 |
| C3 | 78.7 | 30.62 | 0.793 | 101.14 | 38.17 | 0.442 |
| C4 | 85.72 | 24.66 | 0.579 | 106.72 | 30.88 | 0.507 |
| C5 | 114.2 | 27.01 | 0.585 | 129.15 | 30.4 | 0.563 |
| C6 | 131.86 | 25.35 | 0.502 | 141.19 | 27.7 | 0.579 |

### exl3tp4b / boot10 ratio

| C | aggregate | per-stream |
|---|---|---|
| C1 | 1.10x | 1.08x |
| C2 | 0.97x | 1.00x |
| C3 | 1.29x | 1.25x |
| C4 | 1.24x | 1.25x |
| C5 | 1.13x | 1.13x |
| C6 | 1.07x | 1.09x |

### Per-stream tok/s by category at C1

| category | boot10 | exl3tp4b |
|---|---|---|
| coding | 73.78 | 81.54 |
| json | 52.1 | 58.75 |
| narrative | 24.93 | 22.75 |
| prose | 24.37 | 26.75 |
| math | 50.86 | 52.85 |
| reasoning | 37.79 | 46.17 |
| summary | 25.72 | 25.06 |
| format | 55.41 | 56.99 |
| ceiling_count | 62.19 | 65.57 |

### Per-stream tok/s by category at C6

| category | boot10 | exl3tp4b |
|---|---|---|
| coding | 41.47 | 53.65 |
| json | 27.5 | 30.22 |
| narrative | 9.79 | 9.54 |
| prose | 13.76 | 11.6 |
| math | 34.26 | 27.53 |
| reasoning | 24.55 | 26.28 |
| summary | 16.09 | 19.07 |
| format | 35.35 | 43.74 |
| ceiling_count | 33.41 | 62.9 |

### Cold prefill (unique prefix)

| target | boot10 tokens | boot10 TTFT s | boot10 tok/s | exl3tp4b tokens | exl3tp4b TTFT s | exl3tp4b tok/s |
|---|---|---|---|---|---|---|
| 2000 | 2950 | 3.27 | 902.2 | 2950 | 2.224 | 1326.2 |
| 8000 | 11592 | 11.293 | 1026.5 | 11592 | 8.27 | 1401.7 |
| 32000 | 46810 | 30.426 | 1538.5 | 46810 | 48.899 | 957.3 |
| 64000 | 93335 | 78.173 | 1194.0 | 93335 | 91.012 | 1025.5 |

- boot10: C1 per-stream peak 73.8, median 50.9 tok/s; C6 aggregate peak 225.5 tok/s
- exl3tp4b: C1 per-stream peak 81.5, median 52.9 tok/s; C6 aggregate peak 350.3 tok/s
