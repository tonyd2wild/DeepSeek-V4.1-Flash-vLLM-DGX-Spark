### Throughput by concurrency (8 categories; counting ceiling excluded)

| C | exl3tp3a7 agg tok/s | exl3tp3a7 per-stream | exl3tp3a7 TTFT s | exl3tp3a9 agg tok/s | exl3tp3a9 per-stream | exl3tp3a9 TTFT s |
|---|---|---|---|---|---|---|
| C1 | 25.37 | 26.5 | 0.253 | 25.69 | 26.79 | 0.25 |
| C2 | 44.95 | 23.93 | 0.317 | 44.2 | 23.59 | 0.332 |
| C3 | 61.02 | 21.88 | 0.382 | 60.45 | 22.12 | 0.586 |
| C4 | 73.51 | 20.2 | 0.67 | 78.09 | 21.1 | 0.424 |
| C5 | 90.16 | 19.64 | 0.47 | 89.36 | 19.47 | 0.485 |
| C6 | 108.75 | 19.75 | 0.489 | 107.47 | 19.62 | 0.489 |

### exl3tp3a9 / exl3tp3a7 ratio

| C | aggregate | per-stream |
|---|---|---|
| C1 | 1.01x | 1.01x |
| C2 | 0.98x | 0.99x |
| C3 | 0.99x | 1.01x |
| C4 | 1.06x | 1.04x |
| C5 | 0.99x | 0.99x |
| C6 | 0.99x | 0.99x |

### Per-stream tok/s by category at C1

| category | exl3tp3a7 | exl3tp3a9 |
|---|---|---|
| coding | 26.7 | 27.71 |
| json | 27.05 | 27.43 |
| narrative | 26.87 | 26.96 |
| prose | 26.75 | 26.88 |
| math | 25.94 | 26.18 |
| reasoning | 26.2 | 26.45 |
| summary | 26.44 | 26.61 |
| format | 26.02 | 26.1 |
| ceiling_count | 27.06 | 27.2 |

### Per-stream tok/s by category at C6

| category | exl3tp3a7 | exl3tp3a9 |
|---|---|---|
| coding | 21.42 | 20.33 |
| json | 19.84 | 21.74 |
| narrative | 17.13 | 17.23 |
| prose | 19.85 | 18.5 |
| math | 20.85 | 19.98 |
| reasoning | 18.97 | 19.03 |
| summary | 18.14 | 18.16 |
| format | 21.76 | 21.96 |
| ceiling_count | 23.02 | 23.19 |

### Cold prefill (unique prefix)

| target | exl3tp3a7 tokens | exl3tp3a7 TTFT s | exl3tp3a7 tok/s | exl3tp3a9 tokens | exl3tp3a9 TTFT s | exl3tp3a9 tok/s |
|---|---|---|---|---|---|---|
| 2000 | 2950 | 3.486 | 846.3 | 2950 | 4.032 | 731.6 |
| 8000 | 11592 | 9.873 | 1174.1 | 11592 | 10.03 | 1155.7 |
| 32000 | 46810 | 38.134 | 1227.5 | 46810 | 38.871 | 1204.2 |
| 64000 | 93335 | 77.422 | 1205.5 | 93335 | 75.53 | 1235.7 |

- exl3tp3a7: C1 per-stream peak 27.1, median 26.7 tok/s; C6 aggregate peak 134.7 tok/s
- exl3tp3a9: C1 per-stream peak 27.7, median 26.9 tok/s; C6 aggregate peak 135.8 tok/s
