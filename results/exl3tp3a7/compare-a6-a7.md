### Throughput by concurrency (8 categories; counting ceiling excluded)

| C | exl3tp3a6 agg tok/s | exl3tp3a6 per-stream | exl3tp3a6 TTFT s | exl3tp3a7 agg tok/s | exl3tp3a7 per-stream | exl3tp3a7 TTFT s |
|---|---|---|---|---|---|---|
| C1 | 16.92 | 17.38 | 0.262 | 25.37 | 26.5 | 0.253 |
| C2 | 32.65 | 17.09 | 0.337 | 44.95 | 23.93 | 0.317 |
| C3 | 46.7 | 16.64 | 0.399 | 61.02 | 21.88 | 0.382 |
| C4 | 59.13 | 16.19 | 0.682 | 73.51 | 20.2 | 0.67 |
| C5 | 75.66 | 16.37 | 0.48 | 90.16 | 19.64 | 0.47 |
| C6 | 90.55 | 16.35 | 0.511 | 108.75 | 19.75 | 0.489 |

### exl3tp3a7 / exl3tp3a6 ratio

| C | aggregate | per-stream |
|---|---|---|
| C1 | 1.50x | 1.52x |
| C2 | 1.38x | 1.40x |
| C3 | 1.31x | 1.31x |
| C4 | 1.24x | 1.25x |
| C5 | 1.19x | 1.20x |
| C6 | 1.20x | 1.21x |

### Per-stream tok/s by category at C1

| category | exl3tp3a6 | exl3tp3a7 |
|---|---|---|
| coding | 17.86 | 26.7 |
| json | 17.37 | 27.05 |
| narrative | 17.44 | 26.87 |
| prose | 17.1 | 26.75 |
| math | 16.98 | 25.94 |
| reasoning | 17.3 | 26.2 |
| summary | 17.6 | 26.44 |
| format | 17.41 | 26.02 |
| ceiling_count | 17.56 | 27.06 |

### Per-stream tok/s by category at C6

| category | exl3tp3a6 | exl3tp3a7 |
|---|---|---|
| coding | 16.94 | 21.42 |
| json | 16.51 | 19.84 |
| narrative | 15.57 | 17.13 |
| prose | 15.7 | 19.85 |
| math | 16.58 | 20.85 |
| reasoning | 16.03 | 18.97 |
| summary | 16.16 | 18.14 |
| format | 17.29 | 21.76 |
| ceiling_count | 17.45 | 23.02 |

### Cold prefill (unique prefix)

| target | exl3tp3a6 tokens | exl3tp3a6 TTFT s | exl3tp3a6 tok/s | exl3tp3a7 tokens | exl3tp3a7 TTFT s | exl3tp3a7 tok/s |
|---|---|---|---|---|---|---|
| 2000 | 2950 | 3.491 | 845.1 | 2950 | 3.486 | 846.3 |
| 8000 | 11592 | 9.828 | 1179.5 | 11592 | 9.873 | 1174.1 |
| 32000 | 46810 | 39.579 | 1182.7 | 46810 | 38.134 | 1227.5 |
| 64000 | 93335 | 75.589 | 1234.8 | 93335 | 77.422 | 1205.5 |

- exl3tp3a6: C1 per-stream peak 17.9, median 17.4 tok/s; C6 aggregate peak 102.8 tok/s
- exl3tp3a7: C1 per-stream peak 27.1, median 26.7 tok/s; C6 aggregate peak 134.7 tok/s
