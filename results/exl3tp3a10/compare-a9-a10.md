### Throughput by concurrency (8 categories; counting ceiling excluded)

| C | exl3tp3a9 agg tok/s | exl3tp3a9 per-stream | exl3tp3a9 TTFT s | exl3tp3a10 agg tok/s | exl3tp3a10 per-stream | exl3tp3a10 TTFT s |
|---|---|---|---|---|---|---|
| C1 | 25.69 | 26.79 | 0.25 | 24.23 | 25.41 | 0.275 |
| C2 | 44.2 | 23.59 | 0.332 | 39.26 | 21.74 | 0.59 |
| C3 | 60.45 | 22.12 | 0.586 | 57.52 | 20.93 | 0.397 |
| C4 | 78.09 | 21.1 | 0.424 | 71.71 | 19.53 | 0.495 |
| C5 | 89.36 | 19.47 | 0.485 | 87.58 | 19.32 | 0.517 |
| C6 | 107.47 | 19.62 | 0.489 | 106.13 | 19.34 | 0.497 |

### exl3tp3a10 / exl3tp3a9 ratio

| C | aggregate | per-stream |
|---|---|---|
| C1 | 0.94x | 0.95x |
| C2 | 0.89x | 0.92x |
| C3 | 0.95x | 0.95x |
| C4 | 0.92x | 0.93x |
| C5 | 0.98x | 0.99x |
| C6 | 0.99x | 0.99x |

### Per-stream tok/s by category at C1

| category | exl3tp3a9 | exl3tp3a10 |
|---|---|---|
| coding | 27.71 | 26.95 |
| json | 27.43 | 27.07 |
| narrative | 26.96 | 24.69 |
| prose | 26.88 | 26.71 |
| math | 26.18 | 20.37 |
| reasoning | 26.45 | 26.51 |
| summary | 26.61 | 26.33 |
| format | 26.1 | 24.67 |
| ceiling_count | 27.2 | 27.3 |

### Per-stream tok/s by category at C6

| category | exl3tp3a9 | exl3tp3a10 |
|---|---|---|
| coding | 20.33 | 19.04 |
| json | 21.74 | 19.6 |
| narrative | 17.23 | 17.19 |
| prose | 18.5 | 18.53 |
| math | 19.98 | 21.41 |
| reasoning | 19.03 | 18.92 |
| summary | 18.16 | 18.16 |
| format | 21.96 | 21.9 |
| ceiling_count | 23.19 | 22.99 |

### Cold prefill (unique prefix)

| target | exl3tp3a9 tokens | exl3tp3a9 TTFT s | exl3tp3a9 tok/s | exl3tp3a10 tokens | exl3tp3a10 TTFT s | exl3tp3a10 tok/s |
|---|---|---|---|---|---|---|
| 2000 | 2950 | 4.032 | 731.6 | 2950 | 4.142 | 712.2 |
| 8000 | 11592 | 10.03 | 1155.7 | 11592 | 9.815 | 1181.0 |
| 32000 | 46810 | 38.871 | 1204.2 | 46810 | 40.82 | 1146.7 |
| 64000 | 93335 | 75.53 | 1235.7 | 93335 | 80.837 | 1154.6 |

- exl3tp3a9: C1 per-stream peak 27.7, median 26.9 tok/s; C6 aggregate peak 135.8 tok/s
- exl3tp3a10: C1 per-stream peak 27.3, median 26.5 tok/s; C6 aggregate peak 134.8 tok/s
