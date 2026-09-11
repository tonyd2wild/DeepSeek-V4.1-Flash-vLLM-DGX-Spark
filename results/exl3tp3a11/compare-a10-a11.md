### Throughput by concurrency (8 categories; counting ceiling excluded)

| C | exl3tp3a10 agg tok/s | exl3tp3a10 per-stream | exl3tp3a10 TTFT s | exl3tp3a11 agg tok/s | exl3tp3a11 per-stream | exl3tp3a11 TTFT s |
|---|---|---|---|---|---|---|
| C1 | 24.23 | 25.41 | 0.275 | 46.0 | 51.46 | 0.305 |
| C2 | 39.26 | 21.74 | 0.59 | 73.37 | 43.58 | 0.634 |
| C3 | 57.52 | 20.93 | 0.397 | 100.14 | 38.52 | 0.412 |
| C4 | 71.71 | 19.53 | 0.495 | 118.41 | 35.16 | 0.441 |
| C5 | 87.58 | 19.32 | 0.517 | 134.44 | 31.67 | 0.46 |
| C6 | 106.13 | 19.34 | 0.497 | 152.89 | 29.62 | 0.5 |

### exl3tp3a11 / exl3tp3a10 ratio

| C | aggregate | per-stream |
|---|---|---|
| C1 | 1.90x | 2.03x |
| C2 | 1.87x | 2.00x |
| C3 | 1.74x | 1.84x |
| C4 | 1.65x | 1.80x |
| C5 | 1.54x | 1.64x |
| C6 | 1.44x | 1.53x |

### Per-stream tok/s by category at C1

| category | exl3tp3a10 | exl3tp3a11 |
|---|---|---|
| coding | 26.95 | 70.05 |
| json | 27.07 | 42.03 |
| narrative | 24.69 | 28.08 |
| prose | 26.71 | 29.55 |
| math | 20.37 | 76.46 |
| reasoning | 26.51 | 50.67 |
| summary | 26.33 | 36.16 |
| format | 24.67 | 78.65 |
| ceiling_count | 27.3 | 83.08 |

### Per-stream tok/s by category at C6

| category | exl3tp3a10 | exl3tp3a11 |
|---|---|---|
| coding | 19.04 | 40.86 |
| json | 19.6 | 28.29 |
| narrative | 17.19 | 12.97 |
| prose | 18.53 | 17.33 |
| math | 21.41 | 38.31 |
| reasoning | 18.92 | 28.56 |
| summary | 18.16 | 19.89 |
| format | 21.9 | 50.78 |
| ceiling_count | 22.99 | 55.69 |

### Cold prefill (unique prefix)

| target | exl3tp3a10 tokens | exl3tp3a10 TTFT s | exl3tp3a10 tok/s | exl3tp3a11 tokens | exl3tp3a11 TTFT s | exl3tp3a11 tok/s |
|---|---|---|---|---|---|---|
| 2000 | 2950 | 4.142 | 712.2 | 2950 | 2.635 | 1119.4 |
| 8000 | 11592 | 9.815 | 1181.0 | 11592 | 9.774 | 1186.0 |
| 32000 | 46810 | 40.82 | 1146.7 | 46810 | 39.41 | 1187.8 |
| 64000 | 93335 | 80.837 | 1154.6 | 93335 | 77.865 | 1198.7 |

- exl3tp3a10: C1 per-stream peak 27.3, median 26.5 tok/s; C6 aggregate peak 134.8 tok/s
- exl3tp3a11: C1 per-stream peak 83.1, median 50.7 tok/s; C6 aggregate peak 308.0 tok/s
