### Throughput by concurrency (8 categories; counting ceiling excluded)

| C | exl3tp3a11 agg tok/s | exl3tp3a11 per-stream | exl3tp3a11 TTFT s | exl3tp4b agg tok/s | exl3tp4b per-stream | exl3tp4b TTFT s |
|---|---|---|---|---|---|---|
| C1 | 46.0 | 51.46 | 0.305 | 41.62 | 46.36 | 0.37 |
| C2 | 73.37 | 43.58 | 0.634 | 62.18 | 37.55 | 0.768 |
| C3 | 100.14 | 38.52 | 0.412 | 101.14 | 38.17 | 0.442 |
| C4 | 118.41 | 35.16 | 0.441 | 106.72 | 30.88 | 0.507 |
| C5 | 134.44 | 31.67 | 0.46 | 129.15 | 30.4 | 0.563 |
| C6 | 152.89 | 29.62 | 0.5 | 141.19 | 27.7 | 0.579 |

### exl3tp4b / exl3tp3a11 ratio

| C | aggregate | per-stream |
|---|---|---|
| C1 | 0.90x | 0.90x |
| C2 | 0.85x | 0.86x |
| C3 | 1.01x | 0.99x |
| C4 | 0.90x | 0.88x |
| C5 | 0.96x | 0.96x |
| C6 | 0.92x | 0.94x |

### Per-stream tok/s by category at C1

| category | exl3tp3a11 | exl3tp4b |
|---|---|---|
| coding | 70.05 | 81.54 |
| json | 42.03 | 58.75 |
| narrative | 28.08 | 22.75 |
| prose | 29.55 | 26.75 |
| math | 76.46 | 52.85 |
| reasoning | 50.67 | 46.17 |
| summary | 36.16 | 25.06 |
| format | 78.65 | 56.99 |
| ceiling_count | 83.08 | 65.57 |

### Per-stream tok/s by category at C6

| category | exl3tp3a11 | exl3tp4b |
|---|---|---|
| coding | 40.86 | 53.65 |
| json | 28.29 | 30.22 |
| narrative | 12.97 | 9.54 |
| prose | 17.33 | 11.6 |
| math | 38.31 | 27.53 |
| reasoning | 28.56 | 26.28 |
| summary | 19.89 | 19.07 |
| format | 50.78 | 43.74 |
| ceiling_count | 55.69 | 62.9 |

### Cold prefill (unique prefix)

| target | exl3tp3a11 tokens | exl3tp3a11 TTFT s | exl3tp3a11 tok/s | exl3tp4b tokens | exl3tp4b TTFT s | exl3tp4b tok/s |
|---|---|---|---|---|---|---|
| 2000 | 2950 | 2.635 | 1119.4 | 2950 | 2.224 | 1326.2 |
| 8000 | 11592 | 9.774 | 1186.0 | 11592 | 8.27 | 1401.7 |
| 32000 | 46810 | 39.41 | 1187.8 | 46810 | 48.899 | 957.3 |
| 64000 | 93335 | 77.865 | 1198.7 | 93335 | 91.012 | 1025.5 |

- exl3tp3a11: C1 per-stream peak 83.1, median 50.7 tok/s; C6 aggregate peak 308.0 tok/s
- exl3tp4b: C1 per-stream peak 81.5, median 52.9 tok/s; C6 aggregate peak 350.3 tok/s
