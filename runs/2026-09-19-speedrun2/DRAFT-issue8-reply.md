Measured tonight (2026-09-19, speed run 2), as promised. One-change A/B on the live EXL3 TP4 stack at 300K: base config vs base + `NCCL_BUFFSIZE=1048576`, same boot path, same SCREEN bench (C1/C3/C6 across 8 categories, cold prefill 8K and 32K, idle test), quality gate PASS on both.

| | base | + `NCCL_BUFFSIZE=1048576` |
|---|---|---|
| KV pool (tokens) | 3,725,682 | 3,726,094 |
| C1 / C3 / C6 aggregate tok/s | 62.0 / 121.5 / 195.0 | 59.5 / 127.4 / 198.4 |
| C1 code tok/s | 92.2 | 95.8 |
| cold prefill 11.6K / 46.8K tokens | 1,917 / 2,021 | 1,920 / 2,036 |
| idle test count / code tok/s | 117.9 / 84.7 | 117.8 / 86.1 |

So on our lane the KV pool does not move (+412 tokens), and everything else is inside the boot-to-boot noise we see between identical configs (about ±4% on single cells). The difference from your setup is where decode all-reduces go: ours run over the b12x RoCE one-shot path up to 2 MB, with NCCL only carrying prefill-sized collectives, so the smaller NCCL buffers have nothing to give back to the pool. Your TP4+DCP2 numbers stand for an NCCL-decode fleet; I'd keep the flag as a recommendation there and not here. Leaving it out of our default config, and leaving this open in case someone measures it on a third layout.
