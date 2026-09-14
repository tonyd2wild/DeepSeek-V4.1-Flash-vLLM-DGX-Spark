# Speed run, 2026-09-14 (overnight, 02:00-08:00 ET)

Goal: make DeepSeek-V4.1-Flash faster on the four DGX Sparks. The targets were decode, prefill, TTFT and throughput at C1-C6 across code, JSON, math, prose, reasoning, tables, summary, narrative and counting. The night ends with the uncensored build (EXL3-Pollard-Abliterated) serving on the best config found.

**Status: in progress.** This page is filled in as results land. The live log is [`STATE.md`](STATE.md).

## Starting point (baseline, as found at 02:00 ET)

`exl3tp4b-ablit`:
- bot-lab-21's EXL3 3.5 bpw experts with the abliterated `wo_b` overlay;
- TP4 on 4 Sparks, DSpark k=5, CUDA graphs FULL_AND_PIECEWISE;
- 300K context, gmu 0.80, vision and tool calling on;
- node-local Engram rows.

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 56.27 | 63.19 | 0.276 |
| C2 | 86.89 | 49.95 | 0.310 |
| C3 | 114.70 | 43.60 | 0.352 |
| C4 | 136.30 | 39.70 | 0.419 |
| C5 | 156.62 | 36.72 | 0.449 |
| C6 | 167.68 | 32.76 | 0.456 |

- **Per-stream C1 by category (tok/s):**

  | code | JSON | math | reasoning | tables | prose | narrative | summary | counting |
  |---|---|---|---|---|---|---|---|---|
  | 84.0 | 75.4 | 78.5 | 70.2 | 91.1 | 37.0 | 34.9 | 34.5 | 98.3 |

- **Cold prefill:** 1,369 / 1,443 / 1,454 / 1,468 tok/s at 2,950 / 11,592 / 46,810 / 93,335 prompt tokens.
- **KV pool:** 3,274,912 tokens.

## How each change was tested

- **One change per boot.** Each boot is scripted: `tools/sr_mkgo.sh`, `tools/sr_boot.sh`, `tools/sr_run.sh`.
- **The same SCREEN protocol on every boot** (`tools/sr_screen.sh`):
  - an idle-vs-back-to-back step test;
  - a quality gate (`tools/sr_quality.py`: count, JSON, runnable code, math, prose);
  - a discarded warm-up pass;
  - the fixed v1 prompt set at C1/C3/C6, plus cold prefill at 8K and 32K.
- **Final confirmation:** the best config gets a full C1-C6 run.

## Results

SCREEN results against the baseline SCREEN. Aggregate tok/s is the mean of the 8 categories; per-stream numbers are C1. Prefill is cold on each fresh boot, compared with the baseline's cold full-bench numbers (1,443 at 8K, 1,454 at 32K).

| # | change (stacked) | KV pool | C1 agg | C3 agg | C6 agg | code | JSON | math | prose | prefill 8K / 32K | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | as found | 3,274,912 | 54.3 | 114.8 | 166.3 | 80.6 | 63.6 | 78.2 | 37.3 | 1,443 / 1,454 | - |
| E01 | `NCCL_MAX_NCHANNELS=8` (bot-lab-21) | 3,512,346 (+7.3%) | 57.4 (+5.8%) | 117.3 (+2.2%) | 184.2 (+10.7%) | 85.4 | 72.5 | 82.9 | 36.5 | 1,279 / 1,486 | - |
| E02 | + Engram fast staging (ours) | 3,505,010 (+7.0%) | **58.8 (+8.4%)** | **119.1 (+3.8%)** | **189.5 (+13.9%)** | 83.5 | **84.0** | 81.8 | 37.7 | **1,665 / 1,751** | PASS |
| E03 | E02 + b12x MXFP8 dense kernel (dropped) | 3,484,154 (+6.4%) | 54.6 (+0.5%) | 118.1 (+2.9%) | 188.1 (+13.1%) | 82.1 | 59.0 | 77.8 | 37.6 | 1,566 / 1,691 | PASS |

- **Prefill probe** (one 39,624-token cold prompt): baseline 1,671 tok/s (23.7 s TTFT); E02 **2,016 tok/s** (19.7 s).
- **GPU utilization during that prefill** is still 43-47%, so host-side work is still the prefill limit. That's the next target.

## What we built tonight

- **Engram fast staging** (`patches/dsv41-exl3-sr1/engram.py`, env `DSV41_ENGRAM_FAST=1`):
  - A numpy memmap gather of the de-duplicated Engram rows, split across the read pool, instead of a Python `preadv` loop per row.
  - Raw fp8 rows and scales go to the GPU, and dequantization happens there.
  - Bit-identical to the old path: `tools/test_stage_fast.py`, 4 passes.
  - Offline on a GB10, a prefill-sized gather (98,304 rows) takes 311 ms cold and 17 ms warm, against 589-655 ms and 355-486 ms for the old path.
- **An image with b12x** (`vllm-dsv41:exl3b` = exl3a + `b12x==1.3.0`), so vLLM can pick the b12x MXFP8 kernel.

## Credits

Tonight's levers came from other people's work:
- **bot-lab-21:**
  - the EXL3 3.5 bpw Pollard checkpoint and its V4.1 overlays;
  - the Hugging Face tuning ladder that measured `NCCL_MAX_NCHANNELS=8`, async scheduling and the b12x MXFP8 kernel on this model and hardware.
- **Zeuss5/cuda-exl3** (with @NNNtrance): the EXL3 CUDA plugin and its GB10 performance notes.
- **MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks:** the SGLang recipe. Its NCCL settings, b12x dense-kernel routing and prefill numbers set tonight's targets.
- **im0xMagnus:**
  - the vLLM commit pin (PR #3), repo-relative diffs (PR #4) and the `verify5` fix (PR #5);
  - the 8-Spark fork and issue #1 data.
- **ecohash-co and hyudryu:** the issue #2 bisect and the `expandable_segments` finding.
- **tmolteno:** issue #1 data.
- **The vLLM team** (dsv41-feat), and **Kai** (the original Engram-on-disk and SM12x patches).
