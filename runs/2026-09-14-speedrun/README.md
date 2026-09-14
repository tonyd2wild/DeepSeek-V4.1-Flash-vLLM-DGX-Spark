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
| E09 | E02 + 128 Engram read threads | 3,501,822 (+6.9%) | 56.3 (+3.7%) | 118.0 (+2.9%) | 185.4 (+11.5%) | 85.5 | 71.5 | 78.3 | 34.3 | **1,897 / 1,983** | PASS |
| **b1** | **E02 + E09 + E19 (best so far)** | **3,625,526 (+10.7%)** | **58.9 (+8.5%)** | **124.8 (+8.7%)** | **197.9 (+19.0%)** | **91.8** | 62.0 | 80.2 | **42.5** | **1,918 / 1,977** | PASS, needle 65K/131K PASS |
| b1-idxsplit | b1 + indexer prefill TP-split (ours) | 3,588,541 (+9.6%) | 60.0 (+10.6%) | 126.5 (+10.2%) | 190.4 (+14.5%) | 92.9 | 64.3 | 84.5 | 40.4 | 1,895 / 1,993 | PASS; needle 131K prefill 2,042 vs b1 1,921 (+6.3%) |
| E14 | E02 + indexer prefill TP-split (ours) | 3,486,710 (+6.5%) | 57.5 (+5.9%) | 122.5 (+6.7%) | 191.8 (+15.3%) | 86.0 | 74.3 | 79.3 | 37.4 | 1,653 / 1,765 | PASS, needle 65K/131K PASS |
| E19 | E02 + b12x RoCE one-shot all-reduce (@original-el8, @lukealonso) | **3,608,160 (+10.2%)** | **60.1 (+10.8%)** | **125.3 (+9.2%)** | **191.7 (+15.2%)** | **92.5** | 72.3 | 81.1 | **43.9** | 1,682 / 1,793 | PASS |
| E05 | E02 + DSpark k=10 (dropped) | 3,450,837 (+5.4%) | 44.8 (-17.5%) | 87.6 (-23.7%) | 126.2 (-24.1%) | 74.3 | 52.8 | 66.5 | 23.9 | 1,723 / 1,768 | PASS |
| E04 | E02 + batch-sharded sampling (dropped) | 3,494,513 (+6.7%) | 58.1 (+7.1%) | 110.1 (-4.0%) | 186.5 (+12.1%) | 88.1 | 76.4 | 81.1 | 38.4 | 1,625 / 1,757 | PASS |
| E03 | E02 + b12x MXFP8 dense kernel (dropped) | 3,484,154 (+6.4%) | 54.6 (+0.5%) | 118.1 (+2.9%) | 188.1 (+13.1%) | 82.1 | 59.0 | 77.8 | 37.6 | 1,566 / 1,691 | PASS |

- **Prefill probe** (one 39,624-token cold prompt): baseline 1,671 tok/s (23.7 s TTFT); E02 2,016 tok/s (19.7 s); b1 **2,124 tok/s** (18.7 s).
- **Long context (needle at 30% depth, cold):** b1 finds the passphrase at 65K (1,998 tok/s prefill) and 131K (1,921 tok/s).
- **GPU utilization during prefill** was 43-47% on E02, so host-side work is still a prefill limit.
- Dropped after testing: b12x MXFP8 dense kernel (E03), batch-sharded sampling (E04), DSpark k=10 (E05). Dropped after offline tests or code study: native MXFP8 `wo_a` (slower at decode sizes under graphs), skipping the drafter on non-final prefill chunks (0.3-0.6%), draft-KV page 256 (does not apply to V4.1's cache groups), forcing MoE block_m 16 (already the decode tier).

## What we built tonight

- **Engram fast staging** (`patches/dsv41-exl3-sr1/engram.py`, env `DSV41_ENGRAM_FAST=1`):
  - A numpy memmap gather of the de-duplicated Engram rows, split across the read pool, instead of a Python `preadv` loop per row.
  - Raw fp8 rows and scales go to the GPU, and dequantization happens there.
  - Bit-identical to the old path: `tools/test_stage_fast.py`, 4 passes.
  - Offline on a GB10, a prefill-sized gather (98,304 rows) takes 311 ms cold and 17 ms warm, against 589-655 ms and 355-486 ms for the old path.
- **An image with b12x** (`vllm-dsv41:exl3b` = exl3a + `b12x==1.3.0`), so vLLM can pick the b12x MXFP8 kernel.
- **The b12x RoCE all-reduce on our stack** (`patches/roce-port/`): `vllm-dsv41:exl3b-roce` overlays `b12x.comm.roce` from local-inference-lab/b12x at `b58f34ea` (no PyPI release has it yet) and builds its RDMA proxy. local-inference-lab/vllm#597 is ported onto our dsv41-feat tree as 5 mounted files (patch set `dsv41-exl3-sr1roce`). All-reduces up to 2 MB and all-gathers up to 16 MB go over RoCE; NCCL stays the fallback. Rollback: `VLLM_ENABLE_ROCE_ALLREDUCE=0`.
- **Indexer prefill TP-split** (`patches/idxsplit-draft/`, env `DSV41_INDEXER_TP_SPLIT=1`): each rank computes the lightning indexer's top-k for a quarter of the prefill rows, then one all-gather. Tested on a GB10 against the unsplit path (same top-k sets, byte-equal candidates, identical ranks). Neutral up to 40K on E02; the long-context A/B is below.
- **Harness:** a dynamic queue (`tools/sr_dq.sh`, edit `queue.txt` while it runs), one-shot idle-window GPU tests between boots (`prehook.sh` in `tools/sr_run.sh`), and a log-only generation liveness probe (`tools/liveness.sh`).

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
- **Jason @original-el8 and Luke Alonso @lukealonso (local-inference-lab):** b12x RoCEnante, the one-shot RoCE all-reduce (b12x#295, #315, commits aa90a277 and 1a7e3ec2), and its vLLM integration (local-inference-lab/vllm#597). Tonight's biggest decode win.
- **@tobymao:** the b12x#313 stuck-rank report, which is why the final config runs a generation liveness probe.
- **The tuning backlog shipped in the model folder** (`TUNING_BACKLOG.md`): pointed us at the RoCE all-reduce, the draft-page study and the block_m and NCCL thread levers.
- **@NNNtrance:** the draft-KV page study (Zeuss5/cuda-exl3#2). It does not carry over to V4.1's cache groups, but it led to the pool-geometry notes in `STATE.md`.
- **NVIDIA's NCCL tuning guide** (Williams, Mubarak, Caton, Nicely) and **UMD PSSG** (arXiv 2511.09557): the channel and algorithm reasoning behind the NCCL experiments.
- **The vLLM team** (dsv41-feat), and **Kai** (the original Engram-on-disk and SM12x patches).
