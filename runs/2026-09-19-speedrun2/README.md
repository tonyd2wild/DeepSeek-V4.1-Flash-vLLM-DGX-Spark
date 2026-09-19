# Speed run 2, 2026-09-19 (overnight, 01:50-09:25 ET)

Goal: a fresh C1-C6 baseline of the serving EXL3 lane (TTFT, decode, prefill, throughput across code, JSON, math, reasoning, tables, summary, prose, narrative, counting), then tune configs and kernels for decode, prefill, load time and concurrency, and leave the best DeepSeek-V4.1-Flash TP4 config serving at 9:25 AM ET.

**Status: complete.** The final config has been serving since 08:52 UTC (4:52 AM ET), verified with the full bench, two needles and a C8/C12 pass. The live log is [`STATE.md`](STATE.md); the go script is `asusi:~/sr2-e12-final500k-s16-go.sh`.

## Final vs baseline (same full bench, same prompts)

The baseline is the 500K lane exactly as found at 05:53 UTC, which had been serving for 15 hours. The final is the same lane plus tonight's changes, fresh.

| | Baseline (as found) | **Final (serving)** | delta |
|---|---|---|---|
| C1 / C2 / C3 aggregate tok/s | 58.3 / 90.6 / 110.8 | **61.3 / 102.3 / 128.9** | +5% / +13% / +16% |
| C4 / C5 / C6 aggregate | 146.0 / 164.3 / 184.8 | **153.4 / 174.3 / 189.3** | +5% / +6% / +2% |
| C1 per-stream | 64.0 | **67.4** | +5% |
| TTFT C1 / C6 (s) | 0.237 / 0.429 | **0.205 / 0.381** | -13% / -11% |
| Cold prefill, 2,950 tokens | 1,378 | **1,939** | +41% |
| Cold prefill, 11,592 tokens | 1,322 | 1,878 * | +42% |
| Cold prefill, 46,810 tokens | 1,419 | 2,007 * | +41% |
| Cold prefill, 93,335 tokens | 1,401 | **2,048** | +46% |
| Cold prefill, unique 39.6K probe | 1,664 | **2,223** | +34% |
| Idle test, count / code tok/s | 98.2 / 65.2 | **117.3 / 95.3** | +19% / +46% |
| C8 coding / counting aggregate | not possible (max 8 streams) | **376.4 / 490.0** | new |
| C12 coding / counting aggregate | not possible | **474.7 / 639.9** | new |
| Boot to serving | ~504 s | **~440 s** | -63 s |
| KV pool (tokens, reported) | 4,295,367 | 4,011,085 | see note |
| Needle 65,076 / 130,291 tokens | not run | **PASS / PASS** (2,112 / 1,885 tok/s) | |
| Quality gate, vision + tools | PASS, 7/7 | PASS, 7/7 | |

\* The full bench's 8K/32K prefill cells hit the prefix cache on the final lane (the e12 screen had just run the same prompts; this build has no reset endpoint), so those two numbers are from e12's own screen at boot on the identical config, before anything was cached. Every other cell is from the final run.

**Per-stream C1 by category, final:** code 90.7, JSON 75.8, math 87.5, reasoning 75.1, tables 98.6, summary 41.1, prose 39.0, narrative 31.4, counting 113.0.

**What is actually new and what is not.** Be careful reading the deltas. Most of the prefill and idle-decode gap between "baseline" and "final" is a lane-age effect, not tuning: the same 500K script booted fresh (e10) already measured 1,869 / 1,963 prefill and 114.7 / 85.1 idle. The tuning on top of a fresh lane is worth about +3-5% decode, +3-6% prefill, -63 s boot and the C8-C12 capability. Both facts matter; the first one arguably more.

## What changed (the final config)

`sr2-e12-final500k-s16-go.sh` = the 09-14 best config plus:

| Change | How | Measured effect | Credit |
|---|---|---|---|
| SwiGLU clamp inside the cuda-exl3 GLU kernel | image `vllm-dsv41:exl3b-roce-lim`: `exl3_moe_glu_had_in` gains `float limit=0.`; the clamp runs in the fp32 lambda, NaN-propagating, bit-identical to the torch fallback for bf16 (10.0 is exact). Removes two `clamp_` kernels per MoE layer per step | e06 vs e00: C1 61.3 vs 59.5, C6 195.2 vs 187.6, math 88.4 vs 81.3, prefill 1,922/2,017 vs 1,867/1,959. Probe 2,226 vs 2,098 | ours; kernel by Zeuss5/cuda-exl3, clamp semantics from bot-lab-21's `exl3_moe.py` overlay |
| Engram all-gather through the RoCE one-shot path | patch set `dsv41-exl3-sr4`, `engram.py`: gather the free `[T, L*D]` view along the last dim instead of `dim=1` on the 3-D tensor, which b12x refused (fell back to pynccl plus a `movedim` inside the FULL graph). Byte-identical | decode flat within noise, as predicted for ~0.3-0.5 ms/step | ours; RoCE path by @original-el8 and @lukealonso (local-inference-lab/b12x) |
| Draft loader reads 3 of 49 shards | `default_loader.py`: the DSpark draft is loaded by its own loader instance, which walked all 49 shards for 97 `mtp.*` params (32 s head, 59 s NFS workers). Now filtered by the safetensors index | boot 441 s vs 504 s | ours |
| `SEQS=16` | `--max-num-seqs 16`, 30 CUDA graph sizes to 96 | C8 coding 376-408, C12 coding 475-495 aggregate; C1-C6 within noise; KV -5% | ours |
| 500K context, confirmed free | unchanged from 09-14 | fresh 500K equals fresh 300K on every cell (e10 vs e00) | |

Opt-in, measured, not default: `GMU=0.82` at 500K gives KV 4,775,278 (+13%) with 15 GB left on Spark4 and Asusi (the bar that kept it out on 09-14 is now met); decode moved -1 to -8% on the screen, all inside the noise band but all the same direction.

## Tried and dropped (one change per boot, on the best so far)

| # | change | result |
|---|---|---|
| e05 | RoCE all-reduce cap 2 MB -> 32 MB (prefill chunks through RoCE) | decode within noise, 8K prefill -9%, KV -1.7%. A one-shot all-reduce at 20 MB is bandwidth-bound |
| e07 | `DSV41_ENGRAM_FAST_GATHER_ROWS=4` (smaller parallel fault chains for cold decode rows) | prefill -15%: tiny tasks shred the 8K-row prefill gathers. Idle code +7% only |
| e08 | `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=8192` (overlap the shared expert at prefill) | prefill -2%. No idle SMs at prefill on 48 SMs |
| e01 | `NCCL_BUFFSIZE=1048576` (issue #8) | KV +412 tokens, everything else noise. Decode all-reduces are on RoCE here, so NCCL buffers have nothing to give back |
| e02 | `GMU=0.82` | opt-in, above |

Prepared but not booted: vLLM #57432 (DSpark acceptance fix) and #56441 (KV-only draft insertion) ported into patch set `dsv41-exl3-sr3` with env toggles. The port found that #57432's two defects live in the SM100 TRTLLM path, which SM121 never executes (we run `DeepseekV4FlashInferSM120Attention._forward_decode`), and #56441 needs a compiled op this image lacks. Expected flat; kept for a future image.

## Findings worth more than the config

1. **Lanes age.** The 15 h old lane ran ~30% slower on prefill and 17-30% slower on idle decode than the same config fresh. Dropping page cache on all four nodes the way the cron does did not reproduce it (probe 2,226 before, 2,226 and 2,223 after). Allocator state, prefix-cache churn and NFS client state are the remaining candidates. A re-check of the final lane at 3.6 hours old showed no aging at all (probe 2,230 vs 2,223 fresh; idle 115.9 / 95.7 vs 117.3 / 95.3), so either it needs longer or it was specific to that lane's history (it was booted 90 minutes after Reddie's second reboot of 2026-09-18). Until the cause is found, a fresh boot recovers it.
2. **The "500K tax" was that aging.** Fresh 500K equals fresh 300K. And the KV pool number is a reporting artifact: `kv_cache_utils.py:2306` reports `max_concurrency x max_model_len`; actual KV bytes were 29.97 GiB at 500K vs 30.14 GiB at 300K. Same pages.
3. **Where the decode step goes** (per-step profile with bpftrace on all four worker main threads): GPU 49-52 ms, of which ~45 ms is batch-independent; host bubble 1.0-1.2 ms floor, 5.2 ms p50 / 11.8 ms p90 on cold Engram rows. ~90% of spikes are page faults on one rank (Bluey, per-fault 465 us vs 182 us on Spark4). The remaining spike cluster was our own liveness probe (16 tokens every 60 s -> a PIECEWISE step plus C2-speed steps); it now sends 2 tokens every 120 s. Not clocks or power: 2171-2190 MHz, 23-25 W at C1, no throttle reasons.
4. **DSpark acceptance is the code/math ceiling.** k=5 caps tok/step at 5.85; counting sits at 97% of it. Per-category accepted length: count 4.8-5.0, code 4.0-4.3, tables 4.0, math 3.3, JSON 3.3, reasoning 3.0, summary 1.2, prose 1.0, narrative 0.9. The "first 10 steps of code are slow" observation was this ramp (tok/step 4.0 -> 5.35), not ms/step.
5. **The step has exactly one host round trip:** the Engram hash sync. It cannot be prestaged: step N+1's rows hash over the draft tokens produced at the very end of step N. Speculative prefetch is not possible for the same reason.
6. **PP2xTP2 would not help** this class of problem: with expert parallelism `ep_size` derives from TP only, so local expert-layers = 40 x 384 / 4 regardless of the split; per-rank weights and buffers are byte-identical. DSpark also refuses PP.

## Tools added (`tools/`)

`sr2_mkgo.sh` (go scripts from the live 500K base), `sr2_run.sh` (boot + screen + table row), `sr2_dq.sh` (dynamic queue with a STOP_AT and the two-failure restore rule), `s2_agingtest.sh`, `s2_final_measure.sh`, `roll_lim_image.sh`, `catch_stack.sh` (py-spy on a worker during a memory climb, from the NVFP4 lane), plus the four cuda-exl3 source diffs in `patches/exl3-limit/` and patch sets `sr3`, `sr4`.

## Credits

- **bot-lab-21:** the EXL3 3.5 bpw Pollard checkpoint and the `exl3_moe.py` overlay whose clamp semantics the kernel now implements.
- **Zeuss5/cuda-exl3** (and @NNNtrance): the EXL3 CUDA plugin; the `limit` change is a 3-file diff on their HEAD.
- **@original-el8, @lukealonso** (local-inference-lab/b12x): the RoCE one-shot collectives the Engram gather now uses.
- **gitbisector** (issue #8): the NCCL proxy-buffer analysis; measured here as neutral on a RoCE-decode fleet.
- **vLLM team:** dsv41-feat, PRs #57432 and #56441 (ported, parked).
- **Kai:** the original disk Engram and SM12x page patches this lane still runs on.
