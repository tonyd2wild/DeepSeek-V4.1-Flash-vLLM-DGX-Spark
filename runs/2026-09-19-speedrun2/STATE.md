# Speed run 2 (2026-09-19): STATE. Read this first after any context compaction.

## Deadline and goal (Tony's /goal, 05:50 UTC)
- **Run until 13:25 UTC (9:25 AM ET). At 13:25 the BEST DeepSeek V4.1 Flash TP4 config must be serving.**
- Baseline C1-C6 (TTFT, decode, prefill, throughput; structured, count, prose, json, code, math, narrative) on the
  live EXL3 lane, then tune configs and kernels for decode, prefill, load time and concurrency. Spawn agents as
  needed. Update the repo.

## House rules (carry over)
- One change per boot, stacked on the best so far. Quality gate must PASS to win. Two failed boots in a row: stop,
  restore the last known-good. Nothing deleted anywhere without Tony. System settings (BIOS, driver, clocks,
  sysctl, cron, fstab) are Tony's. Measured numbers only, no absolutes, credit everything used from others.
- **Known-good = the live lane:** `asusi:~/exl3tp4b-ablit-best500k-go.sh`, image `vllm-dsv41:exl3b-roce`, patch set
  `dsv41-exl3-sr2roce`, gmu 0.80, MAXLEN 500000, DSpark k=5, FULL_AND_PIECEWISE graphs. Restore:
  `GO=exl3tp4b-ablit-best500k-go.sh bash /root/restore_exl3tp4b_ablit_best.sh` (root on Reddie).
- All experiments stay at **MAXLEN=500000** so numbers compare to what is served.

## Fleet as found (05:47 UTC)
- Serving 15 h, container up since 2026-09-18 14:57:08 UTC, 0 restarts. Reddie rebooted twice on 09-18 (08:15 and
  13:24 UTC, not by this session); liveness died with it and showed 12 stale ALERTs; restarted 05:50.
- Drivers/BIOS split: Reddie and Bluey on BIOS 5.36_0ACUM018 / driver 580.142 / kernel 1014; Spark4 and Asusi on
  GX10DGX.0104 / 580.159.03 / kernel 1021. Clock lock 2418 MHz on all four. (issue #1 data: 0ACUM027 + 580.173.02
  showed 0 slow seconds on 8 nodes.)
- Fresh idletest (05:49): count warm 116.9 tok/s at 49.6 ms/step, 5.85 tok/step; after idle 103.4 at 56.2 ms;
  code 83-94 at 53-57 ms, 5.0-5.1 tok/step. Traces show clean 49-51 ms runs with spikes to 62-90.

## Where the room is (from the 09-14 log, not to be re-run)
- DSpark k=5 caps tok/step at 5.85; counting is at 97% of that. Single-stream speed = step time.
- Dead, measured 09-14: k=4, k=10, batch-sharded sampling, b12x MXFP8 dense, 16K chunks (KV -42%), indexer logits
  cap, gmu 0.85 (KV +36% but 6 GB left and 8K prefill -28%), ENGRAM_THREADS=64, NCCL_PROTO=^LL128, NCCL_NTHREADS=256.
- Open: step spikes (Engram host sync, torch SwiGLU clamp 0.5-3 ms/step via cuda-exl3 rebuild, first-10-step code
  slowness); NCCL_BUFFSIZE (issue #8, KV lever); image limit (issue #6); gmu 0.82 opt-in (KV 4.11M at 300K).

## Harness (root on Reddie)
- `sr_final.sh <label>` full measurement -> `/var/tmp/boot-results/speedrun/<label>/`
- `sr_mkgo.sh` builds go scripts from the 300K base; tonight's variant `sr2_mkgo.sh` uses the 500K base.
- `sr_boot.sh <go> <label>` + `sr_screen.sh <label>` = one experiment cycle (~8.5 min boot + ~6 min screen).

## Log
- 05:52:58 baseline `s2-00-baseline` started (sr_final on the live lane).
- 05:55 harness: `/root/sr2_mkgo.sh <label>` (500K base) and `/root/sr2_run.sh <label>` (boot + screen + row in
  `/var/tmp/boot-results/speedrun/s2-table.txt`, status in `s2-status.txt`). Go scripts are `asusi:~/sr2-<label>-go.sh`.
- Agents out 05:53: (1) step profiling on the live lane, (2) external research since 09-01, (3) source audit of the
  running image for unpulled levers.
- Queue prepared (one line each, on the live base): e01-buffsize `NCCL_BUFFSIZE=1048576` (issue #8),
  e02-gmu82 `GMU=0.82`, e03-seqs16 `SEQS=16` (concurrency: capture sizes go to 96), e04-chunk64 `ENGRAM_CHUNK=64`
  (untested knob, default 16). Order and later entries wait on the baseline and the agents.

## Baseline s2-00-baseline (05:53-06:00 UTC, live 500K lane, full sr_final)
| | 09-14 final-best (300K) | s2-00-baseline (500K) | delta |
|---|---|---|---|
| KV pool | 3,757,748 | 4,295,367 | +14% |
| C1 / C3 / C6 agg | 58.3 / 128.3 / 190.0 | 58.3 / 110.8 / 184.8 | 0 / -14% / -3% |
| C1 code / json / math / prose | 85.2 / 65.0 / 87.9 / 41.0 | 90.9 / 68.0 / 85.0 / 39.5 | mixed |
| cold prefill 2K / 8K / 32K / 64K | 1,836 / 1,997 / 2,014 / 2,032 | **1,378 / 1,322 / 1,419 / 1,401** | **-25 to -34%** |
| idle count / code | 118.1 / 93.2 | 98.2 / 65.2 | -17% / -30% |
| vision+tools, quality | 7/7, PASS | 7/7, PASS | |
- All levers verified live on all 4 ranks at 06:01 (Engram node-local + FAST, indexer split, RoCE all-reduce and
  all-gather, ch8, expandable_segments:False). Software identical to 09-14.
- Prefill probe 06:02: 39,624 tokens at **1,664 tok/s** (09-14 same config: 2,124-2,146). GPU util 52%, 22 W (fast
  state), main worker thread 83-88% CPU. Not the GB10 slow state.
- The one variable left is MAXLEN 500K vs 300K; 09-14 never measured prefill at 500K.
- **06:02:57 e00-ctx300k launched:** the 09-14 final-best script unchanged. If prefill returns to ~2,000, 500K costs
  ~30% prefill and Tony gets that trade-off explicitly; if not, something on the box changed since 09-18's reboots.

## e00-ctx300k (06:03-06:16): THE 500K TAX, measured
The unchanged 09-14 final-best script at 300K brings every number back:
| | s2-00-baseline (500K) | e00-ctx300k (300K) |
|---|---|---|
| prefill 8K / 32K / probe 39.6K | 1,322 / 1,419 / 1,664 | **1,867 / 1,959 / 2,098** |
| C1 / C3 / C6 agg | 58.3 / 110.8 / 184.8 | 59.5 / 121.7 / 187.6 |
| C1 code / json / math / prose | 90.9 / 68.0 / 85.0 / 39.5 | 88.9 / 68.1 / 81.3 / 40.9 |
| idle count / code | 98.2 / 65.2 | 116.2 / 85.6 |
| KV pool | 4,295,367 | 3,782,286 |
**Serving at 500K costs ~30% prefill and 10-25% on multi-stream and idle decode.** Per-step work scaling with
max_model_len rather than live length. Agent dispatched 06:18 to find it in code (indexer buffers, block tables,
DSpark buffers, graph capture, scheduler). Fixing it = 500K at 300K speed, the single biggest lever tonight.
- Decision 06:18: **experiments run on the 300K base** (`exl3tp4b-ablit-best-go.sh`) so levers are measured without
  the tax; the 500K tax gets its own fix track. At 13:25 the served context is decided by whether the tax is fixed:
  fixed -> serve 500K (Tony's 09-14 choice); not fixed -> serve 500K anyway and put the 30% number in front of Tony,
  because context was his explicit call and speed-vs-context is his trade to make. (Revisit if he answers.)
- Queue regenerated on the 300K base 06:19: e05-roce32mb, e01-buffsize, e03-seqs16, e02-gmu82, e04-chunk64.
- Agents 06:12-06:18: (a) port vLLM #57432 (DSpark acceptance, +21% tok/step at TP4 upstream) and #56441 (KV-only
  DSpark insertion, +13-15% H100) into patch set `dsv41-exl3-sr3` with env toggles DSV41_DSPARK_ACCEPT_FIX /
  DSV41_DSPARK_KV_ONLY; (b) build image `vllm-dsv41:exl3b-roce-lim` = cuda-exl3 with the swiglu `limit` in-kernel
  (drops 80 clamp kernels/step; est. 0.3-3 ms/step, ~2% prefill; source is in the image, ~100 s build);
  (c) research: cuda-exl3 #4 fusion regression NOT in our build, indexer JIT fully pre-warmed (0 compiles/prefill),
  kho=off N/A (kernels 6.17), no firmware newer than 0ACUM027/580.173.02 for FE.
- 06:18 built `vllm-dsv41:exl3b-roce-lim` (113 s; schema `exl3_moe_glu_had_in(..., int block_m, float limit=0.)`,
  sm_121a cubin, clamp folded into the GLU lambda, NaN-propagating compares). Rolling to workers
  (`/root/roll_lim_image.sh`, log `roll-lim.log`). Experiment e06-exl3lim = 300K base + `IMAGE=...-lim`.
- 06:22 source audit (image + patch set) highlights: decode has ONE host round trip per step (the Engram hash sync,
  `engram.py:1464-1469`, gather cannot be prestaged because step N+1's ids come from step N's draft); the two Engram
  all-gathers per step go NCCL+movedim inside the FULL graph because they are dim=1 on a 3-D tensor (RoCE accepts dim
  0/last only, 0.3-0.5 ms/step); `DSV41_ENGRAM_DISK_CHUNK` is dead under FAST, THREADS is inert at decode (step=16
  regardless), only `DSV41_ENGRAM_FAST_GATHER_ROWS` moves decode gather cost; shared-expert overlap is gated to
  <=256 tokens (`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`); wo_a runs EmulationMxfp8 (bf16, 2x bytes, ~1.4 ms/step,
  medium-large code); FlashInfer sampler flag is inert on the spec path; async scheduling already on; boot = 237 s
  weight load (all 4 ranks read all ~233 GiB, NVMe-bound at 3.9 GiB/s) + 32-59 s draft re-load of 49 shards for 97
  params + 8 s autotune + 4 s capture, serving 6.6 min after container start; DeepGEMM/FlashInfer/Triton caches persist
  under EXP_NAME. Ranks 1-2 carry 69 GiB of weights vs 56.7 on 0/3 (uneven expert split) and size the KV pool.
- Queue 06:23: e05-roce32mb, e06-exl3lim, e07-gather64 (`DSV41_ENGRAM_FAST_GATHER_ROWS=64`), e08-sharedexp
  (`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=8192`), e01-buffsize, e03-seqs16. e04-chunk64 dropped (dead knob).
  `/root/sr2_dq.sh` runs it from 06:32 (`s2-queue.txt`, editable; `s2-queue.stop`; STOP_AT 12:50; two boot fails
  -> restore 300K best).
- Agent 06:23: patch set `dsv41-exl3-sr4` = sr2roce + Engram all-gather via RoCE (`DSV41_ENGRAM_ROCE_GATHER`) +
  draft-load shard filter (`DSV41_DRAFT_SHARD_FILTER`, -60 s boot).
- sr3 (vLLM #57432 / #56441 ports): expected FLAT on SM121 (the fixed code paths are SM100 TRTLLM; we run the
  FlashInfer SM120 decode) and #56441 needs a compiled op the image lacks. Parked; `lite` variant available.
- 06:31 max_model_len audit: **the code does not explain the 500K drop.** Prefill has no W-scaled work; decode has
  ~0.1-0.7 ms/step of indexer candidate-mask writes at 500K (`candidate_blocks.py:191-234`, 4 layers x rows x W x 4 B).
  **The KV pool number is a reporting artifact** (`kv_cache_utils.py:2306`: max_concurrency x max_model_len);
  actual KV is 29.97 GiB at 500K vs 30.14 GiB at 300K, same pages. Likely confound: page-cache state at measurement
  time (baseline was a 15 h old lane with flushers cycling Cached 3.7-23.9 GiB; e00 was a fresh boot).
  Optional low-risk code lever noted for later: bound the decode candidate kernels to the live width
  (`sparse_attn_indexer.py:~944`, `logits[:, :live_w]`), needs a 250K needle to verify.
- 06:32 dq started: e05-roce32mb booting. **e10-ctx500k inserted next** = the live 500K script, fresh boot + same
  screen. Decides whether the "tax" is real. If fresh-500K matches e00, serve 500K at the end with no penalty.

## Step profile (agent, 06:16-06:31 on the 300K lane; raw data `/var/tmp/agents-sr2/prof/` on Reddie)
- C1 step = GPU 49-52 ms + host bubble (max-rank host_pre) 1.0-1.2 ms floor, **5.2 p50 / 11.8 p90 on cold Engram
  rows**. GPU work is batch-independent for ~45 ms; +8.5 ms per added stream (C6 step 95.7 ms, 271.9 agg code).
- Spikes: ~90% are Engram cold-row page faults on ONE rank (Bluey straggler in 77% of cycles; per-fault latency
  Bluey 465 us, Reddie 500, Spark4 182, Asusi 205). The rest were `/root/liveness.sh` (16 tokens/60 s -> a PIECEWISE
  step + 4-5 C2-speed steps, 200-250 ms/min). **Not** clocks/power (2171-2190 MHz, 23-25 W C1 / 29-30 W C6).
- The `/etc/cron.d/drop-caches` cron on Reddie and Bluey (KaiClaude 2026-08-13) evicts Engram rows every 5 min: the
  same prompt goes cold again (Reddie host_pre 4.1 vs 1.1 ms). **System setting: Tony's call to remove.**
- First-10-steps "slowness" on code is the DSpark acceptance ramp (tok/step 4.0 -> 5.35), not ms/step. Per-category
  accept_len: count 4.8-5.0, code 4.0-4.3, format 4.0, math 3.3, json 3.3, reasoning 3.0, summary 1.2, prose 1.0.
- Levers ranked by the data: (1) hide Engram cold faults, -3 to -6 ms/step at C1 (+8-10%): `DSV41_ENGRAM_FAST_GATHER_ROWS=4`
  (smaller parallel fault chains), cron removal, a prefetch patch; (2) lighter liveness probe; (3) Bluey NVMe fault
  latency; (4) trimming the 294 eager launches/step only pays once GPU work shrinks.
- Actions 06:35: liveness probe changed to 2 tokens every 120 s (`/root/liveness.sh`, backup `.pre-0919`);
  **e07 is now `gather4`** (was gather64, which the data says goes the wrong way).

## Experiment log (SCREEN = C1/C3/C6 + 8K/32K cold prefill + idle test + quality; base e00 = 300K best)
| # | change | KV | C1 agg | C3 agg | C6 agg | C1 code | 8K | 32K | idle count / code | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| e00-ctx300k | 09-14 best, 300K | 3,782,286 | 59.5 | 121.7 | 187.6 | 88.9 | 1,867 | 1,959 | 116.2 / 85.6 | REFERENCE |
| e05-roce32mb | RoCE all-reduce cap 32 MB (took effect: max=33554432) | 3,719,280 | 60.0 | 124.8 | 191.7 | 90.6 | 1,704 | 1,972 | 115.8 / 84.9 | **DROP**: decode within noise, 8K -9%, KV -1.7% |
| e10-ctx500k | live 500K script, FRESH boot | 4,172,017 | 56.1 | 124.1 | 190.1 | 85.9 | 1,869 | 1,963 | 114.7 / 85.1 | **500K costs nothing fresh.** The "tax" was lane AGE: the 15 h old lane ran -30% prefill / -20% idle decode |
- Decision 07:00: **final config serves 500K** (Tony's choice, now shown to cost nothing). New primary finding: lanes
  degrade with age. Mechanism per the profiler: `drop_caches` cron evicting Engram rows. Aging test queued
  (`/root/s2_agingtest.sh`: probe, drop caches on all four like the cron, probe, probe) right after e06; dq paused
  via `s2-queue.stop` for it, then restarted.
| e06-exl3lim | image `exl3b-roce-lim` (swiglu clamp in-kernel; fallback warning gone) | 3,729,968 | **61.3** | 122.3 | **195.2** | 90.2 | **1,922** | **2,017** | 116.7 / 83.6 | **KEEP, NEW BASE**: +3-5% decode, +3% prefill, math 88.4 (+9%), quality PASS |
- 07:12 queue restacked on e06 (`BASE=sr2-e06-exl3lim-go.sh`): e09-sr4, e07-gather4, e08-sharedexp, e01-buffsize,
  e03-seqs16. Aging test running on the e06 lane first (background task).
- 07:15 aging test on the fresh e06 lane: probe 39.6K = 2,226 tok/s; `echo 1 > drop_caches` on all four (what the
  cron does) -> 2,226 and 2,223 again. **Page cache does not reproduce the aging effect.** The 15 h old lane's
  1,664 tok/s and 98/65 idle numbers remain unexplained (candidates: allocator state, prefix-cache churn, NFS client
  state). Not reproducible in minutes. Operational note for Tony: a fresh boot recovers ~30% prefill; consider a
  scheduled restart until the cause is found. Idle after the test: count 118.1-118.3, code 94.7-95.0 (best tonight).
- 07:15 dq restarted: e09-sr4 booting.
| e09-sr4 | e06 + patch set sr4 (Engram gather via RoCE, draft reads 3/49 shards) | 3,725,682 | 62.0 | 121.5 | 195.0 | 92.2 | 1,917 | 2,021 | 117.9 / 84.7 | **KEEP, NEW BASE**: boot 441 s vs 504 s (-63 s); decode flat as predicted |
- 07:28 e07-gather4 booting (on the e06 script; A/B vs e06). e08, e01, e03 regenerated on `sr2-e09-sr4-go.sh`.
| e07-gather4 | e06 + `DSV41_ENGRAM_FAST_GATHER_ROWS=4` | 3,768,739 | 61.0 | 120.1 | 189.6 | 90.6 | **1,636** | **1,700** | 116.0 / 89.1 | **DROP**: prefill -15/-16% (tiny tasks shred the 8K-row prefill gathers); idle code +7% only |
- Queue after e03: e11-final500k (e09 stack at 500K = the final candidate), e02-gmu82 (e11 + GMU=0.82).
| e08-sharedexp | e09 + `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=8192` | 3,814,599 | 61.8 | 124.0 | 201.3 | 91.6 | 1,884 | 1,984 | 117.7 / 84.5 | **DROP**: prefill -2% (its target); decode within noise |
| e01-buffsize | e09 + `NCCL_BUFFSIZE=1048576` (issue #8) | 3,726,094 | 59.5 | 127.4 | 198.4 | 95.8 | 1,920 | 2,036 | 117.8 / 86.1 | **NEUTRAL, drop**: KV unchanged (+412 tokens); decode all-reduce is on RoCE here, so NCCL buffers have nothing to give back. Post on #8 |
| e03-seqs16 | e09 + `SEQS=16` (30 capture sizes to 96, graphs 1.27 GiB) | 3,748,406 | 58.8 | 130.4 | 190.9 | 91.0 | **1,644** | 2,015 | 115.1 / 84.9 | **CONCURRENCY WIN, 8K prefill to re-check**: C8 coding 407.8 agg (55.6/stream), C8 count 510.1; C12 coding 494.7 (45.3/stream), math 422.8, format 424.9, count 622.9. C6 coding on the baseline was 306 |
- 08:20 e11-final500k booting (e09 stack at 500K, SEQS=8). Then e02-gmu82, then e12-final500k-s16 (e11 + SEQS=16).
  The final full measurement goes to the better of e11/e12 on the 8K prefill and C1 columns.
| e11-final500k | e09 stack at MAXLEN=500000 (final candidate, SEQS=8) | 4,224,689 | 60.8 | 120.7 | 197.5 | 93.6 | 1,923 | 2,017 | 116.2 / 86.0 | **FINAL CANDIDATE**: equals e09 at 300K within noise; vs the aged 500K baseline +4/+9/+7% agg, prefill +45%, idle +18/+32% |
| e02-gmu82 | e11 + `GMU=0.82` | **4,775,278** (+13%) | 59.8 | 123.6 | 188.0 | 88.5 | 1,908 | 2,002 | 114.7 / 85.3 | **OPT-IN, not default**: decode -1 to -8% (noise band, same direction), prefill -1%; MemAvailable Spark4/Asusi 15 GB (bar met, was 10-11 on 09-14). One line: `export GMU=0.82` |
| e12-final500k-s16 | e11 + `SEQS=16` | 4,011,085 | 59.9 | 125.8 | 190.2 | 90.7 | 1,878 | 2,007 | 115.7 / 84.7 | **FINAL**: e03's 8K -14% did not reproduce (-2%); KV -5% for 15 more graphs; 12+ streams served |

## FINAL = e12-final500k-s16 (chosen 08:57 UTC)
`asusi:~/sr2-e12-final500k-s16-go.sh` = 09-14 best + image `vllm-dsv41:exl3b-roce-lim` (swiglu clamp in-kernel) +
patch set `dsv41-exl3-sr4` (Engram all-gather via RoCE, draft loader reads 3/49 shards) + MAXLEN 500000 + SEQS 16.
gmu 0.80 (0.82 documented as the context opt-in: KV 4,775,278, 15 GB free on the tight nodes).
- 08:57 full measurement running (`/root/s2_final_measure.sh`): sr_final vs s2-00-baseline, needles 65K/131K, C8/C12.
- Plan for the remaining hours: leave the final up and re-measure prefill at ~12:30 (3.5 h old) for the aging
  question; post the issue #8 answer; write README; final health check before 13:25. dq stopped (queue empty).
