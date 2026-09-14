# Speed run 2026-09-14: STATE (read this first after any context compaction)

Owner: Tony (asleep). Operator: Claude (this session). This file is the source of truth for where the run is.

## Deadlines (Tony is on ET = UTC-4)
- **Until 11:00 UTC (7:00 AM ET):** speed work. Baseline, sub-agent research, experiments (one change per boot).
- **11:00-12:00 UTC (7-8 AM ET):** answer every open issue and PR. Merge what makes sense and decline what doesn't, keeping it quick. Hold off on vLLM-only items that don't apply to the EXL3 lane.
- **By 12:00 UTC (8:00 AM ET):** DeepSeek V4.1 Flash UNCENSORED (EXL3-Pollard-Abliterated) serving on the BEST config found, verified, and this folder complete and pushed.

## Rules (Tony's house rules plus tonight's goal)
- One change per boot. Two failed boots in a row = stop experimenting and restore the last known-good config.
- Nothing deleted on any box. No system settings (clock locks, sysctl, cron, firmware, fstab).
- Measured numbers only. Keep a monitor on every boot.
- Tailscale SSH may ask for a login check; if it does, the fleet is unreachable until Tony approves. Remote jobs keep running and log on the Sparks.
- Known-good restore of the serving config: `bash /root/restore_exl3tp4b_ablit.sh` on Reddie (EXL3-Pollard-Abliterated, TP4, go script `~/exl3tp4b-ablit-go.sh` on Asusi).

## Baseline config (as found at 06:00 UTC)
- Model `/var/tmp/models/DeepSeek-V4.1-Flash-EXL3-Pollard-Abliterated`, image `vllm-dsv41:exl3a`, patch set `dsv41-exl3-tp3e`.
- TP4 (head Reddie), DSpark k=5 (probabilistic draft, block rejection), CUDA graphs FULL_AND_PIECEWISE.
- max_num_batched_tokens 8192, max_num_seqs 8, gmu 0.80, 300K, block 128, vision (4 images), tools, thinking off.
- Node-local Engram rows. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- KV pool 3,274,912 tokens (10.92x at 300K); model 56.61 GiB on the head.
- GPU clocks reported 2411 MHz idle on three nodes (lock not active); left as Tony set them.

## Phases
- [x] 0. Found state, pulled repo, extracted engine source to scratchpad `src/` (vllm, cuda_exl3, flashinfer, patch set).
- [x] 1. Baseline: idle test plus full C1-C6 plus prefill 2K-64K → `results/00-baseline/` (remote `/var/tmp/boot-results/speedrun/00-baseline`). Done 06:14 UTC.
  - Aggregate tok/s: C1 56.3, C2 86.9, C3 114.7, C4 136.3, C5 156.6, C6 167.7. Per-stream C1 63.2, TTFT 0.28-0.46 s.
  - Per-stream C1: code 84.0, JSON 75.4, math 78.5, tables 91.1, reasoning 70.2, prose 37.0, narrative 34.9, summary 34.5, counting 98.3.
  - Prefill tok/s: 2,950 tok → 1,369; 11,592 → 1,443; 46,810 → 1,454; 93,335 → 1,468. Flat per token, so a per-token cost dominates.
  - KV pool 3,274,912 tokens (10.92x at 300K).
  - SCREEN-protocol baseline (`00b-baseline-screen`) runs after it.
- Web agent: prefill is the biggest gap. The SGLang/MiaAI recipe reports 3,350-3,768 tok/s on the same 4 Sparks. Their NCCL env (`NCCL_PROTO=^LL128 NCCL_MAX_NCHANNELS=8 NCCL_BUFFSIZE=1048576 NCCL_LL128_BUFFSIZE=262144`) cut pinned memory from 4.7 to 0.14 GB per node. Other levers: the b12x dense FP8 kernel (vs EmulationMxfp8), `--long-prefill-token-threshold`, `CUDA_EXL3_MOE_BLOCK_M` for C6.
- [ ] 2. Sub-agents: web levers (running); vLLM engine; EXL3 MoE kernels; prefill/TTFT; DSpark.
- [ ] 3. Experiments (SCREEN protocol below), ranked by expected gain.
- [ ] 4. Full C1-C6 on the best candidate.
- [ ] 5. 11:00 UTC: issues and PRs.
- [ ] 6. 12:00 UTC: final config serving (uncensored), postcheck, README of this folder, push.

## SCREEN protocol (per experiment boot)
1. Boot via `sr_boot.sh <go-script>` (Reddie) and wait for serving.
2. Warm-up: `idletest.py`, then one C1 pass of all categories (discarded).
3. Screen: `v41bench.py --levels 1,3,6 --prefill 8000,32000` → `results/NN-name/`.
4. Compare with the baseline's own SCREEN run (same protocol) using `bench_report.py --vs`.

## Measured levers from bot-lab-21's HF card (DSV4.1 EXL3 on 4 Sparks, recipe built on our repo)
- `--async-scheduling`: per-stream C1 62.2→68.0 (+9%), C6 −3%.
- `NCCL_MAX_NCHANNELS=8`: C6 178.6→199.0 (+11%); prefill at 47K 1323→1423.
- Force the b12x MXFP8 dense kernel via `VLLM_DISABLED_KERNELS=FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel,MarlinMxfp8LinearKernel`: C6 +3.7%.
- b12x RoCE one-shot all-reduce (`ROCE=1`, 2 MB): C1 +5.5%. Needs the b12x package (checking).
- Rejected on their ladder: `NCCL_PROTO=^LL128` and the BUFFSIZE settings, `ENGRAM_THREADS=64`, resident Engram rows, `NCCL_NTHREADS=256`.
- The EXL3 MoE path is near the bandwidth floor. The only waste is the torch clamp fallback (0.5-3 ms/step), fixed by a kernel rebuild (optional, later).

## Queue (one change per boot, stacked on the best so far)
1. E01 `NCCL_MAX_NCHANNELS=8` (`sr-e01-nccl-ch8-go.sh`). Queued to boot after the probe (task bk49942qq).
2. E02 + `--async-scheduling`: `sr-e02a-ch8-async-go.sh` if E01 wins, else `sr-e02b-async-go.sh`.
3. E03 + b12x MXFP8 dense via VLLM_DISABLED_KERNELS (if b12x can be picked).
4. Prefill levers (pending the prefill agent and probe): long-prefill threshold, shared-expert stream threshold, max_num_batched_tokens.
5. DSpark k (pending the DSpark agent).

## Credits (Tony: credit everyone whose work we use; carry this into the run README and the main README)
- **bot-lab-21:** EXL3 3.5 bpw Pollard checkpoint and HF tuning ladder: NCCL_MAX_NCHANNELS=8, async scheduling, b12x DISABLED_KERNELS, the clamped-SwiGLU exl3_moe.py overlay.
- **Zeuss5/cuda-exl3** (and collaborator @NNNtrance): the EXL3 CUDA plugin and its perf studies.
- **MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks** (SGLang recipe): NCCL settings, the b12x MXFP8 idea, prefill numbers.
- **im0xMagnus:** PRs #3-#5 (vLLM commit pin, diffs, verify5), the 8x-Spark fork, issue #1 data.
- **ecohash-co and hyudryu:** the issue #2 bisect (expandable_segments finding).
- **tmolteno:** issue #1 data.
- **antheas/spark_hwmon:** a lead, not used.
- **vLLM team:** dsv41-feat. **Kai:** the original Engram-on-disk and SM12x page patches.

## Tony's direction 06:21 UTC
- SGLang is allowed if it's better ("vLLM or SGLang, do whatever is BEST").
- Caveat: the uncensored model is EXL3 (vLLM plugin). Tonight: port SGLang's ideas into vLLM; write up the SGLang option with numbers.

## Agent findings (all in)
- **vLLM engine:** async scheduling is auto-on (skip E02-async). `--enable-batch-sharded-sampling` is small at C2+. wo_a runs a BF16 bmm via Emulation (2x bytes; code patch). The Engram host sync cuts the async overlap every step.
- **Prefill:**
  - The indexer is replicated on every rank; splitting it is a big code patch (93K 78→52 s est.).
  - Engram staging is 5-20% of prefill by estimate. The probe shows GPU util only 54% during a 39.6K prefill (1,671 tok/s), so host work is bigger than that estimate.
  - `DSV41_ENGRAM_DISK_CHUNK=256` is restart-only.
  - Prefix caching is ON (hits on 128-token boundaries), which helps multi-turn TTFT.
  - Capture sizes >48 would help short-prompt TTFT, but padding risks #5015.
- **DSpark:** k=6-9 are rejected; k=10 is allowed (a gamble: code could gain a lot, prose loses 10-20%). At temp 0 the sampling method is irrelevant; adaptive verification is blocked.
- **b12x:** installed as image `vllm-dsv41:exl3b` (exl3a + b12x 1.3.0) on Spark4, Asusi and Bluey; Reddie is rebuilding. The CUDA MXFP8 list order is Cutedsl, Cutlass, Marlin, B12x, so disabling the first three picks B12x.
- **Engram fast path (my patch, set `dsv41-exl3-sr1`, env `DSV41_ENGRAM_FAST=1`):** numpy memmap gather (no per-row Python preadv loop, threads without the GIL), raw fp8 rows to the GPU, dequant on the GPU (bit-exact math), no bf16 pinned copy.

## Engram fast path: offline test on Bluey (CPU, node-local rows, layer 1 rank 3)
- Bit-identical to the preadv path at 288 and 98,304 rows (random rows, duplicates, unowned rows): PASS in 4 runs.
- Timing, CPU dequant in the test (the patch dequantizes on the GPU):

  | gather rows per task | 288 rows cold / warm | 98,304 rows cold / warm |
  |---|---|---|
  | 16 | 6.4 / 1.8 ms | 618 / 79 ms |
  | 128 | 42.5 / 1.2 ms | 372 / 27 ms |
  | 1024 | 46.8 / 0.9 ms | **311 / 17 ms** |
  | old preadv path | 4.2 / 3.5-6.1 ms | 589-655 / 355-486 ms |

- Final patch: auto rows per task = ceil(n/32) clamped to [16, 1024]. `engram.py` md5 is in the chain log. Staged in `~/patches/dsv41-exl3-sr1` on all 4 nodes.

## Decisions
- **cuda-exl3 clamp fix: SKIPPED.**
  - `had128_warp_glu_in` reads gate/up from global memory and lives in `exl3_had.cuh`, which isn't in the installed package.
  - A fused clamp without editing that header saves only 2 graph launches per layer (~0.1-0.2 ms/step), not worth a 4-node rebuild.
- **Plan after E02:** E03, E04 and E05 each add ONE change on top of E02 (clean A/B vs a common base):
  - E03: b12x image with MXFP8 forced to b12x;
  - E04: `--enable-batch-sharded-sampling`;
  - E05: `SPEC_K=10`.
  - The final config is E02 plus the winners, validated with a full C1-C6 bench on a final boot.
- **Quality gate** (`sr_quality.py`, inside `sr_screen.sh` from E02 on): count 1..100, JSON keys, runnable `is_prime` code, 17*23, prose not degenerate. A config that fails the gate cannot win.

## SGLang option (agent memo, 06:40 UTC): stay on vLLM-EXL3 tonight
- **SGLang has no EXL3 support** (no quant method; cuda-exl3 is vLLM-only). Switching means a different uncensored checkpoint, e.g. dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8, on an untested stack. That's high bring-up risk tonight.
- **MiaAI TP4 measured** (prose only):
  - aggregate C1 45.4, C4 103.1, C8 114.1;
  - TTFT 0.21-0.27 s;
  - **prefill 3,350-3,782 tok/s**;
  - 1M context with KV pinned at 8M.
- **ecohash (SGLang TP4 on GB10):** 60 tok/s C1, 138 at C4.
- **Ours (E01):** C1 57.4 aggregate (code 85, prose 37), C6 184, prefill ~1,400-1,500, KV 3.5M at 300K.
- **Verdict:** SGLang wins prefill (2.4x), TTFT and context; decode is roughly equal (prose C1 +20% for SGLang; we lead C6). A later half-day window could try MiaAI `68b34cfd0` with the dealignai checkpoint. Steps are in the memo (scratchpad `mia/`).
- **Ideas ported tonight:**
  - NCCL ch8 (E01, done);
  - b12x MXFP8 (E03);
  - `expandable_segments:False` (E08; ecohash saw 2.45x KV on vLLM);
  - MiaAI's smaller prefill chunks (1,024) as a possible later test.

## PR pre-checks (done early, to post at 11:00 UTC)
- **PR #4:** applies cleanly to main. Its `patch/verify_diffs.sh`, run against a local vLLM checkout at `e47aa780b` (`scratchpad/vllm-e47`), passes 7/7 full diffs and 4/4 per-fix chains. Merge at 11:00.
- **PR #5:** `verify5.py` compiles, and the logic matches hyudryu's report. It includes a stray `build/__pycache__/verify5.cpython-314.pyc`: merge it, then remove the pyc and add `__pycache__/` to `.gitignore` in a follow-up commit.
- Drafts are in `DRAFT-issue-pr-replies.md`. Re-read every thread for new comments before posting.

## Queues running on Reddie (detached)
- **queue-1** (`queue-1.log`): after E02 → E03 `e03-b12x`, E04 `e04-bss`, E05 `e05-k10`. Falls back to `e01-nccl-ch8` if E02 fails boot or quality.
- **queue-2** (`queue-2.log`, NOBASE=1): after E05 → E06 `e06-mb16k` (MAX_BATCHED 16384), E07 `e07-shexp` (VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=8192), E08 `e08-noexpseg` (expandable_segments:False). All three stack on E02. Fallback is `e02-ch8-fast`.
- Expected timeline, about 20 min each: E03 ~06:52, E04 ~07:12, E05 ~07:32, E06 ~07:52, E07 ~08:12, E08 ~08:32.
- Then: the combination of winners plus a full C1-C6 run by ~09:30.
- **To stop a queue:** `touch /var/tmp/boot-results/speedrun/queue.stop` (checked before each label), or `pkill -f "sr_queue.sh"`. Do NOT kill a running `sr_run.sh` mid-boot.

## Runner notes
- Chains run DETACHED on Reddie (`setsid nohup`), so they survive losing the Tailscale connection.
- `/root/sr_chain_e02.sh`: waits for E01, then runs E02 (`sr-e02-ch8-fast-go.sh`: ch8 plus `PATCH_NAME=dsv41-exl3-sr1` plus `DSV41_ENGRAM_FAST=1`), and runs the offline test (auto chunk) during E02's load → `test-fast-auto.txt`.
- Status files: `/var/tmp/boot-results/speedrun/run-<label>.status`, `chain-e02.log`.

## Experiment log
| # | change | boot | KV | C1 agg | C3 agg | C6 agg | code C1 | prefill 32K | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 00b | baseline SCREEN (as found, warm) | - | 3,274,912 | 54.3 | 114.8 | 166.3 | 80.6 | 1,454 (full-bench cold) | reference |
| E02 | E01 + Engram FAST staging (`dsv41-exl3-sr1`, `DSV41_ENGRAM_FAST=1`) | 534 s, OK ("Engram FAST staging on" logged) | pending | | | | | | screen running 06:40 |
| E01 | `NCCL_MAX_NCHANNELS=8` | 8.7 min, OK | 3,512,346 (+7.3%) | 57.4 (+5.8%) | 117.3 (+2.2%) | 184.2 (+10.7%) | 85.4 (+5.9%) | 1,486 (+2%) | **KEEP** (bot-lab-21 saw +11% C6) |

## Findings so far
- **Baseline idle test** (06:07 UTC), after the first token:
  - count: 101.0 (after idle) / 106.4 / 106.5 tok/s, 54-58 ms/step, 5.85 tok/step;
  - code: 81.1 / 88.8 tok/s, 57-60 ms/step.
  - Much faster than boot 10 (63 ms). This is EXL3, with clocks unlocked at 2411 MHz.
- **Head log, cuda-exl3:** "compiled exl3_moe_glu_had_in has no `limit` argument; swiglu_limit models will clamp in torch before the kernel (slower). Rebuild the extension." This is a kernel lever (rebuild cuda-exl3 with limit support).
- **Head log, other kernels:**
  - "Using EmulationMxfp8LinearKernel for MXFP8 GEMM" (once) and "FlashInferCutlassMxfp8LinearKernel" (once).
  - Drafter MoE on DEEPGEMM_MXFP4.
  - All-reduce is PYNCCL only.
- **Harness deployed on Reddie:** `/root/sr_boot.sh <go> <label>` and `/root/sr_screen.sh <label>`. Local copies are in `tools/`.
- **Mac:** keep-awake requested (session_idle).

## Next action
Wait for the baseline bench (background task), then run SCREEN on the baseline; meanwhile read the sub-agent reports and build the experiment queue.
