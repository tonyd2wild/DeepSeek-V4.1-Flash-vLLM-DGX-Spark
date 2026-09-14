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

## Experiment log
| # | change | boot | KV | C1 agg | C3 agg | C6 agg | code C1 | prefill 32K | verdict |
|---|---|---|---|---|---|---|---|---|---|

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
