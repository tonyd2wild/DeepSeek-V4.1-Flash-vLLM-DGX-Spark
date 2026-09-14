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

## Read-pool size test (offline, Bluey, 06:48 UTC): the fast path scales with threads
Fast path, prefill-sized gather of 98,304 rows (cold / warm), fresh rows each run:

| threads | cold | warm |
|---|---|---|
| 32 | 387 ms | 24 ms |
| 64 | 194 ms | 26 ms |
| **128** | **105 ms** | 26 ms |

- The old preadv path was 604-634 ms cold at any thread count.
- Decode-sized gather (288 rows): 4.9-7.6 ms cold, 1.6-2.2 ms warm at every thread count.
- Bit-identical in all runs.
- Hence **E09** = E02 + `ENGRAM_THREADS=128` (`sr-e09-thr128-go.sh`), moved up to run right after E03.

## E03 check
The container runs `vllm-dsv41:exl3b` and logs "Using B12xMxfp8LinearKernel for MXFP8 GEMM", with the `VLLM_DISABLED_KERNELS` set in place. The experiment is valid.

## Queues running on Reddie (detached)
- **06:49 reorder:**
  - queue-1 was stopped. Its E03 run continued as `sr_run.sh e03-b12x`.
  - **queue-1b** (`queue-1b.log`, NOBASE=1): after E03, runs E09 `e09-thr128`, then E04 `e04-bss`, then E05 `e05-k10`.
  - queue-2 is unchanged: it waits for E05, then runs E06, E07, E08.
- **queue-3** (`queue-3.log`, NOBASE=1): after E08, runs E10 `e10-idxlogits` (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1024`, prefill sub-chunks).
- Chain: E03 → E09 → E04 → E05 → E06 → E07 → E08 → E10, about 20 min each, ending ~09:20 UTC. After that: combine the winners (one boot), then the full C1-C6 run (`sr_final.sh`).
- **Gotcha:** never `pgrep -f <pattern>` and `kill` from an ssh `bash -c` whose own command line contains the pattern: it kills your own shell. Use a script file (`/root/start_q1b.sh` does this).
- **queue-1** (`queue-1.log`): after E02 → E03 `e03-b12x`, E04 `e04-bss`, E05 `e05-k10`. Falls back to `e01-nccl-ch8` if E02 fails boot or quality.
- **queue-2** (`queue-2.log`, NOBASE=1): after E05 → E06 `e06-mb16k` (MAX_BATCHED 16384), E07 `e07-shexp` (VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=8192), E08 `e08-noexpseg` (expandable_segments:False). All three stack on E02. Fallback is `e02-ch8-fast`.
- Expected timeline, about 20 min each: E03 ~06:52, E04 ~07:12, E05 ~07:32, E06 ~07:52, E07 ~08:12, E08 ~08:32.
- Then: the combination of winners plus a full C1-C6 run by ~09:30.
- **07:05 switch to a DYNAMIC queue.** Queues 2, 3 and 4 (still waiting) were killed by `/root/switch_to_dq.sh`. `/root/sr_dq.sh e05-k10 e02-ch8-fast` (`dq.log`, started by `/root/start_dq.sh`) waits for E05, then pops labels one at a time from `/var/tmp/boot-results/speedrun/queue.txt`. **To reorder, insert or drop: write `queue.txt.new` and `mv` it over `queue.txt`.** Done labels go to `queue.done`. Two consecutive BOOT-FAILs boot the fallback `e02-ch8-fast` and stop.
  - queue-1b (E09 running, then E04, E05) is unchanged.
  - Order at 07:05: e12-ch4, e14-idxsplit (drop it if the prehook test fails), e06-mb16k, e10-idxlogits, e15-tree, e11-ctx1m, e07-shexp, e08-noexpseg, e13-gmu85, e16-ch2.
  - New go scripts: `sr-e14-idxsplit-go.sh` (patch set `dsv41-exl3-sr2` = sr1 + the split indexer, md5 f87bc894, on all 4 nodes; `DSV41_INDEXER_TP_SPLIT=1`), `sr-e15-tree-go.sh` (`NCCL_ALGO=Tree`), `sr-e16-ch2-go.sh` (`NCCL_MAX_NCHANNELS=2`).
  - Prehook armed before E04: the idxsplit exactness test on Bluey (`test-idxsplit.txt`).
- **Gotcha 2:** `ps -eo args | grep -q <pattern>` from an ssh command whose own text has the pattern matches the tailscaled login shell. Anchor it (`grep -q "^bash /root/sr_dq\.sh"`) and run checks from script files.
- **(superseded) queue-4** (`queue-4.log`, NOBASE=1, started 06:57 by `/root/start_q4.sh`): after E10, runs E11 `e11-ctx1m` (MAXLEN=1048576; the model's native max_position_embeddings is 1,048,576 with yarn x16), E12 `e12-ch4` (NCCL_MAX_NCHANNELS=4), E13 `e13-gmu85` (GMU 0.85; 28-29 GB free per node while serving at 0.80). All stack on E02. Fallback `e02-ch8-fast`.
- **One-shot prehook** (`sr_run.sh` runs `/var/tmp/boot-results/speedrun/prehook.sh` once before the next boot, while the old server is idle): the wo_a standalone GPU test on Bluey (skips if under 20 GB free) → `test-woa.txt`, then renamed `prehook.sh.done-before-<label>`.
- **wo_a native MXFP8 test (06:59, Bluey, prehook): DROP.** Numerics PASS (relA 0.027, equal to plain FP8 quant-dequant error), graph replay PASS. The native path is slower at decode sizes: under CUDA graphs 142-150 us at T=1-24 vs 25-33 us for today's BF16 bmm; equal only near T=2048. Full output in `results/test-woa.txt`.
- **Indexer prefill TP-split draft** (agent, `scratchpad/idxsplit/`): one file, `sparse_attn_indexer.py`, flag `DSV41_INDEXER_TP_SPLIT=1`. Projected prefill +27% at 47K and +48% at 93K (estimate). Needs the GPU exactness test before any boot.
- **NCCL research agent (07:03):** ranks `NCCL_MAX_NCHANNELS=4` then `2` (NVIDIA NCCL tuning blog), `NCCL_ALGO=Tree` (UMD PSSG arXiv 2511.09557), `NCCL_PROTO=LL` diagnostic only, a size-banded tuner plugin, and `NCCL_BUFFSIZE=2097152`. Custom/symm-mem all-reduce is single-node only in vLLM. E12 (ch4), E15 (Tree) and E16 (ch2) come from this.
- **Model-folder `TUNING_BACKLOG.md` leads (07:08):** b12x RoCE one-shot all-reduce (b12x `docs/rocenante.md`, local-inference-lab/vllm#597: 48 KB AR 65.5 → 23.6 us on 4 Sparks; b12x 1.3.0 in exl3b has only `comm/pcie`, so this needs a newer b12x plus a vLLM integration); draft KV group page size 256 (Zeuss5/cuda-exl3 #2, NNNtrance: KV +82%, TTFT -20-30%); `CUDA_EXL3_MOE_BLOCK_M=16/32`; `NCCL_NTHREADS=256`. Two agents are researching the RoCE port and the page-size patch (`scratchpad/roce/`, `scratchpad/draftpage/`).
- **E19 b12x RoCE one-shot all-reduce (agent report 07:24; files in `scratchpad/roce/`, copied to `/root/roce`):**
  - No PyPI b12x has `comm.roce`. `Dockerfile.roce` overlays `b12x/comm/roce` from local-inference-lab/b12x at `b58f34ea` onto exl3a (every other b12x kernel stays 1.3.0) and builds its RDMA proxy .so, giving `vllm-dsv41:exl3a-roce`.
  - vLLM side: local-inference-lab/vllm#597 ported onto our tree as 5 files (new `b12x_roce_all_reduce.py`, plus `cuda_communicator.py`, `parallel_state.py`, `envs.py`, `gpu_worker.py`). The 4 base files match the exl3a image md5 for md5. Patch set `dsv41-exl3-sr1roce` = sr1 + those 5 (18 mounts) on all 4 nodes.
  - `sr-e19-roce-go.sh`: IMAGE exl3a-roce, sr1roce, E02 env plus `VLLM_ENABLE_ROCE_ALLREDUCE=1`, max 2MB all-reduce / 16MB all-gather, HCA rocep1s0f0, GID 3, spin limit 300M (about 5 min, so boot-time JIT skew cannot trip it). Rollback: `VLLM_ENABLE_ROCE_ALLREDUCE=0`.
  - `/root/build_roce.sh` (detached, `build-roce.log`): waits for E05's boot to start, builds on all 4 nodes, and on 4/4 success puts `e19-roce` first in `queue.txt`.
  - **07:30 build try 1 failed on all 4 nodes:** `exl3a` has no b12x at all (`ModuleNotFoundError`); b12x 1.3.0 is only in `exl3b`. **07:31 rebuild** with `BASE=vllm-dsv41:exl3b`, tag `vllm-dsv41:exl3b-roce`; the E19 go script now uses `exl3b-roce`. Confound to check in E19's boot log: with b12x importable, a layer that fell to `EmulationMxfp8LinearKernel` on exl3a may now pick B12x (E03 forced B12x on every MXFP8 layer and lost). Try-1 logs kept as `build-roce-*.try1.log`. **Rebuild OK on 4/4 at 07:31:52** (proxy .so `roce_proxy-a35f54cf75d6abf2.so` loads, `roce API_VERSION 1`); `e19-roce` auto-queued first. A background watchdog reports E19's kernel/RoCE log lines and flags a stall if the screen runs over 15 min after BOOT-OK.
  - Upstream numbers (b12x `docs/rocenante.md`): 48 KB all-reduce 23.6 vs 65.5 us NCCL, 256 KB 58 vs 174 us. Estimate for us: 2.5-3.5 ms of a ~50 ms step, about +5-7% at C1.
  - Risks: b12x#313 (one rank stuck while /health stays 200), so watch the E19 screen for stalls; custom all-reduce must stay enabled or RoCE turns off silently; 160 MB pinned per rank. Check the boot log for `B12X_ROCENANTE`.
  - Credits: Jason @original-el8 (b12x#295, #315, vllm#597), Luke Alonso @lukealonso (aa90a277, 1a7e3ec2), @tobymao (b12x#313 hang report).
- **Draft KV page 256 (NNNtrance, Zeuss5/cuda-exl3#2): not queued.** On V4.1 the 3 DSpark draft layers (40-42) are pure sliding window and share the target's SWA groups (64-token page); there is no 16-token draft group, and `_largest_kernel_block_within` is never called. The SM120 FlashInfer SWA backend accepts only 64-token pages, so 256 would fail at KV allocation. Our offline pool model: removing the drafter entirely is only +1.5% pool; page 256 would be -69%. Lead for later: layer 20's ratio-1 caches take ~52% of per-request blocks; 128-token pages there model at +59% pool, but the same 64-token kernel limit blocks it. `max_num_batched_tokens 4096` models at +12% pool (costs prefill). Files in `scratchpad/draftpage/`.
- **Drafter skip on non-final prefill chunks: not queued.** The agent found it safe only as a partial skip (DSpark's 3 draft layers are 128-token sliding window, so the last 512 prompt rows must keep drafter KV). The drafter costs about 1.7 TFLOP per 8192-token chunk, roughly 15-30 ms of a ~4.7 s chunk (0.3-0.6%), inside noise. Draft kept in `scratchpad/drafterskip/`.
- **idxsplit GPU test (07:13-07:14, Bluey, prehook; `results/test-idxsplit.txt`):**
  - First-chunk ratio-1 fp8 case, all three candidate modes, TP 4/3/2 simulated: split output SET-EQUAL to the unsplit reference, which itself varies in order between runs; candidates BYTE-EQUAL.
  - Real 4-process spawn (gloo all-gather, the patch's full split path): SET-EQUAL, and all ranks hold identical bytes.
  - Ties mode: differences only among exactly tied keys (the reference differs from itself too).
  - The "late-chunk r2" case crashes inside the TEST's unsplit reference call (DeepGEMM `t.dim() == N`), a test-harness input-shape bug, so ratio-2 is not covered by the unit test.
  - **Rerun with the fixed test (07:28-07:29, `results/test-idxsplit2.txt`): PASS on every case.** The old crash was the test's fp4 Q-scale shape (fp4 cannot serve on GB10: `dsa_indexer_uses_fp4` needs sm_10x), not ratio 2. Now covered: first-chunk, late-chunk r2, multi-request with decode rows (r1 and r2), 600 rows, 520 rows with an empty last rank, 8190 rows, 512 rows at MIN, and an 88K-key long case. Every case: top-k SET-EQUAL (order only; the reference differs in order from itself), candidates BYTE-EQUAL, slice logits BYTE-EQUAL, 4-rank spawn ranks identical. Ties: PASS (TIE-ONLY differences). E14 moved to the top of `queue.txt` at 07:30.
  - Decision: keep E14 in the queue; its screen adds needle checks at 64K and 128K (`sr_screen.sh` runs them for labels with idxsplit, combo or best). The agent is fixing the r2 test case.
- **E17 block_m16: not queued.** `exl3_moe._block_m` already picks 16 at decode (384 experts, 6 per token: C6 = 36 tokens → 0.6 rows per expert); forcing 16 would only slow prefill (128 rows per expert → tier 128). Spare go script `sr-e18-nth256-go.sh` (`NCCL_NTHREADS=256`) exists if a slot opens.
- **Engram GPU-resident idea: dead.** The Engram tables are most of the 430 GB model folder (16M-row vocab), far beyond the ~28 GB free per node.
- **MTP:** the checkpoint has `mtp.0-2`, but vLLM rejects `method="mtp"` on V4.1 (speculative.py:687). DSpark stays.
- **E20 k=4 (queued 07:45 after e12-ch4):** E05 showed step cost grows a lot with k (k=10 lost even on counting). Prose accepts ~1.9 tokens per step, so a k=4 round (5-token verify) may help prose and cost code a little. `sr-e20-k4-go.sh`.
- **DSpark k rule** (`vllm/config/speculative.py`): n_predict = dspark_block_size (5); k above 5 must be a multiple of 5. So k=1-5 and 10 are valid; k=4 is a possible prose lever.
- **To stop a queue:** `touch /var/tmp/boot-results/speedrun/queue.stop` (checked before each label), or `pkill -f "sr_queue.sh"`. Do NOT kill a running `sr_run.sh` mid-boot.

## Runner notes
- Chains run DETACHED on Reddie (`setsid nohup`), so they survive losing the Tailscale connection.
- `/root/sr_chain_e02.sh`: waits for E01, then runs E02 (`sr-e02-ch8-fast-go.sh`: ch8 plus `PATCH_NAME=dsv41-exl3-sr1` plus `DSV41_ENGRAM_FAST=1`), and runs the offline test (auto chunk) during E02's load → `test-fast-auto.txt`.
- Status files: `/var/tmp/boot-results/speedrun/run-<label>.status`, `chain-e02.log`.

## Experiment log
| # | change | boot | KV | C1 agg | C3 agg | C6 agg | code C1 | prefill 32K | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **b1-best** | **E02 + E09 + E19** (exl3b-roce, sr1roce, ch8, Engram fast, 128 read threads, RoCE) | 516 s, OK; RoCE live; MXFP8 kernels unchanged | **3,625,526** (+10.7% vs baseline) | **58.9** (+8.5%) | **124.8** (+8.7%) | **197.9** (+19.0%) | **91.8** (+13.9%) | **1,977** (+36% vs baseline cold 1,454); 8K **1,918** (+33%); probe **2,124**; needle 65K PASS 1,998 tok/s, 131K PASS 1,921 tok/s | **BEST SO FAR.** Prose 42.5 (+14%), idle count 116.6, idle code 86.3, quality PASS. JSON 62.0 (JSON is the noisiest category). Saved as `asusi:~/exl3tp4b-ablit-best-go.sh` at 08:28 |
| E14 | E02 + indexer prefill TP-split (`dsv41-exl3-sr2`, `DSV41_INDEXER_TP_SPLIT=1`) | 514 s, OK; "DSV41_INDEXER_TP_SPLIT on" logged | 3,486,710 | 57.5 | 122.5 | 191.8 | 86.0 | 1,765 (+0.8%); 8K 1,653; probe 2,043 (+1.3%); **needle 65K PASS 1,873 tok/s, 131K PASS 1,892 tok/s** | **NEUTRAL up to 40K** (the projected +27-48% did not show). Quality PASS, needles PASS. Long-context A/B queued: b1-best vs b1-idxsplit both run the 64K/128K needle |
| E19 | E02 + b12x RoCE one-shot all-reduce (`exl3b-roce`, `dsv41-exl3-sr1roce`) | 512 s, OK; "RoCEnante all-reduce is live" (first 409,600 B), all-gather live; backends `B12X_ROCENANTE` first; MXFP8 kernels unchanged (Emulation + FlashInferCutlass, as on exl3a) | **3,608,160** (+2.9% vs E02) | **60.1** (+2.2%) | **125.3** (+5.2%) | **191.7** (+1.2%) | **92.5** (+10.8%) | 1,793; 8K 1,682; probe 2,016 | **KEEP, big decode win**: prose 43.9 vs 37.7 (+16%), code 92.5 vs 83.5, idle count 117.3 vs 110.6, idle code 84.6 vs 81.5. Quality PASS. Credits @original-el8, @lukealonso |
| E05 | E02 + `SPEC_K=10` | 544 s, OK | 3,450,837 | 44.8 | 87.6 | 126.2 | 74.3 | 1,768; 8K 1,723; probe 2,012 | **DROP**: C1 -24%, C6 -33% vs E02; even counting falls (idle 80.7 vs 110.6) and prose 23.9. A 10-token DSpark round costs more than it accepts |
| E04 | E02 + `--enable-batch-sharded-sampling` | 541 s, OK | 3,494,513 | 58.1 | 110.1 | 186.5 | 88.1 | 1,757; 8K 1,625; probe 1,986 | **DROP**: C3 -7.6% and C6 -1.6% vs E02, C1 flat. Quality PASS |
| E09 | E02 + `ENGRAM_THREADS=128` | 513 s, OK | 3,501,822 | 56.3 | 118.0 | 185.4 | 85.5 | **1,983** (+13% vs E02); 8K **1,897** (+14%); 39.6K probe 2,109 (E02 2,016) | **KEEP for prefill** (+13-14%). Decode -2 to -4% vs E02 (prose 34.3 vs 37.7), likely noise but check in the final combo; if decode holds lower, try 64 threads |
| E03 | E02 + b12x MXFP8 dense (`vllm-dsv41:exl3b`, other MXFP8 kernels disabled) | 514 s, OK ("Using B12xMxfp8LinearKernel") | 3,484,154 | 54.6 | 118.1 | 188.1 | 82.1 | 1,691; 8K 1,566; 39.6K probe 1,914 (E02 2,016) | **DROP**: slower than E02 on C1, prefill and the probe. JSON C1 59.0 vs E02 84.0 shows JSON is noisy run to run |
| 00b | baseline SCREEN (as found, warm) | - | 3,274,912 | 54.3 | 114.8 | 166.3 | 80.6 | 1,454 (full-bench cold) | reference |
| E02 | E01 + Engram FAST staging (`dsv41-exl3-sr1`, `DSV41_ENGRAM_FAST=1`) | 534 s, OK ("Engram FAST staging on" logged) | 3,505,010 (+7.0%) | 58.8 (+8.4%) | 119.1 (+3.8%) | 189.5 (+13.9%) | 83.5 (+3.5%) | **1,751** (+20% vs baseline cold 1,454); 8K 1,665 (+15%) | **KEEP**. Quality PASS 5/5. JSON C1 84.0 (+32%). Idle back-to-back count 110.6. Prefill probe GPU util still 43-47% |
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

## UNATTENDED FINALIZER (armed 08:30, `/root/sr_finalize.sh`, status `finalize.status`, log `finalize.log`)
- At 09:55 UTC it sets `queue.stop`, waits for the running label, then builds the final go script from **`/var/tmp/boot-results/speedrun/final-labels.txt`** (now: `b1-best`; add winning `b1-*` labels on new lines, b1-best first; several labels are merged by `sr_combo.sh`).
- It saves that as `asusi:~/exl3tp4b-ablit-best-go.sh` (old copy kept as `.bak-HHMM`) and boots it with `/root/restore_exl3tp4b_ablit_best.sh`, which also tests the documented restore path. If that boot fails it restores b1-best; if that fails too, the pre-speed-run config.
- Then `sr_final.sh final-best` (full C1-C6, prefill 2K-64K, vision/tools, quality), a 131K needle, and `liveness.sh` (log only).
- To finalize early: `touch /var/tmp/boot-results/speedrun/finalize.now`. To cancel: kill the `^bash /root/sr_finalize.sh` process from a script file (pgrep gotcha).
- `asusi:~/exl3tp4b-ablit-best-go.sh` = `sr-b1-best-go.sh` since 08:28 (md5 3c124022).

## 07:59 rebase: the b1 ladder (`/root/mk_b1.sh`)
- **b1** = E02 (ch8 + Engram fast) + E09 (`ENGRAM_THREADS=128`) + E19 (RoCE, image `exl3b-roce`, patch set `dsv41-exl3-sr1roce`). Every `b1-*` label is b1 plus ONE change.
- Go scripts on Asusi: `sr-b1-best`, `sr-b1-idxsplit` (patch set `dsv41-exl3-sr2roce` = sr1roce + split indexer, on all 4 nodes), `sr-b1-k4`, `sr-b1-mb16k`, `sr-b1-idxlogits`, `sr-b1-ctx1m`, `sr-b1-noexpseg`, `sr-b1-gmu85`, `sr-b1-shexp`.
- `queue.txt` after the running e14-idxsplit (E02 base): b1-best, b1-k4, b1-mb16k, b1-idxlogits, b1-ctx1m, b1-noexpseg, b1-gmu85, b1-shexp. If E14 wins, insert `b1-idxsplit` right after b1-best.
- Removed: e12-ch4, e15-tree, e16-ch2 (with RoCE, NCCL only carries messages over 2 MB, i.e. prefill, so small-message NCCL tuning no longer applies) and the E02-based e20/e06/e10/e11/e07/e08/e13 (replaced by their b1 versions).
- Timeline: e14 ends ~08:15, then ~15 min per label; the `*best*` and `*idxsplit*` screens add the 64K/128K needle. Cut at ~10:15 for the final.

## Next action (updated 07:16)
- Running: E04 `e04-bss` (queue-1b), then E05 `e05-k10`, then the dynamic queue (`queue.txt`) from ~07:45.
- Keepers so far: E01 (ch8), E02 (Engram fast), E09 (128 read threads, prefill). Dropped: E03 (b12x), wo_a, drafter skip, block_m16.
- Waiting on agents: idxsplit test fix (r2 case), b12x RoCE port (`scratchpad/roce/`), draft KV page size (`scratchpad/draftpage/`).
- ~10:15 UTC at the latest: stop the dynamic queue after the current label (`touch queue.stop`), build `exl3tp4b-ablit-best` with `sr_combo.sh` from the keepers, boot it with `sr_run.sh`, then `sr_final.sh`. Copy the go script to `asusi:~/exl3tp4b-ablit-best-go.sh`.
- 11:00 UTC: issue and PR pass (`DRAFT-issue-pr-replies.md`, PR #7 added). 12:00 UTC: best config serving, repo pushed, memory updated.
