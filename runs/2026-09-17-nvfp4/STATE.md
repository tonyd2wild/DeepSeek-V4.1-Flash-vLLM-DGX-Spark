# NVFP4 lane, 2026-09-17

Question: is `nvidia/DeepSeek-V4.1-Flash-NVFP4` faster than what we serve, on 4 DGX Sparks at TP4?
Our serving config is the 2026-09-14 speed-run best (EXL3 3.5 bpw experts, 500K context). The honest
comparison for the checkpoint swap is against **boot 10**, which is the same stack on the official
MXFP4 pack, because NVFP4 only re-quantises the MoE experts.

Nothing here is a recommendation to switch. Measured numbers only; where a number is computed it says so.

## The checkpoint

Measured on Reddie, 2026-09-17:

| | NVFP4 (nvidia) | official pack |
|---|---|---|
| total, summed from safetensors headers | **527,273,322,840 B = 491.06 GiB** | **510,286,023,000 B = 475.24 GiB** |
| tensors | 188,245 | 96,085 |
| MoE expert bytes (`layers.N.ffn.experts.*`) | 305,765,130,240 = 284.77 GiB | 288,777,830,400 = 268.95 GiB |
| everything else | 221,508,192,600 = 206.30 GiB | **the same 221,508,192,600 B** |
| Engram, stays on disk | 202,758,032,400 = 188.83 GiB | identical |
| Engram, actually loads | 315,043,840 = 0.29 GiB | identical |
| MTP / DSpark draft tensors | 7,932,874,632 = 7.39 GiB | identical (drafter stayed MXFP4) |
| layer 1 / 14 Engram rows | 384,006,168 / 384,016,682 | identical |
| routed experts | 384 (`n_routed_experts`), 40 layers | identical |

The non-expert bytes are **byte-identical** between the two packs, so the repack touched only the
main-model routed experts. The entire 16,987,299,840 B delta is expert scales: NVFP4 `weight_scale`
F8_E4M3 `[2304,320]` = 737,280 B per expert replacing MXFP4 `scale` F8_E8M0 `[2304,160]` = 368,640 B,
plus 8 B of fp32 scalars per expert. Expert bits per weight: **NVFP4 4.500, MXFP4 4.250**.
99.84% of Engram never enters memory.

**Trap:** the NVFP4 pack's `model.safetensors.index.json` `total_size` field reports 510,286,023,000,
which is the MXFP4 figure. It under-counts the NVFP4 pack by exactly 15.82 GiB. Sum the headers.

- `hf_quant_config.json`: `quant_algo MIXED_PRECISION`, `moe_quant_algo NVFP4`, `group_size 16`,
  `kv_cache_quant_algo null`, ignore list `*.attn.*`, `*.ffn.shared_experts.*`, `head`, `mtp.*`.
  So only MoE experts change; attention, shared experts, head, MTP and the KV path are untouched.
- The official pack already ships FP4 experts (`expert_dtype: fp4`, MXFP4 block 32). NVFP4 group 16
  carries twice the scale factors, which is where the +15.9 GiB on disk comes from.
- **That extra disk does not obviously cost per-rank memory.** Computed per-rank resident for NVFP4 at
  TP4 is 80,652,239,808 B = 75.11 GiB for the main model plus 2.02 GiB for the draft = **77.14 GiB**,
  against a **measured** 81.36 + 2.57 = 83.93 GiB on the MXFP4 boots 9 and 10. The 10.2 GiB gap is
  closed, as inference not measurement, by two candidates: the loader converting MXFP4 g32/E8M0 expert
  scales into the g16/E4M3 layout on Blackwell (+3.96 GiB/rank, which is precisely what the NVFP4 pack
  already ships) and Engram `embed.scale` staying resident (+5.72 GiB/rank). If that holds, NVFP4 boots
  at the same ~81 GiB/rank as MXFP4 and the 15.8 GiB of disk buys skipped conversion work, not KV
  headroom. One boot log line decides it: ~75 GiB means NVFP4 wins ~6 GiB of KV, ~81 GiB means a wash.
- EP4 versus TP4 makes no meaningful difference to resident bytes: 82,826,050,440 vs 82,826,326,920,
  a 276,480 B difference caused by per-tensor fp32 scalars being replicated under TP and partitioned
  under EP. EP is a load-path fix, not a capacity fix.
- Sanity check on the accounting: EXL3 3.5 bpw computes to 55.37 GiB/rank against 56.61 GiB measured,
  a 1.24 GiB residual for the trunk and draft. The expert-parameter count (543,581,798,400) holds.
- **Trap:** the Engram tensors sit at different byte offsets in the NVFP4 shards. In the official
  shard 47, `layers.1.engram.embed.weight` starts at data offset 0; in NVFP4, `k_weight` is at 0 and
  `embed.weight` at 3,072,284,864. Row ranges are identical, byte layout is not, so node-local Engram
  copies made for the official pack must not be reused. `tools/engram_local.py` re-reads the source
  header per run, so a fresh staging pass is correct.

Per-rank Engram ranges (unchanged from boot 10, read from each worker's live `engram-local.json`):

| rank | node | layer 1 | layer 14 |
|---|---|---|---|
| 1 | Spark4 | 96000564:192001740 | 96003054:192007016 |
| 2 | Asusi | 192001740:288003654 | 192007016:288011564 |
| 3 | Bluey | 288003654:384006168 | 288011564:384016682 |

## Fleet state before this run

- All four Sparks rebooted 21:03-21:04 UTC. Burn gate 21:15: Reddie 85.3, Spark4 83.7, Asusi 81.6,
  Bluey 77.4 TFLOPS at 2164-2411 MHz. The GB10 clock latch that blocked the earlier attempt is cleared.
- The previous attempt (18:49 UTC, gmu 0.80, 131,072 context, Engram over NFS, no memory guard) loaded
  all 48 shards and then hard-reset every node. Reddie's journal for that boot ends at **19:00:12**
  with no OOM kill, no Xid, no panic and no shutdown sequence, and the next boot starts 19:03:14.
- Free space after staging: Reddie 158 GB, Spark4 52 GB, Asusi 42 GB, Bluey 949 GB.
  Asusi still holds 63 GB of `engram-local/DeepSeek-V4.1-Flash-EXL3-TP3` from a retired TP3 lane.
  Nothing deleted; Tony's call.

## Boot A (21:26:53 UTC)

`/root/final/bootnvfp4a-go.sh`: boot 10 stack, checkpoint swapped, gmu 0.78.

| knob | boot A | boot 10 |
|---|---|---|
| model | DeepSeek-V4.1-Flash-NVFP4-nvidia | DeepSeek-V4.1-Flash |
| image / patches | vllm-dsv41:overlay5 / dsv41-boot10 | same |
| gmu | 0.78 | 0.80 |
| max context | 300,000 | 300,000 |
| decode | DSpark k=5, FULL_AND_PIECEWISE | same |
| Engram | disk, node-local, 32 threads | same |
| tools / vision | on, 4 images | same |

Guardrails the earlier attempt did not have: node-local Engram staging (8008 rows verified per worker,
0 mismatches), `prelaunch-nvfp4.sh` (mounts, md5s, Engram presence, burn >= 50 TFLOPS), and
`memguard_nvfp4.sh` on all four, which removes the local container if MemAvailable stays under 5 GiB
for two 3 s samples. That turns a unified-memory wedge into a clean stop instead of a watchdog reset.

### Result: aborted by the memory guard, 21:30:35

All four guards fired within a minute of each other, each at 3-4 GiB available:

| node | fired | available |
|---|---|---|
| Reddie | 21:30:35 | 4 GiB |
| Asusi | 21:31:01 | 3 GiB |
| Bluey | 21:31:11 | 4 GiB |
| Spark4 | 21:31:33 | 3 GiB |

Reddie's curve: 24 GiB at 21:29:17, 16 at 21:29:47, 8 at 21:30:17, 4 at 21:30:32. A straight line to
zero in 75 s, the same shape as the 19:00 hard reset. Every node stayed up; no reboot, no watchdog.
So this is not a head-node or NFS-server effect: every rank ran out of host memory in the same phase,
after weight load and before KV allocation.

## Boot B (21:34:00 UTC): expandable_segments:False

One change: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, appended to `NCCL_EXTRA` so it
overrides the launcher's default. Precedent: the speed run measured +6.3% KV pool from this flag on
the EXL3 lane (issue #2, ecohash-co). Also added a breakdown sampler to the guard.

**Aborted 21:38:01, and the sampler named the mechanism.** Reddie's curve:

| time | avail | cached | anon |
|---|---|---|---|
| 21:35:24 | 114 | 1 | 2 |
| 21:35:42 | 36 | 8 | 3 |
| 21:35:51 | 27 | 13 | 3 |
| 21:36:27 | 27 | 34 | 3 |
| 21:38:01 | **3 (fired)** | 10 | **27** |

- The 72 GiB that vanishes at 21:35:42 is the weight allocation. On GB10 unified memory it appears in
  neither `AnonPages` nor `Cached`, and `nvidia-smi --query-gpu=memory.used` returns `[N/A]`, so
  `MemAvailable` is the only usable signal.
- `Cached` growing while `avail` stays flat is normal page cache and reclaimable.
- What kills the node is **`AnonPages` climbing to 27 GiB**: host-side loader buffers. The default
  iterator in `weight_utils.py` does `safe_open(...)` then `param = f.get_tensor(name)` per tensor,
  which materialises each tensor in host memory before the TP slice is copied to the device.
- Ruled out by reading the logs, not assumed:
  - **NFS auto-prefetch is not involved.** The workers logged
    `Network filesystem (NFS) detected but checkpoint total size (491.08 GiB) exceeds 90% of available
    RAM (28.27 GiB). Skipping auto-prefetch.`
  - **Engram is behaving.** Workers logged `read from node-local /engram-local` for both layers with
    the NVFP4 byte offsets, and `Engram table DISK-backed: ... 23.60 GiB not allocated` per layer, so
    the staged copies are being used and the tables are not resident.
  - Page cache is not the problem; with no containers running, Reddie shows 122 GiB available with
    51 GiB of `Cached` counted inside it.

## Boot C (21:40:31 UTC): expert parallelism

One change: `--enable-expert-parallel`. Under EP, `default_loader.py:399` calls
`compute_local_expert_ids`, and the `should_skip_weight(name, local_expert_ids)` guard that sits
directly above the `get_tensor` call in the iterator starts returning True for other ranks' experts.
Each rank then loads **64 of 256 experts** instead of a slice of all 256, so per-rank read volume and
the host buffers that killed A and B both drop about 4x. Resident weight bytes per rank are unchanged:
whole experts instead of quarter experts.

### Result: aborted 21:45:13, anon=27 GiB again, and it showed why

`EP weight filter` never appeared in the log. `--enable-expert-parallel` alone is not enough:
`default_loader.py` requires **three** conditions, and the third is
`parallel_config.enable_ep_weight_filter`, which **defaults to False** (`config/parallel.py:174`).
EP was active for the MoE layers (workers renamed `Worker_TP1_EP1`, `expert_map_manager` logged
`[EP Rank 1/4] Expert parallelism is enabled`) but every rank still read all experts, so the load
behaved exactly like boot B.

## Boot D (21:46:23 UTC): the EP weight filter actually on

One change: `--enable-ep-weight-filter` alongside `--enable-expert-parallel`. The flag is real in this
build (`engine/arg_utils.py:2309`).

**The filter engaged and fixed the load phase.** Each rank logged
`EP weight filter: ep_size=4, ep_rank=N, loading 96/384 experts` (384 experts, not 256), and through
the entire weight load `AnonPages` stayed at **2-3 GiB** on all four nodes, against 27 GiB without it.
Available memory sat flat at 27-28 GiB while only `Cached` moved.

**But the head still died, at 21:51:00, in a later phase.** The workers were untouched and still
loading over NFS, idle at 2 GiB, when rank 0 went:

| time | avail | cached | anon |
|---|---|---|---|
| 21:49:56 | 15 | 22 | 15 |
| 21:50:23 | 10 | 16 | 20 |
| 21:50:51 | 5 | 12 | 25 |
| 21:51:00 | **4 (fired)** | 11 | **27** |

A steady ~2 GiB per 9 s, starting after rank 0 finished its own load. Rank 0 gets there first because it
reads the checkpoint from local disk in 45 s while the workers take minutes over NFS. So the remaining
cost is **post-load, rank-0-first**, not the load itself.

Two candidates, both documented upstream (found by a research pass, listed with sources in README.md):
- the DSpark drafter's **second 48-shard load pass**, which uses its own loader instance and so does not
  get `local_expert_ids`;
- KV-cache profiling and CUDA graph capture, which vLLM issue #56824 reports as ~29 GiB vanishing on a
  GB10 with a modelopt NVFP4 model while `MemAvailable` still reads ~22 GiB.

Also fixed after this boot: the guard now saves `docker logs --tail 400` to `/var/tmp/guard-lastlog.txt`
before removing the container. Boots A through D destroyed the head's log on the way down.

## Boot E (21:54:06 UTC): drafter off

One change: `SPEC=none`. Tests the drafter-second-pass candidate. Decode will be slower without DSpark;
the point is to separate the two causes and get the model serving.

### Result: aborted 21:58, same curve, so the drafter is not the cause

With `SPEC=none` the head still climbed to anon 27 GiB after its load, at the same ~2 GiB per 9 s.
The guard's new log capture settled where: the saved head log ends **exactly** at
`Loading safetensors checkpoint shards: 100% Completed | 48/48` and contains nothing after it. So the
memory goes in the unlogged `process_weights_after_loading` phase, between the last shard and the first
profiling line.

The arithmetic of that phase matches the observation. For the auto-selected `FLASHINFER_CUTLASS`
backend, `flashinfer_fp4_moe.py:337-349` calls `reorder_w1w3_to_w3w1(w13, w13_scale)` per layer, then
`swizzle_blockscale`. Per rank that is 96 experts x 5,898,240 B x 40 layers = **21.1 GiB** of w13 plus
**2.6 GiB** of scales = **23.7 GiB**, against 25-27 GiB measured, over about 40 intervals for 40 layers.

## Boot F (22:01:19 UTC): --moe-backend cutlass

Rejected at startup in 2 minutes, which closes off the whole backend menu:

```
ValueError: NvFp4 MoE backend 'VLLM_CUTLASS' does not support the deployment configuration
since kernel does not support parallel config FusedMoEParallelConfig(tp_size=1, pcp_size=1,
dp_size=1, ep_size=4, ... use_ep=True, all2all_backend='allgather_reducescatter', ...)
```

`VLLM_CUTLASS` was the one available backend whose prepare path skips the w13 reorder, and it cannot
run with EP. Of the rest: `FLASHINFER_CUTEDSL` reorders **and** interleaves w13 and its scales **and**
swizzles both (`flashinfer_fp4_moe.py:143-153`), `FLASHINFER_TRTLLM` reorders then pads then shuffles
(`:364-391`), `MARLIN` is reported at ~105 GiB per rank (vLLM issue #50925), and `EMULATION`
dequantises. **FLASHINFER_CUTLASS is the leanest backend this parallel config accepts.**

## Boot G (22:06:28 UTC): page cache and vision

Boot E plus two host-memory changes in the same phase:
1. `cachekeeper.sh` on all four nodes drops clean page cache every 10 s for the boot. On GB10 the
   unified-memory driver counts page cache as used while `MemAvailable` counts it as free, so the
   ~107 GiB of shard pages each rank has just read can starve the conversion while the node still looks
   free. Same mechanism as the existing `/etc/cron.d/drop-caches`, at 10 s instead of 5 minutes.
2. Vision and parsers off, which frees the vision tower and its 1 GiB host cache before the conversion.

### Result: aborted 22:11:03, and it ruled out page cache

20 forced cache drops, vision off, identical death at anon 27 GiB. Page cache is not the constraint and
neither is the vision tower.

## Boots H through L: chasing the phase, and a correction

**Boot H (22:30, patch set dsv41-nvfp4).** Patched `modelopt.py` to call `torch.cuda.empty_cache()` after
the per-layer `replace_parameter` calls in the NVFP4 MoE `process_weights_after_loading`. The patch was
verified live inside the running container, and **its log marker never printed**, so that code path was
never reached before the node died. A first attempt at this boot was void: the go script lost its
`export` lines to a scripting error and ran patch set `dsv41-boot3` with default knobs.

**Boot I (22:36, `VLLM_LOGGING_LEVEL=DEBUG`).** The window is silent even at DEBUG. Last worker line is
`Loaded weight lm_head.weight`, then nothing while memory drains.

**Boot J (22:42, py-spy).** Installed py-spy in a throwaway venv on the host and dumped the worker's
Python stack during the climb. This corrected the diagnosis:

```
climb detected 22:45:31, avail 21 GiB
Thread 270 (active): "MainThread"
    _load_w2 (vllm/model_executor/layers/fused_moe/routed_experts.py:561)
    _load_model_weight_or_group_weight_scale (routed_experts.py:367)
    weight_loader (routed_experts.py:806)
    load_weights (vllm/models/deepseek_v4_1/nvidia/model.py:769)
    ...
    load_model (vllm/v1/worker/gpu_worker.py:502)
dump 2 at 22:45:43, avail 19 GiB:  _load_w13 (routed_experts.py:526)
```

**The memory goes inside weight loading, not after it.** The earlier reasoning was wrong: the tqdm line
`Loading safetensors checkpoint shards: 100% Completed | 48/48` counts **files opened by the iterator**,
not tensors copied into the model, so per-expert copies continue after the bar reads 100%. The cost is
`expert_data.copy_(loaded_weight)` in `_load_w13` / `_load_w2`, run for 96 local experts x 40 layers.
This matches vLLM PR #47580 upstream, which reports the fused-MoE loader materialising a non-contiguous
CPU source and proposes doing the work on the GPU instead. Not merged.

Note the size coincidence, offered as a lead rather than a conclusion: per-rank w2 bytes are 23.7 GiB,
and the climb tops out at 27 GiB.

**Boot K (22:47, glibc allocator).** `MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_=64 MiB`,
`MALLOC_MMAP_THRESHOLD_=128 KiB`, `MALLOC_TOP_PAD_=0`. Died 22:52:18, anon 27 GiB. **glibc arena
retention is ruled out.**

**Boot L (22:53).** Patches `routed_experts.py` at all five `expert_data.copy_` sites: every 256 copies,
synchronize, log `torch.cuda.host_memory_stats()` and call `torch._C._host_emptyCache()`. Tests whether
PyTorch's pinned host caching allocator is holding a staging buffer per pageable CPU->GPU copy, and
reports its own numbers either way.

### Result: the pinned host allocator is ruled out, with numbers

`expert copy 256 ... 6912: pinned host allocated 0.00 GiB, reserved 0.00 GiB, cache released`, all the
way to death at 22:58. PyTorch's pinned host caching allocator holds nothing here.

## The mechanism, measured

With the climb in progress, `/proc/<worker pid>/smaps` named it. The largest anonymous regions are the
**checkpoint shards themselves**, mapped `rw-p`:

```
1.85 GiB  rw-p  /models/DeepSeek-V4.1-Flash-NVFP4-nvidia/model-00003-of-00048.safetensors
1.31 GiB  rw-p  /models/DeepSeek-V4.1-Flash-NVFP4-nvidia/model-00004-of-00048.safetensors
Private_Dirty: 13,584,672 kB    Anonymous: 6,121,040 kB
```

Per-process sampling 25 s apart, while every other process stayed flat:

| process | anon t1 | anon t2 |
|---|---|---|
| `VLLM::Worker_TP0_EP0` | 14,876 MiB | **19,420 MiB** |
| `VLLM::EngineCore` | 682 MiB | 682 MiB |
| `vllm` (API server) | 892 MiB | 892 MiB |

A second, independent source-reading pass reached the same place and sized it exactly: one shard holds
one layer's 384 experts, this rank's 96/384 slice is **1,822 MiB**, and each mapping shows about
1,890 MiB anonymous. About 15 such mappings resident is the 27 GiB. It also confirmed, from the absence
of `default_loader.py:431 "Loading weights took"` and `base_loader.py:72` in every boot log, that the
worker **dies before `process_weights_after_loading` ever runs** - so the boot H patch was a no-op
against this symptom, and the NVFP4 kernel conversion is not the consumer. That conversion costs about
202 MiB net per layer, not 21 GiB: `reorder_w1w3_to_w3w1` is **in place** with a 64 MiB transient cap
(`flashinfer_fp4_moe.py:35-52`), and only the block scales are copied.

**So: private file mappings of the shards are COW-broken during loading, which turns file pages into
anonymous pages the kernel cannot reclaim.** Inferred mechanism, not read from source: the driver pins
host pages for DMA with `FOLL_WRITE` as queued H2D copies drain, which breaks COW on a `MAP_PRIVATE`
mapping. No in-place write to `loaded_weight` exists anywhere in `fused_moe/`, `models/deepseek_v4*/`
or `modelopt.py`.

## Boots M and N: copy off the mapping

**Boot M (22:59)** patched the iterator to `param.clone()` before yielding, breaking the aliasing
(`DSV41_COPY_WEIGHTS=1`). The dirty file pages disappeared, and the same volume reappeared as plain
anonymous allocations in the worker (24.3 GiB: one 11.4 GiB region plus several of 1-2 GiB). Loading also
slowed from about 30 s to a ~4 minute ETA for the extra memcpy. So the bytes are **held in flight**, not
merely touched: copying relocates them rather than freeing them.

**Boot N (23:04)** added the glibc return-to-OS settings on top, which only become meaningful once the
bytes are `malloc`'d rather than dirty file pages.

Both were stopped when Tony called the lane for the night. The untested combination that follows from the
mechanism is **clone plus a periodic drain** (`DSV41_COPY_WEIGHTS=1` with `DSV41_EXPERT_COPY_SYNC=64`,
staged as `bootnvfp4o-go.sh`): the clone keeps the mapping clean, and the drain lets each batch of copies
complete so the buffers are actually released. M and N each had one half of that.

## Where this lane stands

Not serving. The blocker is an upstream memory-management problem, not our configuration, and it needs a
loader patch rather than a flag. Ruled out with measurements, each in its own boot: page cache (20 forced
drops), transparent huge pages (`madvise`, `AnonHugePages: 0`, `thp_file_alloc: 0`), glibc arena
retention alone, PyTorch's pinned host allocator (0.00 GiB), the DSpark drafter, every available NVFP4
MoE backend, and the PP2xTP2 layout.

**Contrast measurement, same box, minutes later.** The EXL3 lane restore was sampled mid-load at the same
point in its load (shard 10 of 49): worker `Anonymous` **2 MiB**, `Private_Dirty` **2 MiB**, node anon
4 GiB, 53 GiB available. Against the NVFP4 lane's 14.9 to 19.4 GiB of worker anon and 6 GiB available.
So this is not a generic "big checkpoint on unified memory" problem: the EXL3 pack does not dirty a single
page. It is specific to the fused-MoE expert copy path these FP4 packs take, which is consistent with the
py-spy stack landing in `_load_w13` / `_load_w2`.

One fix is real and worth keeping for any future attempt: **`--enable-expert-parallel` plus
`--enable-ep-weight-filter`**, which skips 194.6 of the 302 GiB body per rank before any read and took
the load phase from anon 27 GiB to 2-3 GiB.

Also worth recording: two independent sources report NVFP4 losing on SM121 (one measured Marlin at
50.0 tok/s / 32 GB against FlashInfer 42.6 tok/s / 39 GB on a different model), NVIDIA validated this
checkpoint on GB300 rather than GB10, and no public report exists of anyone serving it on a Spark. The
speed upside was never established, so the value of this lane was the diagnosis, not a config to ship.
