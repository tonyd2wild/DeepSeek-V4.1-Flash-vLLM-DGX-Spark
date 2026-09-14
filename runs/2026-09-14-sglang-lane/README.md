# SGLang lane, 2026-09-14

**Status: experimental, not the serving config.** It boots and serves, and holds a 2.5x larger KV pool than the vLLM lane, but on this configuration it decodes 2-2.5x slower and ran the nodes out of memory under the full bench. The vLLM speed-run config stays the default. The live log is [`STATE.md`](STATE.md).

## Result vs the vLLM lane (same bench, same prompts)

Boot `sg5` (mem-fraction 0.82, 500K context). The bench was cut off by a node hang after the 32K prefill, and the numbers were taken while the nodes were under memory pressure, so they understate SGLang somewhat; the gap is still large.

| | SGLang `sg5` | vLLM best (run 2) |
|---|---|---|
| C1 / C2 / C3 aggregate tok/s | 23.4 / 43.0 / 55.2 | 58.3 / 96.0 / 128.3 |
| C4 / C5 / C6 aggregate tok/s | 75.0 / 82.9 / 82.8 | 152.4 / 171.1 / 190.0 |
| code / prose, 1 stream | 34.7 / 16.3 | 85.2 / 41.0 |
| cold prefill at 2,950 / 11,592 / 46,810 tokens | 711 / 800 / 652 | 1,836 / 1,997 / 2,014 |
| counting speed, step time | 45.8 tok/s, 127 ms | 117.6 tok/s, 49 ms |
| KV pool | **9,323,776 tokens** (500K ctx) | 3,757,748 tokens (300K ctx) |

**Memory:** at 0.80 and at 0.82, long prefills in the bench drove the workers to ~0 GB available (Bluey 165 MB, Spark4 0 GB, swap in use) and hung the head node twice. The container `--memory 112g` limit did not prevent it. Before another attempt: lower `--mem-fraction-static` (~0.72-0.75), `--chunked-prefill-size 1024`, and a memory limit that is enforced for GPU allocations.

## How to run it

`bash /root/sg_up.sh <label>` on Reddie (stops vLLM, starts the 4 ranks, polls until serving), `bash /root/sg_final.sh <label>` for the full bench, `bash /root/sg_down.sh` to stop. Back to vLLM: `bash /root/sg_down.sh; bash /root/restore_exl3tp4b_ablit_best.sh`. Knobs: `MEMFRAC`, `CTX`, `CHUNK`, `SEQS`, `EP` in the environment of `sg_up.sh`.

Goal: serve DeepSeek-V4.1-Flash UNCENSORED on SGLang across the same four DGX Sparks and compare it head to head with the vLLM speed-run config (`runs/2026-09-14-speedrun`) on the same bench: C1-C6 over all categories, cold prefill, TTFT and KV pool.

## What runs

- **Engine:** upstream SGLang, image `lmsysorg/sglang:dev-dsv41` (arm64, commit `da64c5cb`), TP4 across 4 nodes with expert parallel 4, DSpark k=5, 300K context, vision (4 images) and tool calling on, thinking off by default.
- **Weights:** the official `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint (MXFP4 experts, FP8 dense) with Kai's uncensored overlay (`wo_b` of layers 10-35, 52 tensors, same dtypes and shapes as the originals). The folder `DeepSeek-V4.1-Flash-Ablit` hardlinks every official file and repoints those 52 tensors in the index, so it costs no extra disk. SGLang cannot load EXL3, so this lane does not use the EXL3 checkpoint.

## What we built (patch set `sglang-dsv41-disk`, 4 files, mounted over the image)

Built by `patches/make_sglang_disk_patch.py` from the image's own files with anchored edits.

| File | Change |
|---|---|
| `srt/layers/engram.py` | Disk-backed Engram rows (`SGLANG_DSV41_ENGRAM_DISK=1`). On GB10 the GPU and host share one 121.7 GiB pool, so neither upstream layout fits the ~47 GiB per-rank Engram shard next to the weights. Each rank memmaps its rows from the checkpoint (or a node-local copy), gathers only the rows a step needs on a 128-thread pool, and dequantizes on the GPU. Runs as a graph break, so decode uses the breakable CUDA graph backend. GPU test: byte-equal to direct reads. Ported from our own vLLM patches (Kai 2026-09-10, speed run 2026-09-14). |
| `srt/model_loader/weight_utils.py` | Load each tensor only from the file the index maps it to, so the uncensored overlay wins on every rank regardless of file order; skip the Engram tables in disk mode. |
| `kernels/ops/attention/flash_mla_sm120.py` | Split the extra (low-ratio) KV source into 64-token pages for FlashInfer's SM120 sparse MLA, in a separate buffer, when its page size is a multiple of 64 above 64. |
| `srt/layers/attention/deepseek_v4_backend.py` | Use DeepGEMM's own metadata planner for the ratio-1/2 indexers on SM120 (it matches DeepGEMM's SM120 kernels). |

## Credits

- **SGLang** (Apache-2.0): the engine, the DeepSeek-V4.1 support, the SM12x kernels, and the files our patches edit.
- **0xSero** ([deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000), MIT): the Engram-on-NVMe idea and the SM12x extra-KV and indexer-metadata issues our fixes address. Our code is original.
- **Kai (Tech2Wild):** the original Engram-on-disk design and the uncensored overlay.

## Memory setting (`--mem-fraction-static`, the gmu equivalent)

**Stable default: 0.80** (Tony, 2026-09-14). At 0.80 each GPU holds weights 75.3 GB, the DSpark drafter 3.8 GB and 11.49 GB of KV (7,012,096 tokens at 1,670.75 bytes per token), with 15.6 GB free after CUDA graphs for prefill scratch, NCCL, the OS and the Engram buffers.

**More-context levers** (estimates from the per-token cost, not yet measured; test each with a ~300K-token prompt before serving it, since long-prompt prefill scratch is what the free memory is for):

| `--mem-fraction-static` | KV pool (est.) | free after graphs (est.) | status |
|---|---|---|---|
| 0.80 | 7.0M tokens (measured) | 15.6 GB (measured) | **stable default** |
| 0.82 | ~8.5M tokens | ~13 GB | lever, untested |
| 0.85 | ~10.7M tokens | ~9.4 GB | lever, untested; tight for 300K prompts |

Set it with `MEMFRAC=0.82` (or `0.85`) in the environment of `/root/sg_up.sh`.

## Results so far

- **KV pool: 7,748,608 tokens** at memory fraction 0.80 (SGLang's own log: 1,670.75 bytes per token per GPU, 12.64 GB for KV), about 2.06x the vLLM lane's 3,757,748 on the same four Sparks. Measured on boot `sg2` before it failed.
- **Weights loaded** with Engram on disk: 74.68 GB per GPU plus the 2.08 GB DSpark drafter, with the uncensored overlay and expert parallel 4.
- **Serving since 14:28 UTC (boot `sg3`)**, KV pool **7,012,096 tokens** at 300K context (the drafter's extra memory on this boot left 11.49 GB for KV), full CUDA graphs for target and draft verify, Engram rows read from disk by the native reader inside the graphs.
- Before that, two boots failed (the vLLM config was restored in between, then stopped again at Tony's request):
  1. `sg1`: the MXFP4 MoE kernel needs 128-aligned expert slices; TP-splitting each expert gives 576. Fixed with expert parallel 4.
  2. `sg2`: upstream V4.1 code does not support breakable decode CUDA graphs (its Engram hashing needs a context only the piecewise backend sets), and our disk lookup needs a graph break. Next step: a graph-safe native row reader so decode can use the full CUDA graph.
- No speed numbers yet.
