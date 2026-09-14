# SGLang lane, 2026-09-14

**Status: in progress.** The live log is [`STATE.md`](STATE.md).

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
