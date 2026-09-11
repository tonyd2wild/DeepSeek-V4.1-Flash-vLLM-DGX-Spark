# Recipe: DeepSeek-V4.1-Flash on four DGX Sparks (vLLM, TP4, DSpark, CUDA graphs)

This is the full path from an empty fleet to the serving endpoint. Every file it names is in this repo. The [README](../README.md) boot log records what each step fixed.

## 0. Hardware and layout

**Machines:**
- Four DGX Spark (GB10, SM 12.1, 128 GB unified memory each, about 121.7 GiB visible to the OS).
- ConnectX-7 RoCE fabric on 192.168.192.0/24.

**Where the model lives:**
- One node (the head, "Reddie" here) keeps the 510 GB checkpoint on local NVMe at `/var/tmp/models`.
- It exports that directory read-only over NFS. The other three mount it at `/mnt/reddie-models`.
- Nothing is copied to the workers.

| rank | node | IP | weights from |
|---|---|---|---|
| 0 | Reddie (head, API on :8000) | 192.168.192.2 | local NVMe |
| 1 | Spark4 | 192.168.192.4 | NFS |
| 2 | Asusi | 192.168.192.3 | NFS |
| 3 | Bluey | 192.168.192.1 | NFS |

**Why four nodes and a disk patch:**
- The routed experts (296 GB MXFP4) split four ways fit.
- The two Engram n-gram tables (203 GB FP8) do not: stock vLLM puts them in host memory, and on GB10 host memory is the GPU pool.
- With the Engram-on-disk patch, each rank loads 78.79 GiB without DSpark, or 81.36 GiB with the DSpark draft layers (measured). The tables stay on disk.
- TP2 does not fit either way.

## 1. Download (head node)

`build/dl-ds41.sh` pulls `deepseek-ai/DeepSeek-V4.1-Flash` into `/var/tmp/models/DeepSeek-V4.1-Flash`.

## 2. NFS

- Export `/var/tmp/models` from the head.
- Mount it read-only on each worker: `mount -t nfs -o ro,vers=3 192.168.192.2:/var/tmp/models /mnt/reddie-models`.
- Put the mount in `/etc/fstab`. Two of our workers had it only as a manual mount and lost it after a watchdog reset.

Load times (measured, cold page cache, because the launcher drops caches every boot):
- The head reads its weights in about 10 minutes.
- Workers take 10 to 18 minutes over NFS.
- DSpark adds a second pass over all 48 shards for the draft layers.
- The head waits for the slowest worker before profiling.

## 3. Engine image

Build each image on every node; they are node-local.

| image | how | why |
|---|---|---|
| `vllm-dsv41:overlay1` | `build/Dockerfile.overlay` on `vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c` (the merge-base of vLLM branch `dsv41-feat` at commit `e47aa780b`, the tree every patch here targets; see "Pin the branch commit" below), plus `_C_stable_libtorch` rebuilt for sm_121a (`build/build_stable_ext.sh`) | The branch's kernel changes all live in that one extension. |
| `vllm-dsv41:overlay3` | `build/build_overlay3.sh`: FlashInfer v0.7.0rc1 (`07869c61`) with pinned submodules; the stale 0.6.18 jit-cache/cubin packages are removed | FlashInfer 0.6.18's SM120 sparse-MLA decode lacks V4.1's topk of 1152. |
| `vllm-dsv41:overlay4` | `build/build_overlay4.sh`: prebuilds `mxfp8_gemm_cutlass_sm120` with `MAX_JOBS=2` | Its runtime compile (7 CUTLASS files, 22 parallel jobs) exhausted host memory on all four nodes at once in boot 3. |
| `vllm-dsv41:overlay5` | `build/build_overlay5.sh` + `build/prewarm5.py`: rebuilds `sparse_mla_sm120` under the exact runtime environment; `build/verify5.py` checks that nothing compiles at runtime | This is the serving image. |

### Pin the branch commit, not the branch

The seven patch files are whole-file replacements of `dsv41-feat` files **as they stood at commit
`e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`** (2026-09-10 07:24 UTC). The branch was force-pushed at 00:49 UTC on 2026-09-11 and
took more model commits after that (among them `99c83fbc4`, which changes the indexer around the file `patch/attention.py`
replaces). Building the overlay from the branch head after that date and mounting these patches gives an engine that boots
without a single error, passes profiling and graph capture, and emits one repeated garbage token from the first position, with
DSpark accepting nothing. It is not the checkpoint and not the node count; it is the tree under the patches.

The branch itself is gone: it was merged into main (vllm-project/vllm#56214) and deleted at 09:11 UTC on 2026-09-11, so
`git clone --branch dsv41-feat` now fails. GitHub still serves the commit by full sha (branch `dsv41-optimized` also pointed at
it on 2026-09-12):

```
git clone https://github.com/vllm-project/vllm.git
cd vllm && git fetch origin e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba && git checkout e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba
```

`build/fetch_vllm_branch.sh` does exactly that. If you must move to a newer branch commit, re-derive the patches from the
per-fix diffs in `patch/*/` on the new files and re-run the tests next to them; do not mount the old files on a new tree. The
before/after shas of every force push are listed by `GET /repos/vllm-project/vllm/activity?ref=refs/heads/dsv41-feat`.

## 4. Patches (bind-mounted over the image; nothing baked)

Copy these seven files and the manifest `patch/mounts.txt` to `~/patches/dsv41-boot10/` on every node (the launcher reads `~/patches/$PATCH_NAME`; boots 3-9 used `dsv41-boot3`, the same files with the previous `engram.py`). The launcher mounts each file over the site-packages path the manifest gives. `tools/prelaunch-8.sh` does this with md5 checks. The top-level `patch/` files are byte-identical to what the serving boot mounts; `patch/README.md` lists their md5s, and the subfolders hold each fix's diff and test.

| file (repo) | mounted over (`vllm/...`) | what it does |
|---|---|---|
| `patch/engram.py` | `models/deepseek_v4_1/common/engram.py` | **Engram on disk.** Rows are read with `preadv` from shards 47/48 and dequantized on the CPU. Includes the rank-offset fix (without it, ranks 1-3 read rank 0's rows; [details](boot3-wedge-and-engram-offset.md)). Includes one shared read pool, so every row for both Engram layers is in flight at once. Adds `EngramDiskStager`. |
| `patch/model_state.py` | `models/deepseek_v4_1/nvidia/model_state.py` | Stages the Engram rows in `prepare_inputs`, **before the forward**: one GPU hash, one host sync, parallel reads into the persistent `staged_rows` buffer. The forward then has no host round trip, so it can be captured as a CUDA graph. |
| `patch/weight_utils.py` | `model_executor/model_loader/weight_utils.py` | The loader skips the two Engram tables (203 GB never read at load). |
| `patch/attention.py` | `models/deepseek_v4_1/attention.py` | SM12x page sizes: the SWA cache and compressed-KV pages come from the backend. The indexer cache holds 64 states per page (64 tokens at ratio 1, 128 at ratio 2), because DeepGEMM's paged MQA logits only takes 32 or 64 ([details](boot5-indexer-pages.md)). |
| `patch/flashinfer_sparse.py` | `models/deepseek_v4_1/nvidia/flashinfer_sparse.py` | Pages hold 64 compressed states. Adds a 64-token SWA backend, the only page size FlashInfer's SM120 sparse-MLA kernels are built for. |
| `patch/sparse_swa.py` | `v1/attention/backends/mla/sparse_swa.py` | The `get_swa_block_size()` hook. |
| `patch/sparse_attn_indexer.py` | `model_executor/layers/sparse_attn_indexer.py` | SM12x decode top-k uses `top_k_per_row_decode`. `persistent_topk` oversubscribes GB10's 48 SMs on long rows and kills the engine; the generic kernel matches `torch.topk` and is 1.6-3.6x faster on GB10 ([results](../patch/sm12x-indexer-topk/RESULTS.md)). |

## 5. Launch

- `launch/dsv41-tp4.sh <rank>` starts one rank.
- `launch/boot_dsv41.sh` (run on any node with SSH to the others) starts ranks 3, 2 and 1, then the head, with identical knobs.
- `tools/launch.sh <N>` wraps it. It **stops `vllm_dsv41` on every node first, head first**, runs `/root/prelaunch-<N>.sh` if present, then runs `launch/boot<N>-go.sh`. Stopping everything first matters: a new worker that starts while an old head is still listening on the same port joins that head's rendezvous and hangs the new boot.
- The serving configuration is `launch/boot10-go.sh` (`PATCH_NAME=dsv41-boot10`, `ENGRAM_LOCAL=1`).
- **Bring it back after a reboot or crash:** `bash /root/restore_boot10.sh` on the head.
  - It runs `launch.sh 10` with `PRELAUNCH=/root/prelaunch-quick.sh`. That check refuses to launch on a missing NFS mount (it remounts first), a missing node-local Engram copy, files that differ between nodes, or a GPU burning under 50 TFLOPS (clock latch).
  - It also re-stages `/tmp/boot10-go.sh` on the fan-out node, then waits for the server and runs `postcheck10.sh`.
  - Used on 2026-09-11 after a worker was powered off by accident: serving again 11 minutes after the start.
- **Vision and tool calling:** `dsv41-tp4.sh` defaults to text-only with tools off (`TEXT_ONLY=1`, `PARSERS=0`). `boot10-go.sh` sets `TEXT_ONLY=0 PARSERS=1` plus `--limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1`, and that is what turns both on. `tools/vision_tools_demo.py` checks them end to end: 3 image tests, 4 tool-calling tests.
- **Node-local Engram rows (boot 10).** Before that boot, run `tools/engram_local.py` once on each worker, reading from the NFS mount and writing to `/var/tmp/engram-local/DeepSeek-V4.1-Flash`.
  - Pass the rank's row ranges, taken from the boot log line `Engram DISK mode: layer L rows [start, end)`. Example for rank 1: `1:96000564:192001740 14:96003054:192007016`.
  - Each worker needs about 48 GB of local disk. The script verifies random rows against the source.
  - `ENGRAM_LOCAL=1` mounts the copy. `patch/engram.py` uses it only when its recorded range covers the rank, and falls back to NFS otherwise. The head already reads from local disk.

Flags and environment that matter, and why:

| flag / env | why |
|---|---|
| `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[5,6,10,12,15,18,20,24,25,30,35,36,40,42,48]}` | CUDA graphs are the throughput fix: eager decode on this model is host-bound (about 200 ms per step, GPUs nearly idle). FULL graphs cover uniform decode batches; mixed batches use breakable PIECEWISE graphs. With DSpark k=5, every decode batch is a multiple of 6 target tokens, or 5 draft tokens, so each has an exact graph and nothing is padded. Padded speculative batches can hang SM120 sparse MLA (FlashInfer #5015). |
| `-e VLLM_USE_BREAKABLE_CUDAGRAPH=1` | Set explicitly on every node, because the eager-break decorator binds when the model imports. |
| `--speculative-config {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}` | DSpark with the checkpoint's own draft layers. Adaptive verification stays off: it forces variable-length decode graphs with padded rows, the #5015 trigger. |
| `--gpu-memory-utilization 0.80` | Measured: target graphs take 1.85 GiB and draft graphs 0.56 GiB; the KV pool is 4.84 GiB. |
| `--max-model-len 300000`, `--max-num-seqs 8`, `--max-num-batched-tokens 8192` | 300K context; KV pool **1,032,963 tokens** (3.44x at 300K), with vision and tools on. |
| `--block-size 128` | Required. vLLM would otherwise pick 64 (the smallest size the patched main backend lists), and the V4 indexer backend refuses it at KV init ([details](boot4-block-size.md)). The per-layer 64-state pages from the patches still apply. |
| `--engram-config '{"cpu_offload": false}'` + `DSV41_ENGRAM_DISK=1` | Engram from disk (32 read threads). |
| `--limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1` | Vision on: up to 4 images per request. The encoder adds about 0.22 GiB per rank (81.58 GiB with DSpark). |
| `--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41` | Tool calling on; the parsers and the Rust tool-parser extension ship in the image. |
| `--default-chat-template-kwargs '{"thinking": false}'` | Thinking off by default. A request can turn it on with `chat_template_kwargs`. |
| `-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1` | If anything still compiles at runtime, it cannot take the host down. |
| `-e VLLM_USE_FLASHINFER_SAMPLER=0` | Native sampler, so the first request does not JIT-compile one. |
| `-e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton` | Kernel caches on the host, so they survive container restarts. |
| NCCL: `NCCL_NET=IB`, `NCCL_IB_HCA=rocep1s0f0`, `NCCL_IB_GID_INDEX=3`, RoCE v2, `NCCL_SOCKET_IFNAME=enp1s0f0np0` | Same fabric settings as our other TP4 Spark recipes. |

**1M max context (proof boot, `launch/boot7-go.sh`):**
- Same stack, but eager, `--max-model-len 1048576`, gmu 0.80.
- It served with a 1,078,380-token DSpark KV pool (1.03x at 1M).
- The indexer prefill buffer grows with max context, about 5.2 GiB at 1M, so 1M needs the extra memory from eager mode and gmu 0.80.
- That boot predates the top-k fix, and a 32K request crashed it at decode. The fix is in the patch set above. Its fix is validated on GB10 at row widths up to 300,000; the full 1M-wide case has not been re-run.

## 6. Verify

- `tools/postserve.sh <label>` runs:
  - Kai's smoke test.
  - A greedy reference capture, or a comparison against an earlier capture.
  - A garble gate.
  - The fixed-prompt benchmark: `bench/v41bench.py`, prompt set `bench/prompts-v1.json`, C1-C6.
- `bench/v41needle.py` is the long-context needle test.
- `tools/hangcheck.sh` flags a #5015-style hang: requests running with 0.0 tok/s for 90 s.

## 7. Host hardening we recommend

- **Check the GPU clocks first.** A GB10 can latch below 1 GHz with no visible cause, and only unplugging the adapter for 30-60 s clears it. Two of our four were stuck. Under a 15 s fp16 burn (`tools/recover.sh`), every node should show about 2.2-2.4 GHz, 80 W+ and 75-90 TFLOPS. [details](gpu-clock-latch.md)

These are system settings, so apply them yourself:
- Enable `dgx-anti-oom` and make its container regex match `vllm_dsv41`.
- Set `vm.min_free_kbytes=1048576` and `vm.watermark_scale_factor=200`.
- Add the fstab mounts from step 2.
- Raise the head's nfsd threads from 8 to 32.
