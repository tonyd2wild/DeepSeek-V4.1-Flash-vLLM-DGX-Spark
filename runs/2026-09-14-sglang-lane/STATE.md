# SGLang lane, 2026-09-14 (started 13:14 UTC): state log

Goal (Tony, 13:10 UTC): stand up DeepSeek-V4.1-Flash UNCENSORED on SGLang across the 4 Sparks and compare it head to head with the vLLM speed-run config (C1-C6 on the same v41bench prompts, cold prefill, TTFT, KV pool). Switch only if it wins. The vLLM lane keeps serving until the SGLang boot; message Tony before that boot.

## Rules for this lane
- License: our repo is MIT. Use upstream SGLang (Apache-2.0), 0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000 (MIT) and original code only. No code from MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks (AGPL-3.0). Tony: no credit to that repo for simply moving to SGLang. Credit every file or idea we do use (0xSero MIT notice, SGLang Apache notice).
- One change per boot, two deaths = stop, restore with `bash /root/restore_exl3tp4b_ablit_best.sh`.

## Facts established
- Image: `lmsysorg/sglang:dev-dsv41` has an arm64 build (14.6 GB compressed). Pull started 13:14:42 on all 4 nodes (`/root/sg_pull.sh`, logs `/var/tmp/boot-results/sglang/`).
- Weights: the official checkpoint (476 GiB) is on Reddie only; the workers mount Reddie's `/var/tmp/models` read-only over NFS at `/mnt/reddie-models`.
- Uncensored overlay: Kai's `wo_b_l10_35_ablit.safetensors` holds 52 tensors (`layers.10-35.attn.wo_b.weight` F8_E4M3 [5120, 8192] and `.scale` F8_E8M0 [160, 256]), identical dtypes and shapes to the official tensors. Folder `/var/tmp/models/DeepSeek-V4.1-Flash-Ablit` (13:15): every file hardlinked from the official folder, the overlay hardlinked in, index repointed for the 52 keys. No extra disk used. Open question: does SGLang's loader honor the index per tensor (agent checking).
- KV math (from the MiaAI README, facts only): SGLang stores this model's context at 1,670.75 bytes per token per rank plus a fixed 0.34 GB sliding-window pool; our vLLM lane uses ~8,660 bytes per token per rank (30.32 GiB for 3,757,748 tokens).
- Note: SGLang cannot load EXL3, so this lane runs the official MXFP4 experts (~77 GiB per rank resident) rather than EXL3 3.5 bpw. The EXL3 memory headroom does not carry over.

- Engram in the official checkpoint: layers 1 and 14 only (model-00047/00048). Each: embed.weight F8_E4M3 [~384.0M, 256] 91.6 GiB + embed.scale F8_E8M0 [rows, 8] 2.86 GiB, plus small q/k/wkv tensors. About 189 GiB total, so it cannot be resident next to ~77 GiB of weights per rank.
- Every node already holds a sparse local copy of shards 47/48 with only its own row quarter (`/var/tmp/engram-local/DeepSeek-V4.1-Flash`, 48 GB on disk, built 2026-09-10 for vLLM; rank 1 = rows [96,000,564, 192,001,740) of layer 1). Possibly reusable by SGLang if it shards Engram rows the same way.

- Engram row quarters per node: rank 0 Reddie reads rows [0, 96.0M) from the full local shards (no engram-local copy); rank 1 Spark4 [96.0M, 192.0M); rank 2 Asusi [192.0M, 288.0M); rank 3 Bluey [288.0M, 384.0M) (layer 14 boundaries differ by a few thousand rows).

## Log
- 13:14 image pull started; model folder built; design agent started (`scratchpad/sglang-lane/`).
- 13:17 Engram facts sent to the design agent.
- 13:25:55 `lmsysorg/sglang:dev-dsv41` pulled on all 4 nodes (image 381b27ffa19b, 33.2 GB). Reddie disk now 110 GB free.
- 13:27 image inspected: aarch64, sglang commit `da64c5cb` "[DSV4.1] Support raw-index output in TopK v2", torch 2.13.0+cu130. Has `models/deepseek_v4.py`, `deepseek_v4_dspark.py`, `deepseek_v41_vit.py` (vision), `layers/engram.py`, DSPARK spec flags, and the SM12x attention kernels upstream (`flash_mla_sm120.py`, `fa4_sm120`), so no third-party attention file is needed.
- **Engram is the blocker:** upstream keeps the tables either TP-sharded in device memory or in host RAM (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`); there is no disk mode. On GB10 host RAM and GPU memory are the same 121.7 GiB, so ~47 GiB of Engram per rank next to ~77 GiB of weights does not fit either way. Plan: port OUR disk-backed Engram reader (MIT, Kai plus the speed-run fast path) into SGLang's `EngramEmbedding` as an env-gated mode reading the existing per-node row quarters. Source extracted to `scratchpad/sglang-src/` for the design agent.
- 13:35 Tony: "just GO complete the move" (boot authorized). Launcher written and deployed (`tools/`):
  - `sg_node.sh <rank>` on every node (`~/sg_node.sh`): docker run of `lmsysorg/sglang:dev-dsv41` with our patches from `~/patches/<PATCH_NAME>/mounts.txt`, model `/models/DeepSeek-V4.1-Flash-Ablit` (Reddie local, workers over NFS), Engram rows at `/engram` (Reddie: the full shards; workers: `/var/tmp/engram-local/DeepSeek-V4.1-Flash`), TP4 over 4 nodes (dist-init 192.168.192.2:29600), port 8000, ctx 300000, mem-fraction 0.80, chunked prefill 2048, 8 running requests, DSPARK block 5, tool parser `deepseekv41`, reasoning parser `deepseek-v41`, thinking off by default, vision 4 images, `expandable_segments:False`, NCCL env as the vLLM lane plus ch8.
  - `/root/sg_up.sh <label>`: pauses liveness, stops vLLM and SGLang on all 4, starts ranks 3, 2, 1 then 0, polls to 50 min, prints SGLang's KV lines, restarts liveness. `/root/sg_down.sh`: stops SGLang on all 4. `/root/sg_final.sh <label>`: the same full bench as the vLLM final (plus a 131K needle).
  - **Back to vLLM:** `bash /root/sg_down.sh; bash /root/restore_exl3tp4b_ablit_best.sh`.
  - Workers see the Ablit folder over NFS (61 files, overlay present, index has the 52 repoints); all users can run docker.
- 13:40 the design agent had written nothing after ~25 min, so I wrote the patch myself (agent kept as a cross-check).
- **Patch set `sglang-dsv41-disk`** (`patches/make_sglang_disk_patch.py` builds it from the image's own files by anchored edits; outputs + diffs in `patches/sglang-dsv41-disk/`, deployed to `~/patches/sglang-dsv41-disk/` on all 4 nodes):
  - `engram.py` (srt/layers/engram.py): `SGLANG_DSV41_ENGRAM_DISK=1` mode. `EngramEmbedding` keeps no weight/scale parameters; `_owned_rows` calls `_disk_lookup` (wrapped in SGLang's `eager_on_graph`, so it runs as a graph break), which de-duplicates the row ids on the GPU, gathers raw fp8 rows and e8m0 scales from read-only memmaps on a 128-thread pool into pinned buffers, dequantizes on the GPU with SGLang's own math, zero-fills unowned rows. Rows outside the node-local copy (engram-local.json range) come from the model folder. SGLang splits rows in exact quarters, our vLLM copies split at hash-head boundaries, so ranks 1 and 2 miss ~1,000-1,300 rows per layer locally; those few read from Reddie over NFS.
  - `weight_utils.py` (srt/model_loader/weight_utils.py): both safetensors iterators yield a tensor only from the file the index maps it to (so the uncensored overlay wins on every rank even if `SGLANG_SORT_WEIGHT_FILES` staggers file order), and skip the Engram embed tables in disk mode before they are materialized.
  - Launcher: `--cuda-graph-backend-decode breakable` (needed for the eager break), `SGLANG_DSV41_ENGRAM_DISK_THREADS=128`.
- 13:44 GPU test skipped on Spark4 (11 GB free under vLLM, guard 15 GB); run on Bluey instead (25 GB free).
- **13:45 Engram disk GPU test PASS** (`patches/test_engram_disk.py`, Bluey, one GPU, next to the serving vLLM):
  - as rank 2 (mixed: Bluey's local copy covers only [288,003,654, 288,004,626) of rank 2's range, the rest from Reddie over NFS): layers 1 and 14 EQUAL=True, 125 of 128 ids nonzero (3 unowned ids zero as required);
  - as rank 3 (pure local): layers 1 and 14 EQUAL=True;
  - timings (second call): 288 rows 0.4-1.3 ms, 98,304 random ids 10.2-13.9 ms.
- **13:46 boot `sg1` started** (`/root/sg_up.sh sg1`, log `/var/tmp/boot-results/sglang/boot-sg1.log`): stops vLLM on all 4 nodes, then SGLang ranks 3, 2, 1, 0.
