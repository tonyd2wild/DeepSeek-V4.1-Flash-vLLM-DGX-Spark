# DeepSeek-V4.1-Flash EXL3 on 3 DGX Sparks (TP3) and 4 DGX Sparks (TP4)

Status: bring-up in progress (2026-09-11). Results tables are filled from measured runs only.

## What this lane is

The same boot 10 recipe, with one different checkpoint: [bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard). Only the 40 x 384 routed experts are EXL3 (K=3/K=4 mix, 3.51 bpw average, quantized with the Pollard method). Every other tensor is byte-identical to the release.

| | release (boot 10) | EXL3 build |
|---|---|---|
| total size | 510.3 GB | 460.0 GB |
| routed experts (shards 3-42) | 295.7 GB | 245.4 GB |
| shards 1, 2, 43-48 (incl. the two Engram tables, 203 GB) | | sha256-identical to the release |

Checked against the Hugging Face LFS sha256 of every shard. Only the 40 changed shards are downloaded; the other 8 are hardlinked from the release copy on the head node (`exl3/tools/dl-exl3.sh`, `exl3/tools/reddie_prep_tp3.sh`).

## Why 3 Sparks now fits

Weights that stay in GPU memory per Spark (Engram tables on disk as in boot 10). The estimate is computed from the safetensors headers; the measured column is vLLM's own "Model loading took" figure, which also counts padding and load-time buffers:

| layout | estimate, GiB per Spark | measured, GiB per Spark |
|---|---|---|
| release, TP4 (boot 10, DSpark + vision) | 73.2 | 81.6 |
| release, TP3 | 97.1 (does not fit) | |
| EXL3, TP4, DSpark + vision (exl3tp4b) | 61.5 (+6.2 on the two ranks with the wider expert slice) | 56.6 on the two narrow-slice ranks, 68.9 on the two wide-slice ranks |
| EXL3, TP3, DSpark + vision (try 11) | 81.5 | 84.2 |
| EXL3, TP3, vision, no DSpark (tries 9, 10) | | 80.3 |
| EXL3, TP3, text-only, no DSpark (tries 6, 7) | 78.0 | 79.9 |

## What TP3 needed (none of it exists upstream for V4.1)

1. **Virtual heads** (`patch/exl3-tp3/virtual_heads.py`, hooked in `attention.py`). V4.1 has 64 attention heads in 8 output groups; neither divides by 3. The TP3 copy of the model declares 72 heads / 9 groups in `config.json` (24 heads = 3 whole groups per Spark). At load time the 64-head tensors are padded before vLLM slices them: `wq_b` and `wo_a` repeat the last real group, `wo_b` gets zero columns (scale 1.0), so the padded group adds exactly nothing and no all-zero block ever reaches the FP8 activation quantizers. Checked offline on the real layer-5 and DSpark weights against the 64-head math: max relative error 3.4e-6, all finite (`exl3/tools/test_virtual_heads.py`). Same idea as the MiniMax-M3 TP3 virtual sharding (64 to 96 heads) in Luke Alonso's vLLM fork.
2. **Vocabulary split** (`patch/exl3-tp3/vocab_parallel_embedding.py`). vLLM splits the vocabulary with an exact `divide(129280, tp)`, which asserts at TP3. The patch pads to a multiple of lcm(64, tp): 129,408 at TP3 (43,136 per Spark); unchanged at TP2/TP4/TP8.
3. **Engram rows for 3 ranks.** Each rank owns 8 of the 24 hash columns. Row ranges (checked by reproducing the published TP4 ranges exactly): rank 1 `1:128000880:256002934 14:128004290:256009984`, rank 2 `1:256002934:384006168 14:256009984:384016682`. Node-local copies made with `tools/engram_local.py`.
4. **cuda-exl3 plugin.** Image `vllm-dsv41:exl3a` = overlay5 + [Zeuss5/cuda-exl3](https://github.com/Zeuss5/cuda-exl3) master `6a1ffc34`, built for sm_121a (`exl3/tools/build_exl3a.sh`). bot-lab-21's `exl3_config.py` (hybrid FP8 delegation, drafter model-path fallback) and `exl3_moe.py` (V4.1 clamped SwiGLU, 128-aligned TP split) are mounted over the plugin.

5. **Streaming weight load** (`patch/exl3-tp3/vl_model.py`). V4.1's multimodal wrapper sorted the whole checkpoint into a list before loading any of it, so every tensor, and every shard's mmap, stayed alive until the last one was loaded. On a GB10 each live mapping also holds host memory for what was copied out of it, so a worker's anonymous memory climbed for the whole load (about 25 GiB per Spark at TP3). The patch streams the `language_model.` weights straight through the loader, still as one contiguous group (the reason for the sort), and buffers only the small vision/aligner remainder. Boot 10 loads through the same function; what it costs there is not measured yet.

6. **DSpark at TP3** (`patch/exl3-tp3/dsv4_nvidia_model.py`). The DSpark drafter builds a standard DeepSeek-V4 MoE layer with 128 routed experts, and that layer's init asserted that the expert count divides the TP size (try 8). With expert parallelism and EPLB off, FusedMoE keeps every expert on every rank and splits the intermediate dimension instead, so the per-rank expert range the check guards is bookkeeping. The patch skips the check only in that case (all experts local); divisible counts, including the main model's 384, run the original code. The drafter's vocabulary head is a `ParallelLMHead`, so the vocabulary padding above covers it.

Replicated on every rank already, so nothing to do: the indexer, the compressor, the Engram `wkv`, the hyper-connection weights.

## Run it

The serving config is try 11: CUDA graphs, DSpark k=5, vision, 300K context, gmu 0.80. The head is Bluey (an NFS client); Reddie, which holds the checkpoint and exports it over NFS, is rank 2 and reads its local copy. Scripts are in `exl3/try4/` unless noted.

1. **Model.** On Reddie, verify the 40 EXL3 shards and build the 72-head TP3 copy with hardlinks: `bash exl3/tools/reddie_prep_tp3.sh`. The other two Sparks read it over NFS at `/mnt/reddie-models`.
2. **Engram rows.** Each rank keeps its 8 hash columns on local disk (`tools/engram_local.py`, ranges under "What TP3 needed"); rank 2 reads its own model directory.
3. **Patches.** Copy `patch/exl3-tp3/` to `~/patches/dsv41-exl3-tp3e/` on all three Sparks and check `MD5SUMS.txt`. `virtual_heads.py` carries an optional per-worker stack and memory sampler, on only when `KAI_STACK_SAMPLER=1` (the go scripts here set it; it samples every 5 s).
4. **Scripts.** `dsv41-tp3b.sh` (per-rank launcher) to `~/` on all three Sparks. On Asusi: `boot_dsv41_tp3b.sh`, `prep_launch_try11.sh`, `exl3tp3a11-go.sh`, `memguard2.sh`, `servguard.sh`, `flusher2.sh`, `memstat.sh`, `bootstatus.sh`. On Bluey: `post_boot2.sh`, `pb_wait.sh`, `vision_after.sh`, `bench/v41bench.py` and `tools/vision_tools_demo.py`.
5. **Launch** from Asusi: `bash ~/prep_launch_try11.sh`. It stops any running `vllm_dsv41` on all three (head first, logs saved), checks the NFS mount and Engram rows, drops caches, runs a 10 s GPU burn on each Spark, starts the page-cache flushers and the load-time memory guard, then fans out rank 2, rank 1 and rank 0. Serving about 12 minutes later.
6. **Once serving.** Stop the flushers (`touch ~/flusher.stop` on each Spark) and keep `servguard.sh` running: at gmu 0.80 the head runs with about 8 to 13 GiB free (see "Memory while serving at gmu 0.80").

## Bring-up log

| boot | outcome | cause | fix |
|---|---|---|---|
| exl3tp3a #1 (2026-09-11 11:44 ET) | died in weight loading | the padding shim sized the pad from vLLM's parameter shape; V4.1's attention linears load through ModelOpt MXFP8 static, which keeps scales per row (1x32) while the checkpoint stores 32x32 blocks (`wo_a.scale` [256, 128] vs a [3072, 128] param) | pad in checkpoint units (length / 8 real groups x 9), real counts from `virtual_heads_from` in the TP3 config; verified with vLLM's own loaders on a GPU, 30/30 (`exl3/tools/test_vh_loaders.py`) |
| exl3tp3a #2 (11:54 ET) | weights loaded clean on all 3 ranks, then unified-memory exhaustion: head rebooted, both workers hung | about 78 GiB of weights per Spark (text-only, no DSpark) plus vLLM startup plus page cache from reading the checkpoint | not retried yet. Next attempt: page-cache flusher during load (bot-lab-21 drops caches whenever Cached passes 40 GiB), lower `--gpu-memory-utilization`, eager first |
| exl3tp3a3 (12:35 ET, after a full power cycle) | stopped cleanly by the memory guard at 12:39 when the head reached 4 GiB free; workers never went under 14 GiB | the head (Reddie) carries two loads the workers do not: its own loader keeps ~19 GiB of mmapped shards resident while loading, and as the NFS server it caches both workers' ~86 GB reads; `gpu-memory-utilization` 0.75 cannot help because this happens before the KV allocation | next: head on a node with its own local copy, so the NFS server serves one worker and is not the head; keep `memguard.sh`. Superseded: tries 4 and 5 found the real cause (below) |
| exl3tp3a4 (head moved to Bluey; Reddie a worker reading its local copy) | stopped cleanly by the memory guard when Reddie reached 4 GiB free | per-process capture: the worker's own anonymous memory was 24.5 GiB, not page cache, so neither the flusher nor moving the head could help. Offline on one Spark: host-to-GPU copies out of a safetensors mmap leave anonymous memory about equal to the bytes copied, held until that file's tensors are released; glibc malloc tuning had no effect | find what keeps the files alive: a stack and memory sampler in every worker |
| exl3tp3a5 (13:22 ET, diagnostic: try 4 plus the sampler) | stopped by hand once the cause was on screen; clean, 116 GiB free after | the wrapper's `load_weights` (`nvidia/vl_model.py`) runs `sorted(...)` over every mapped checkpoint tensor before loading: the head's 48/48 shard bar finished in 36 s (the sort consumed it) while a worker's anonymous memory climbed 5 to 21 GiB in 100 s inside the EXL3 expert loader | stream the language-model weights (item 5 above) |
| exl3tp3a6 (13:30 ET: try 5 plus the streaming fix; sampler still on) | **up at 13:40 ET**; count, tool-call and coherence smokes pass | worker anonymous memory stayed between 1.9 and 3.7 GiB for the whole load. Weights took 129 s on Reddie (local disk) and 309 s and 470 s on Asusi and Bluey (NFS), 79.9 GiB per Spark. At gmu 0.75 the KV pool is 665,230 tokens (2.2 requests at 300K), with 15 to 19 GiB free per Spark after sizing | one change per boot from here: CUDA graphs (try 7), then DSpark, then vision |
| exl3tp3a7 (13:55 ET: try 6 plus CUDA graphs, FULL_AND_PIECEWISE, capture sizes 1 to 8) | **up at 14:05 ET** | weights 130 s on Reddie, 308 s and 474 s on Asusi and Bluey, as in try 6. Graph capture took 6 s for the sizing pass and 3 s for the final pass, at most 1.1 GiB per Spark. KV pool 759,557 tokens (2.5 requests at 300K), set by the tightest rank (2.89 GiB on Bluey); 17 to 19 GiB free per Spark once serving | next: DSpark k=5 (try 8) |
| exl3tp3a8 (14:16 ET: try 7 plus DSpark k=5) | failed at load: clean exit on Reddie, the other two ranks stopped by hand | the DSpark drafter builds a standard DeepSeek-V4 MoE layer (128 routed experts, MXFP4), and that layer's init requires the expert count to divide by the TP size: 128 is not divisible by 3 (`deepseek_v4/nvidia/model.py`, `_init_fused_moe_experts`). The main model's 384 experts divide by 3 | under investigation: relax the check if the per-rank expert range is only bookkeeping with expert parallelism off, or pad the drafter to 129 experts with a dummy one the router can never select. Try 9 goes on without DSpark |
| exl3tp3a9 (14:24 ET: try 7 plus vision, encoder replicated per rank) | **up at 14:35 ET**; smokes and boot 10's vision and tool checks pass (7/7) | the streaming loader's second pass loaded the vision encoder and aligner (80.3 GiB per Spark, 0.36 GiB more than text-only). KV pool 417,333 tokens (1.4 requests at 300K): the vision profiling reserve leaves the tightest rank 1.59 GiB of KV. 18 to 20 GiB free per Spark serving | try 10: gmu 0.80 (boot 10's value) for context and DSpark headroom |
| exl3tp3a10 (14:45 ET: try 9 plus gmu 0.80, boot 10's value; patch set tp3e) | **up at 14:57 ET**; smokes pass | **KV pool 1,995,725 tokens** (6.65 requests at 300K; 7.6 to 8.0 GiB of KV per Spark) against 417,333 at gmu 0.75 and boot 10's 1,070,168 on 4 Sparks. Load times and 80.3 GiB per Spark as in try 9, so tp3e's relaxed MoE check is a no-op for the main model as intended. 11.9 to 13.1 GiB free per Spark serving (8 GiB at the lowest, during the final graph pass); a serving-phase guard (`exl3/try4/servguard.sh`) stops all three below 4 GiB | try 11: DSpark k=5 |
| exl3tp3a11 (15:09 ET: try 10 plus DSpark k=5, patch set tp3e) | **up at 15:22 ET**: the drafter loads past try 8's check on all three ranks | model plus drafter 84.2 GiB per Spark (the drafter costs 3.9 GiB per rank; it loads in 29 s locally, 87 to 91 s over NFS). Graph capture 15 piecewise + 8 full sizes, at most 1.7 GiB. **KV pool 678,950 tokens** (2.26 requests at 300K; 2.62 GiB of KV on Bluey): DSpark costs about two thirds of try 10's context. 8 to 13 GiB free per Spark at startup | **serving config**; benchmark below |

Two deaths = stop and regroup (this repo's rule). After boot #2 the call was to keep going on TP3; every try since runs with a memory guard that stops all three containers before any Spark runs out. The TP4 EXL3 lane (exl3tp4b, four Sparks, no KV pin) is further down.

### Memory while serving at gmu 0.80

Measured on the head (Bluey) during try 10's benchmark. Free memory held at about 11.5 GiB through C1 to C6. When the cold-prefill sweep started (prompts up to 93K tokens), it stepped down about 3.3 GiB to about 8 GiB and then held; the worker's own memory stayed flat at 3.3 GiB, so the growth is GPU-side, beyond what the startup profile measured. Separately, Engram row reads grew the page cache from 6 to 10 GiB and pushed hard free memory (MemFree) down to 3.7 GiB, the zone where GB10's GPU allocator can stall. One cache drop brought MemFree back to 8 GiB. Every serving boot here keeps `exl3/try4/servguard.sh` running (stops all three below 5 GiB free).

## Results

Measured with `bench/v41bench.py`, prompt set v1, the same prompts as boot 10. Full tables per boot are in `results/<boot>/report.md`.

### exl3tp3a6: TP3, eager, no speculative decoding (the starting point)

Text-only, gmu 0.75, 300K context, KV pool 665,230 tokens. Throughput across the 8 prompt categories (the counting ceiling excluded):

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) | boot 10 aggregate tok/s |
|---|---|---|---|---|
| C1 | 16.9 | 17.4 | 0.26 | 38.0 |
| C2 | 32.7 | 17.1 | 0.34 | 64.3 |
| C3 | 46.7 | 16.6 | 0.40 | 78.7 |
| C4 | 59.1 | 16.2 | 0.68 | 85.7 |
| C5 | 75.7 | 16.4 | 0.48 | 114.2 |
| C6 | 90.6 | 16.4 | 0.51 | 131.9 |

Cold prefill (unique prompt, 1-token reply):

| prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|
| 2,950 | 3.49 | 845 |
| 11,592 | 9.83 | 1,180 |
| 46,810 | 39.58 | 1,183 |
| 93,335 | 75.59 | 1,235 |

Boot 10 (902 to 1,539 tok/s prefill) runs CUDA graphs and DSpark on 4 Sparks, so this is not a like-for-like comparison. Per-stream speed stays flat from C1 to C6, which points at per-step launch overhead; CUDA graphs are the next boot.

### exl3tp3a7: TP3, CUDA graphs, no speculative decoding

Try 6 plus CUDA graphs (FULL_AND_PIECEWISE, decode capture sizes 1 to 8). KV pool 759,557 tokens. Throughput across the 8 prompt categories (the counting ceiling excluded):

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) | vs eager (try 6) | boot 10 aggregate tok/s |
|---|---|---|---|---|---|
| C1 | 25.4 | 26.5 | 0.25 | 1.50x | 38.0 |
| C2 | 45.0 | 23.9 | 0.32 | 1.38x | 64.3 |
| C3 | 61.0 | 21.9 | 0.38 | 1.31x | 78.7 |
| C4 | 73.5 | 20.2 | 0.67 | 1.24x | 85.7 |
| C5 | 90.2 | 19.6 | 0.47 | 1.19x | 114.2 |
| C6 | 108.8 | 19.8 | 0.49 | 1.20x | 131.9 |

Cold prefill: 846, 1,174, 1,228 and 1,206 tok/s at 2,950, 11,592, 46,810 and 93,335 prompt tokens, the same as eager: prefill batches are larger than the captured sizes, so they run outside the graphs. Full tables in `results/exl3tp3a7/report.md`; side by side with try 6 in `results/exl3tp3a7/compare-a6-a7.md`.

### exl3tp3a9: TP3, CUDA graphs, vision on

Try 7 plus vision (4 images per request, vision encoder replicated on every rank with `--mm-encoder-tp-mode data`). KV pool 417,333 tokens. Boot 10's vision and tool-calling checks (`tools/vision_tools_demo.py`): 7 of 7 pass. Throughput across the 8 prompt categories:

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) | vs try 7 (text-only) |
|---|---|---|---|---|
| C1 | 25.7 | 26.8 | 0.25 | 1.01x |
| C2 | 44.2 | 23.6 | 0.33 | 0.98x |
| C3 | 60.5 | 22.1 | 0.59 | 0.99x |
| C4 | 78.1 | 21.1 | 0.42 | 1.06x |
| C5 | 89.4 | 19.5 | 0.48 | 0.99x |
| C6 | 107.5 | 19.6 | 0.49 | 0.99x |

Vision costs no throughput; it costs context (417,333 tokens against 759,557 text-only at the same memory setting). Cold prefill: 732 tok/s at 2,950, 1,156 tok/s at 11,592, 1,204 tok/s at 46,810, 1,236 tok/s at 93,335 (prompt tokens). Full tables in `results/exl3tp3a9/report.md`.

### exl3tp3a10: TP3, CUDA graphs, vision on, gmu 0.80

Try 9 at boot 10's memory setting (gmu 0.80 instead of 0.75). **KV pool 1,995,725 tokens**, 6.65 requests at 300K (boot 10 on 4 Sparks: 1,070,168). Vision and tool checks: 7/7 PASS. Throughput across the 8 prompt categories:

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) | vs try 9 (gmu 0.75) |
|---|---|---|---|---|
| C1 | 24.2 | 25.4 | 0.28 | 0.94x |
| C2 | 39.3 | 21.7 | 0.59 | 0.89x |
| C3 | 57.5 | 20.9 | 0.40 | 0.95x |
| C4 | 71.7 | 19.5 | 0.49 | 0.92x |
| C5 | 87.6 | 19.3 | 0.52 | 0.98x |
| C6 | 106.1 | 19.3 | 0.50 | 0.99x |

Within about 5% of try 9 except C2 (0.89x), not yet explained. Cold prefill: 712 tok/s at 2,950, 1,181 tok/s at 11,592, 1,147 tok/s at 46,810, 1,155 tok/s at 93,335 (prompt tokens). Full tables in `results/exl3tp3a10/report.md`.

### exl3tp3a11: the TP3 serving config (CUDA graphs, DSpark k=5, vision, gmu 0.80)

Try 10 plus DSpark. KV pool 678,950 tokens (2.26 requests at 300K). Vision and tool checks: 7/7 PASS. Throughput across the 8 prompt categories:

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) | vs try 10 (no DSpark) | boot 10 aggregate tok/s |
|---|---|---|---|---|---|
| C1 | 46.0 | 51.5 | 0.30 | 1.90x | 38.0 |
| C2 | 73.4 | 43.6 | 0.63 | 1.87x | 64.3 |
| C3 | 100.1 | 38.5 | 0.41 | 1.74x | 78.7 |
| C4 | 118.4 | 35.2 | 0.44 | 1.65x | 85.7 |
| C5 | 134.4 | 31.7 | 0.46 | 1.54x | 114.2 |
| C6 | 152.9 | 29.6 | 0.50 | 1.44x | 131.9 |

**Ahead of boot 10 (four Sparks, release checkpoint) at every concurrency level.** Cold prefill: 1,119 tok/s at 2,950, 1,186 tok/s at 11,592, 1,188 tok/s at 46,810, 1,199 tok/s at 93,335 (prompt tokens); boot 10 does 1,194 tok/s on the same 93,335-token prompt. The counting ceiling row, for reference only: 77.0 tok/s per stream at C1, 308.0 tok/s aggregate at C6. Boot 10 keeps more context (1,070,168 tokens against 678,950). Full tables in `results/exl3tp3a11/report.md`; DSpark's effect in `results/exl3tp3a11/compare-a10-a11.md`.

## TP4 lane: four Sparks on EXL3 (the context lane)

`exl3tp4b` is boot 10's serving config (four Sparks, head Reddie, CUDA graphs, DSpark k=5, vision, 300K per request, gmu 0.80) on the EXL3 checkpoint (`DeepSeek-V4.1-Flash-EXL3-Pollard`, 64 heads): image `vllm-dsv41:exl3a`, patch set tp3e (its TP3-only parts do nothing at TP4), boot 10's node-local Engram rows, no KV pin. Scripts are in `exl3/tp4/` (`prep_launch_tp4.sh` launches from Asusi). Launched 2026-09-11 15:45 ET, serving at 15:54. It is the default serving config since 2026-09-11: `bash /root/restore_exl3tp4b.sh` on Reddie as root restores it (`exl3/tp4/restore_exl3tp4b.sh` with `postcheck_exl3tp4b.sh`), and boot 10's `restore_boot10.sh` stays as the release-checkpoint fallback.

**KV pool 3,304,863 tokens (11.02 requests at 300K)**: 3.1x boot 10's 1,070,168 and 4.9x the TP3 lane's 678,950.

EXL3 splits the 2304-wide experts 512/640/640/512 at TP4, so the ranks are not equal:

| rank | Spark | expert slice | model memory, GiB | KV memory offered, GiB | free while serving, GiB |
|---|---|---|---|---|---|
| 0 (head) | Reddie | 512 | 56.6 | 28.0 | 19 to 23 |
| 1 | Spark4 | 640 | 68.9 | 15.6 | 7 to 12 |
| 2 | Asusi | 640 | 68.9 | 15.5 | 6 to 11 |
| 3 | Bluey | 512 | 56.6 | 28.4 | 19 to 24 |

vLLM sizes the pool from the tightest rank, so the two narrow ranks keep about 12 GiB unused. A KV pin (bot-lab-21 used 12 GiB) would not change how the pool is sized; it would trade some context for headroom on the two wide ranks.

Throughput across the 8 prompt categories, same bench as boot 10:

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) | vs boot 10 | TP3 lane aggregate tok/s |
|---|---|---|---|---|---|
| C1 | 41.6 | 46.4 | 0.37 | 1.10x | 46.0 |
| C2 | 62.2 | 37.5 | 0.77 | 0.97x | 73.4 |
| C3 | 101.1 | 38.2 | 0.44 | 1.29x | 100.1 |
| C4 | 106.7 | 30.9 | 0.51 | 1.24x | 118.4 |
| C5 | 129.2 | 30.4 | 0.56 | 1.13x | 134.4 |
| C6 | 141.2 | 27.7 | 0.58 | 1.07x | 152.9 |

Ahead of boot 10 at 5 of 6 concurrency levels and behind the TP3 lane on the same checkpoint at every level: on EXL3 the fourth Spark buys context rather than speed. Vision and tool checks: 7/7 PASS. Cold prefill: 1,326 tok/s at 2,950, 1,402 tok/s at 11,592, 957 tok/s at 46,810, 1,026 tok/s at 93,335 (prompt tokens); boot 10 does 1,194 tok/s on the same 93,335-token prompt. Full tables in `results/exl3tp4b/report.md`, with comparisons against boot 10 and the TP3 lane.

**Repeat runs (2026-09-11, about 90 minutes after startup).** Two more C1-C6 runs on the same serving config agree with each other and come in well above the first run:

| C | aggregate, first run | aggregate, repeat 1 | aggregate, repeat 2 | per-stream (first / r1 / r2) | mean TTFT s (first / r1 / r2) |
|---|---|---|---|---|---|
| C1 | 41.6 | 54.1 | 57.1 | 46.4 / 60.1 / 63.4 | 0.37 / 0.26 / 0.25 |
| C2 | 62.2 | 90.8 | 89.2 | 37.5 / 52.2 / 51.1 | 0.77 / 0.33 / 0.33 |
| C3 | 101.1 | 115.5 | 112.6 | 38.2 / 44.4 / 44.2 | 0.44 / 0.35 / 0.38 |
| C4 | 106.7 | 137.8 | 142.6 | 30.9 / 40.4 / 42.1 | 0.51 / 0.43 / 0.42 |
| C5 | 129.2 | 167.6 | 165.9 | 30.4 / 38.8 / 39.1 | 0.56 / 0.44 / 0.43 |
| C6 | 141.2 | 170.9 | 176.1 | 27.7 / 33.1 / 34.8 | 0.58 / 0.47 / 0.48 |

The first run was taken right after startup, while the two wide-slice ranks were down to 4-5 GiB of MemFree; the likely cause (not proven) is Engram pages being pushed out of the page cache, which slows decode. Coding and JSON, the first two C1 prompts, match across the first run and repeat 1 (coding about 81 tok/s in all three runs); every later prompt type was 30-60% slower in the first run only. The comparison with boot 10 above uses each config's first post-boot run; boot 10 has no repeat runs, and its drop-caches cron may affect it the same way, so the repeat runs are not a stock-vs-EXL3 claim. Files: `results/exl3tp4b-rep1/`, `results/exl3tp4b-rep2/`.

**Bench conditions.** The two wide ranks served with 7 to 12 GiB free. When the cold-prefill sweep began, their free memory stepped down about 3.7 GiB (the same long-context step as on TP3) while Engram row reads filled the page cache and pushed MemFree on Asusi down to 1.5 GiB. One-shot cache drops, then `exl3/tp4/memfree_flusher.sh` (drops clean cache when MemFree falls under 4 GiB), kept both ranks out of that zone; the 32K and 64K prefill runs overlapped those drops. `flusher2.sh`'s test (Cached minus Mapped minus Shmem) goes negative on these boxes because Mapped already counts the shmem mappings, so it never fired during serving; the next boots should use `memfree_flusher.sh` instead.

## Credits

bot-lab-21 (the EXL3 checkpoint and the V4.1 cuda-exl3 files), WestWaters/pollard-weights (the Pollard method), Zeuss5/cuda-exl3 (the plugin and kernels), Luke Alonso (the TP3 virtual-heads idea for MiniMax-M3), and this repo's boot 10 recipe underneath all of it.
