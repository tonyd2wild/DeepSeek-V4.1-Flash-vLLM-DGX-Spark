# DSV41_INDEXER_TP_SPLIT: split the lightning indexer's prefill rows across TP ranks

Draft patch for `model_executor/layers/sparse_attn_indexer.py`, the sr1 mounted version,
on vLLM dsv41-feat at e47aa780b. Only this one file changes. `attention.py` needs no edit,
because the op is called from `_sparse_indexer_and_attn` (sr1 attention.py:827) and the
change is entirely inside the op.

Status: **not run on a GPU yet.** Both files compile, and the row partition and reorder
logic passed a CPU check. Bit-exactness through the real kernels still has to be shown
with `test_idxsplit.py` on a GB10 before anyone enables the flag.

## Files

| File | What it is |
|---|---|
| `sparse_attn_indexer.py` | Full patched copy. Mount it at `vllm/model_executor/layers/sparse_attn_indexer.py`. |
| `sparse_attn_indexer.idxsplit.diff` | Unified diff against `src/dsv41-exl3-sr1/sparse_attn_indexer.py`. |
| `test_idxsplit.py` | Exactness test: `--mode sim`, `--mode spawn`, `--mode cpu-logic`. |

## Enabling it

| Env | Default | Meaning |
|---|---|---|
| `DSV41_INDEXER_TP_SPLIT` | `0` | `1` turns the split on. It must be identical on every node. |
| `DSV41_INDEXER_TP_SPLIT_MIN` | `512` | Minimum rows in an indexer prefill chunk before it gets split. |

With the flag off, the only runtime difference is one extra Python call per prefill chunk,
which returns 1 immediately. The other difference is at model build: every rank does one
gloo `all_gather_object` to compare the two env values, even with the split off.

## Where the code goes (line numbers in the patched file)

| Lines | Change |
|---|---|
| 5, 12-17 | `import os`, plus `get_tp_group` and `model_parallel_is_initialized` from `vllm.distributed` |
| 139-392 | New block after `_merge_dcp_topk_global`: env constants, `_dsv41_check_tp_split_env_consistent`, `_dsv41_tp_split_world` (the gate), `_dsv41_tp_split_bounds`, `_dsv41_prefill_rows_local`, `_dsv41_scatter_gathered`, `_dsv41_prefill_chunk_tp_split` |
| 755-777 | Prefill chunk loop, just before `if chunk.local_total_seq_lens == 0:` (sr1 line 494): when the gate says split, run the split path and `continue` |
| 1153-1154 | End of `SparseAttnIndexer.__init__`: the one-time env consistency check |

The decode path (sr1 lines 574-762) is unchanged.

## Data flow for one prefill chunk (rows R = token_end - token_start)

1. **Gate.** `_dsv41_tp_split_world` returns tp when all of these hold: the flag is on,
   R >= MIN, DCP is 1, PCP is off, the platform is CUDA and tp > 1. It asserts that the
   stream is not capturing a CUDA graph.
   - Every input to the gate is identical on all ranks, so all ranks take the same branch
     and issue the same collectives. The env is checked at build, and R comes from the
     broadcast scheduler output.
2. **Row slice.** `per = roundup(ceil(R / tp), 64)`. Rank r owns rows `[r·per, min(r·per + per, R))`.
   A rank can own nothing: with 520 rows at tp4, rank 3 is empty.
3. **Local compute.** `_dsv41_prefill_rows_local` runs the unsplit loop body on that slice,
   with the same kernels, in the same order, with the same per-row arguments:
   - `fp8_fp4_mqa_logits` on `q[t0+lo : t0+hi]`, the rank's `weights` rows, the rank's
     slice of `cu_seqlen_ks/ke`, and the whole gathered index K. The K gather itself stays
     replicated.
   - Candidate step, when the layer has one: `_select_candidate_blocks` on the source
     layer, or `_apply_candidate_mask` on the consumer layers (24/28/32/36).
   - `top_k_per_row_prefill`.
   - The output is `[per, 1152]` int32, pre-filled with -1 exactly like the shared buffer
     is at line 452 of sr1. On the candidate-source layer, the `[per, candidate_topk_blocks]`
     candidate ids are concatenated onto it.
4. **All-gather.** `get_tp_group().all_gather(local, dim=0)` goes through PyNCCL on the
   current stream and returns `[tp·per, W]` in rank order.
   - Padding rows, and ranks past the end, land at positions >= R, so `gathered[:R]` is
     the chunk's rows in their original order.
5. **Scatter.** Copy `gathered[:R, :1152]` into `topk_indices_buffer[t0:t0+R]`. On the
   candidate-source layer, also copy the candidate columns into `candidate_blocks[t0:t0+R]`.
   - Every other rank's later indexers mask only their own rows, but the full candidate
     buffer is gathered anyway, so nothing depends on the ratio-1 and ratio-2 layers
     chunking the same way.
6. **Downstream.** Sparse MLA derives the global indices and the lengths from the buffer
   (`compute_global_topk_indices_and_lens` in flashinfer_sparse.py). The buffer is
   byte-identical on every rank, so those derived values are too.

Multiple requests per chunk, and chunks that are query sub-slices, need no special
handling. Each row's `cu_seqlen_ks/ke` already points at its own request's range in the
gathered K, and slicing rows keeps those bounds.

## Correctness argument

**Every rank ends with the same bytes as every other rank.** They all copy from one
gathered tensor. That is stronger than today, where each rank runs its own top-k.

**Match with the unsplit path.** Each row goes through the same kernels, with the same
inputs, as it would unsplit. The indexer inputs (qr, latent, weights) come after the TP
all-reduce, so they are identical on every rank. The output is exact if each kernel's
result for a row depends only on that row's inputs:

| Kernel | Argument | Confidence |
|---|---|---|
| Logits | Each output element is a per-row dot product plus a fixed-order weighted sum over heads. The row block never reduces across rows. | Should be exact |
| Candidate block scores (Triton) | One program per row | Exact |
| `top_k_per_row_prefill` | Its source is **not** in the local tree. If it breaks ties or orders its output with atomics, results can differ in order even between two unsplit runs. | Unverified |
| `torch.topk` inside `select_candidate_blocks` | May pick a different algorithm when it gets R/tp rows instead of R, which only affects exactly tied block scores | Ties only |

Where the last two differ, they differ only among keys with exactly equal scores. That
does not change what the model attends to, only the floating-point summation order.

`test_idxsplit.py` separates the two cases. It runs the reference twice, calls a match
BYTE-EQUAL or SET-EQUAL (reference itself nondeterministic), and calls anything else
MISMATCH. `--ties` forces heavy exact ties.

## Verified vs not

- **Verified locally:** both files pass `py_compile`. `--mode cpu-logic` ran on the real
  `_dsv41_tp_split_bounds` / `_dsv41_scatter_gathered`, pulled out of the patched file,
  and passed 120 partition/reorder checks: tp 1/2/3/4/8, rows 1-8192, with and without
  candidates, including empty ranks, and confirmed rows outside the chunk stay untouched.
- **Not verified:** any GPU kernel call, the PyNCCL path, and a full server boot.
  - Run `--mode sim --quant both`, `--mode sim --ties`, `--mode spawn --tp 4`, and `--long`.
  - Set `--heads`, `--cand-k` and `--cand-block` from the config; the defaults are guesses.
  - Then check greedy output with the flag on and off.

## Cost and expected gain

**Divided by tp:** the logits GEMM, the candidate select/mask, and the row top-k. All of
these scale with rows × context.

**Still replicated:** the index-K gather (memory-bound, about 12 MB per layer per step at
93K), the Q projection, quant and `weights_proj`, none of which depend on context.

**All-gather per indexer layer per chunk** at 8192 rows and tp4:
4 × 2048 × 1152 × 4 B = **36 MiB gathered** (9 MiB sent and 27 MiB received per rank).
- The candidate-source layer adds 4 × 2048 × K × 4 B, which is 2 MiB at K=64.
- Sub-chunking under the 512 MiB logits budget splits this into several smaller gathers,
  with the same total.
- Over RoCE that is about 1.5-2.5 ms per layer per step, so about
  **L_idx × 12 × 2 ms ≈ 0.5 s** at 93K for 20 indexer layers.
- L_idx is `len(index_source_layer_ids)`. It is not available locally; it is at least 5
  (the candidate source plus 24/28/32/36).

**Load balance.** Row cost is roughly ke − ks, and the contiguous split gives the last
rank the longest rows.
- Summed over a 93K prompt (12 steps), the slowest rank does 0.266 of the unsplit work,
  a 3.77× speedup. At 46.8K it is 0.281, a 3.56× speedup.
- The first chunk alone only gets 8/7×.

**Projection** from the other agent's fit (indexer ≈ 35 s of 78.2 s at 93K, ≈ 9 s of 30.4 s at 46.8K):

| Prompt | Today | Projected | Saved |
|---|---|---|---|
| 93K | 78.2 s (1,194 tok/s) | ≈ 53 s (≈ 1,760 tok/s, +48%) | 25.7 s |
| 46.8K | 30.4 s (1,538 tok/s) | ≈ 24 s (≈ 1,950 tok/s, +27%) | 6.5 s |

These are estimates, not measurements.

## Risks

- **Hang if ranks disagree.** If the env differs across ranks, some ranks skip the
  collective and the others wait forever. The build-time check turns that into an error,
  but it only works if every rank runs this patched file.
- **Long context turns the split off.** A ratio-1 sub-chunk has about 134M / N rows, so
  above about 262K keys it has fewer than 512 rows and falls back to the unsplit path.
  Lower `DSV41_INDEXER_TP_SPLIT_MIN` (for example to 128) for long-context serving.
- **Ties** (see above), until the GPU test settles them.
- **CUDA graphs.** A split-sized chunk under graph capture trips an assertion instead of
  capturing NCCL. That is unreachable today, since captures are 48 tokens or fewer.
- **Transient memory.** About 45 MB per call, which is outweighed by logits 4× smaller per rank.
- **Imbalance.** Interleaving the rows across ranks would recover about 6% more at the
  cost of a permutation step. Not done here.
- **Scope.** DCP, PCP and XPU are left unsplit by design.
