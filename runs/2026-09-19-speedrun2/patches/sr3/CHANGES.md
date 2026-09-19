# dsv41-exl3-sr3 — sr2roce + two upstream DSpark fixes (2026-09-19)

sr3 = byte-identical copy of `~/patches/dsv41-exl3-sr2roce/` (the LIVE set) plus
the changes below. Every sr2roce change (SM12x 64-state pages, indexer TP split,
disk Engram + fast staging, RoCE port) is untouched. Base image stays
`vllm-dsv41:exl3b-roce` (vLLM nightly `8a728663c`, `0.28.1rc1.dev388+g8a728663c`).
Nothing here was distributed to other nodes and nothing was booted on a GPU.

Files that differ from sr2roce: `flashinfer_sparse.py`, `sparse_swa.py` (fix A),
`dspark.py` (fix B, NEW mount), `mounts.txt` (one added line), this file.

Upstream paths are `vllm/models/deepseek_v41/...`; our image (and mounts) use
`vllm/models/deepseek_v4_1/...`. Same files, renamed directory.

## Toggles (env vars, read once at module import; workers inherit the env)

| Fix | Env var | Default | Set to get today's (sr2roce) behaviour |
|---|---|---|---|
| A | `DSV41_DSPARK_ACCEPT_FIX` | `1` | `DSV41_DSPARK_ACCEPT_FIX=0` |
| B | `DSV41_DSPARK_KV_ONLY` | `1` | `DSV41_DSPARK_KV_ONLY=0` |

`DSV41_DSPARK_KV_ONLY` also accepts `lite` (see fix B, NOT upstream).

Boot note: the new `dspark.py` line in `mounts.txt` must become a `-v` mount in
the go script (I did not touch any go script). Without that mount, fix B is
simply absent and fix A is unaffected.

---

## Fix A — DSpark non-causal window / acceptance (upstream PR #57432)

* Upstream: vllm-project/vllm PR #57432 "[Bugfix][DSv4.1] Fix FlashInfer DSpark
  non-causal attention", merged 2026-09-17T22:14Z, merge commit
  `80447d27655918da6bfbccd0d3a40e975bda220a` (head `6554023f`, base `9612f770`).
* Upstream files: `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py`,
  `vllm/v1/attention/backends/mla/sparse_swa.py`, plus a new test
  (`tests/v1/attention/test_dspark_noncausal_sparse_mla.py`, not ported: SM100-only).
* Our files: `flashinfer_sparse.py` (already mounted), `sparse_swa.py` (already mounted).
* Pre-image check: the three hunk regions in our sr2roce copies are identical to
  the upstream pre-image (`9612f770`); our other edits in those files are
  elsewhere (SM12x block-size classes, `get_swa_block_size`). Port was mechanical
  apart from the toggle guards.

Hunks, in words:

1. `flashinfer_sparse.py` imports: add `CommonAttentionMetadata` to the
   `vllm.v1.attention.backend` import; add `import os`; add the module constant
   `_DSV41_DSPARK_ACCEPT_FIX`.
2. `DeepseekSparseSWAFlashInferMetadataBuilder`: new `__init__` allocating two
   persistent int32 `[max_num_batched_tokens]` device buffers (graph-stable
   addresses), and a `build()` override that, for a NON-causal batch with decode
   tokens, fills them with `clamp(decode_swa_lens, min=window_size)` and
   `seq_lens[token_to_req_indices]` and publishes them as
   `metadata.flashinfer_decode_topk_lens` / `flashinfer_decode_seq_lens`.
   Guard: with the toggle off, `__init__` allocates nothing and `build()` returns
   the parent's metadata untouched.
3. `DeepseekV4FlashInferMLAAttention._forward`, decode branch: when
   `decode_swa_width > window_size` (DSpark non-causal), pass the per-token
   lengths above instead of `sparse_topk_lens[:n]` / `seq_lens[:num_decodes]`,
   present the query/output as `[tokens, 1, H, D]` (one query per row),
   `cum_seq_lens_q=None`, `max_q_len=1`, so TRTLLM-gen cannot derive a causal
   SWA length from the query position and padded index slots never enter the
   softmax. Guard: toggle off -> exactly the previous call (the refactor into
   `decode_*` locals is behaviour-preserving).
4. `sparse_swa.py` `DeepseekSparseSWAMetadata`: two new optional fields
   `flashinfer_decode_topk_lens`, `flashinfer_decode_seq_lens` (default None,
   not toggle-guarded: pure dataclass fields).

**NOT MECHANICAL / RISK NOTE (read this):** on our GB10s (SM121) the model
selects `DeepseekV4FlashInferSM120Attention` (`model.py:_select_dsv4_attn_cls`,
`device_capability.major == 12`). Its `_forward_decode` calls FlashInfer's
`"sparse"` backend (`flashinfer/mla/_core.py:_trtllm_batch_decode_sparse_mla_dsv4_sm120`,
FlashInfer 0.7.0rc1) with `swa_topk_lens=decode_swa_lens` per token and never
passes `cum_seq_lens_q`/`seq_lens`/`sparse_topk_lens`; the SM120 dispatcher
documents that it "processes the query per token and per-token sparse indices
fully determine visibility". The non-causal `decode_swa_lens` written by
`ComputeDSparkNoncausalSWAIndicesKernel` are already the exact visible count
(trailing window + the full draft block) with `-1` padding beyond. So the two
defects PR #57432 fixes (padding in softmax; causal SWA length derived from
query position) exist only on the SM100 TRTLLM-gen path (`_forward` in
`DeepseekV4FlashInferMLAAttention`), which we never execute. Expected effect on
SM121: acceptance unchanged; cost = one `clamp` + one `index_select` per draft
build (hunk 2 runs because the SM120 backend inherits this builder). The
reported +21 % accepted tokens/step is an SM100 measurement. The port is still
correct and complete so it is ready if a TRTLLM-gen path is ever used.

CPU functional probe of hunk 2 (same mock as the upstream test, minus the
kernel): values match `clamp_min(128)` and `seq_lens[token_to_req]`, buffer
addresses persist across builds, causal builds leave the fields None; with
`DSV41_DSPARK_ACCEPT_FIX=0` no buffers are allocated and the fields stay None.
Script: `/tmp/sr3-work/build_probe.py`.

---

## Fix B — KV-only DSpark context insertion (upstream PR #56441)

* Upstream: vllm-project/vllm PR #56441 "[Perf][DSpark] Add KV-only context
  insertion across V4.1 cache formats", merged 2026-09-16T04:34Z, merge commit
  `6ecd97f1b7fb13991065370f6d6be3dffe9fe1db` (head `3b730dcc`, base `38ca7a89`).
* Upstream files: `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`
  (new op `fused_deepseek_v4_kv_rope_insert` + a 0-head grid fix),
  `csrc/libtorch_stable/ops.h`, `csrc/libtorch_stable/torch_bindings.cpp`,
  `vllm/models/deepseek_v41/nvidia/dspark.py`, plus a new kernel test.
* Our file: `dspark.py` = the image's
  `vllm/models/deepseek_v4_1/nvidia/dspark.py` copied into sr3 and added to
  `mounts.txt` (NEW mount). The image copy differs from the upstream pre-image
  in `_insert_context_kv` (our compiled op has no `kv_mxfp8` argument and takes
  `apply_q_norm=True` by default), and in an unrelated `sp_all_gather` ordering;
  both kept as in the image.

Hunk, in words: `_insert_context_kv(attn, kv, positions, slot_mapping)` — instead
of allocating a zero dummy query `[n_ctx, n_local_heads, 512]` and running the
Q+KV fused insert op (discarding the Q output), call the KV-only op
`torch.ops._C.fused_deepseek_v4_kv_rope_insert(kv, cache, slot_mapping,
positions, cos_sin_cache, block_size, fp8_scale_or_None, kv_mxfp8)`. Ported with
`getattr(attn, "kv_mxfp8", False)` because our attention layer has no such
attribute (584-byte fp8_ds_mla rows).

**BLOCKED ON THE IMAGE (label: could not be fully ported):** the op is a new
compiled kernel. `strings /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so`
has 0 hits for `fused_deepseek_v4_kv_rope_insert` (the three existing
`fused_deepseek_v4_qnorm_rope_kv_rope_*_insert` ops are there, with the schemas
quoted in the log). The image has no `csrc/` tree (`/src` is empty), so it
cannot be rebuilt in place. Therefore, on `vllm-dsv41:exl3b-roce`:

* `DSV41_DSPARK_KV_ONLY=1` (default): probes `hasattr(torch.ops._C,
  "fused_deepseek_v4_kv_rope_insert")` once at first context insert; it is
  False on this image, so it logs ONE warning and falls back to today's
  dummy-query path. Functionally identical to sr2roce. The Python side is in
  place so a rebuilt `_C_stable_libtorch` (or a side-loaded extension
  registering the op into `_C`) is picked up without further edits.
* `DSV41_DSPARK_KV_ONLY=0`: today's path unconditionally, no probe, no warning.
* `DSV41_DSPARK_KV_ONLY=lite` (**NOT UPSTREAM, my addition, opt-in**): keeps
  the existing `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` op but
  shrinks the discarded dummy query from `n_local_heads` (16 at TP4, padded 16)
  to 8 heads with `q_head_padded=8`. Legal per the op's checks
  (`q_head_padded >= q_in.size(1)`), and the 8-head template
  (`kNumHeadsQPadded=8`, both `apply_q_norm` variants, bf16) is compiled into
  our `.so` (verified by `strings`). Halves the dummy Q work and the internal
  padded-q allocation; the KV branch of the kernel is independent of the Q
  head count (that independence is exactly what upstream's 0-head op relies
  on). Only the uint8 (fp8_ds_mla) path is affected; bf16/fp8 caches behave as
  `0`. Untested on GPU. This is a partial stand-in, not the upstream gain
  (+12.8-14.9 % on H100 came from removing the Q work entirely).

To actually get fix B: rebuild `_C_stable_libtorch.abi3.so` from vLLM at
`6ecd97f1` (or apply the three csrc hunks of `/tmp/sr3-work/upstream/pr56441.diff`
onto the image's commit `8a728663c` and rebuild) for `sm_121a`, or build a
standalone stable-ABI extension from the post-image `.cu`
(`/tmp/sr3-work/upstream/kernel.post.cu`; it needs `torch_utils.h`,
`cuda_compat.h`, `dispatch_utils.h`, `type_convert.cuh` from the same tree) that
registers `fused_deepseek_v4_kv_rope_insert` in the `_C` namespace, and load it
before the first draft step. Either is a separate, GPU-tested change.

---

## Verification log (all commands run on this host, no GPU)

AST: `cd ~/patches/dsv41-exl3-sr3 && for f in flashinfer_sparse.py sparse_swa.py dspark.py; do python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read()); print('ast ok', sys.argv[1])" $f; done`
-> `ast ok` for all three.

Import test: `/tmp/sr3-work/import_test.sh <patchdir> [-e VAR=val ...]` builds
`-v <patchdir>/<file>:<site-packages target>:ro` from `mounts.txt` (same mapping
as the live container) and runs
`sudo docker run --rm --entrypoint bash <-v ...> vllm-dsv41:exl3b-roce -c "python3 /tmp/import_probe.py"`
which imports `vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse`,
`vllm.v1.attention.backends.mla.sparse_swa`,
`vllm.models.deepseek_v4_1.nvidia.dspark`,
`vllm.v1.worker.gpu.spec_decode.dspark.utils`.

Results (full log: `/tmp/sr3-work/import_test.log` on this Mac-side scratch,
reproduced by re-running the script):

| mounts | env | result |
|---|---|---|
| sr2roce (baseline) | – | 4/4 IMPORT OK, files unpatched |
| sr3 | default | 4/4 IMPORT OK; ACCEPT_FIX=True, builder overrides present, metadata fields present, KV_ONLY='1' |
| sr3 | `DSV41_DSPARK_ACCEPT_FIX=0 DSV41_DSPARK_KV_ONLY=0` | 4/4 IMPORT OK; ACCEPT_FIX=False, KV_ONLY='0' |
| sr3 | `DSV41_DSPARK_KV_ONLY=lite` | 4/4 IMPORT OK; KV_ONLY='lite' |

Caveat: in the no-GPU container `_C_stable_libtorch.abi3.so` does not load at
all, so the probe's "kv-only op available: False" there is not evidence by
itself; the `strings` check on the `.so` is.

Upstream artefacts kept for review: `/tmp/sr3-work/upstream/` (both `.diff`
files, pre/post images of the touched Python files, the post-image `.cu`),
`/tmp/sr3-work/image/` (pristine copies from the image).

## md5sums (sr3)

```
14677555e9e18b1328f71b43af77681a  CHANGES.md
1069303d0b7a05cd2a7d2fc32257bb4e  attention.py
dd993a7356e051ecd437d8aa365f82b2  b12x_roce_all_reduce.py
71e4a8fa9756efbe5401046f361e9290  cuda_communicator.py
57150cd2f28d9ab86eaee7357514da56  dspark.py
cc6e6017d014d587a3ffbd7aab9afee6  dsv4_nvidia_model.py
e84c730545819d04e2ce2182f45cee84  engram.py
8567871021f5ee2d752842bcd652d288  envs.py
4e80dfba02d7c45b3d07c388201b9162  exl3_config.py
11faaa7a582c1039639457bfe249b6dd  exl3_moe.py
6a7bba7a9b72102c616cbf899f027741  flashinfer_sparse.py
f347a601585907a5463636cdbb8ae403  gpu_worker.py
0a14bee67f103f1d616c6044134c0bab  model_state.py
b4f66f0f36f271579c2532c8100e5f29  mounts.txt
fa13455fa4689b260d20e02257721212  parallel_state.py
f87bc894e244fcf865343cfa7427e19f  sparse_attn_indexer.py
cc73f08663a0ba2d72fca52339d42c90  sparse_swa.py
4b396a58aa980abbb3e805395f12fe06  virtual_heads.py
facb83002478baafe8689f2e7cde4981  vl_model.py
2ab4522b3da83aed97a0f938386fa81a  vocab_parallel_embedding.py
7e1027f15bc1f649bc3d2635e8556ee2  weight_utils.py
```

## diff -rq dsv41-exl3-sr2roce dsv41-exl3-sr3

```
Only in dsv41-exl3-sr3: CHANGES.md
Only in dsv41-exl3-sr3: dspark.py
Files dsv41-exl3-sr2roce/flashinfer_sparse.py and dsv41-exl3-sr3/flashinfer_sparse.py differ
Files dsv41-exl3-sr2roce/mounts.txt and dsv41-exl3-sr3/mounts.txt differ
Files dsv41-exl3-sr2roce/sparse_swa.py and dsv41-exl3-sr3/sparse_swa.py differ
```
