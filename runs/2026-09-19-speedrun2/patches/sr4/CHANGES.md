# dsv41-exl3-sr4 — speed run 2026-09-19 (Claude for Tony / Tech2Wild)

Exact copy of `~/patches/dsv41-exl3-sr2roce/` (all 19 files + mounts.txt) plus two
independent, env-toggled changes. Both default ON in the patched files; `=0` restores
today's (sr2roce) behaviour exactly. One new file (`default_loader.py`, taken verbatim from
the `vllm-dsv41:exl3b-roce` image and then patched) and one new line in `mounts.txt`.

| toggle | default | file | scope |
|---|---|---|---|
| `DSV41_ENGRAM_ROCE_GATHER` | 1 | `engram.py` | decode: Engram all-gathers via b12x RoCE one-shot |
| `DSV41_DRAFT_SHARD_FILTER` | 1 | `default_loader.py` | load: DSpark draft reads only the `mtp.*` shards |

Not touched: the running container, `~/patches/dsv41-exl3-sr2roce/`, go scripts, other nodes.
Not run: any GPU / multi-rank test (no `--gpus` allowed). Both changes were verified in a
no-GPU throwaway container with the sr4 files bind-mounted read-only per `mounts.txt`.

---

## CHANGE 1 — Engram all-gathers through the RoCE one-shot path (`DSV41_ENGRAM_ROCE_GATHER`)

**What.** `Engram.embed` (non-SP path, the one this deployment runs: no EP so
`use_sequence_parallel=False`) all-gathers this rank's staged rows `[T, L, D]` along
`dim=1` (heads). Verified in the image's `b12x/comm/roce/roce_oneshot.py:684-704`
(`should_all_gather`): the RoCE runtime accepts only a contiguous CUDA shard concatenated
along **dim 0 or the last dim**, `0 < nbytes <= max_gather_bytes` (16 MiB via
`VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB`). A 3-D tensor with `dim=1` is neither, so
`cuda_communicator.all_gather` (sr2roce `cuda_communicator.py:429-451`) refused it every
step and fell through to pynccl + `torch.empty` + `reshape/movedim(0,dim)/reshape` = an
extra output copy per Engram layer per step (two Engram layers: 1 and 14 -> two per step),
inside the FULL decode graph.

**Fix.** Gather the free 2-D view `[T, L*D]` along its last dim instead. Concatenating
`[T, L, D]` along dim 1 and `[T, L*D]` along dim -1 produce byte-identical buffers
(`out3[t, r*L + l, d] == out2[t, r*(L*D) + l*D + d]`), so the result viewed back as
`[T, tp*L, D]` **is** the dim=1 result — no movedim, no transpose, nothing to move back.
The RoCE kernel then takes the *direct layout* (`_direct_gather_layout`: `nbytes % 16 == 0`,
`data_ptr % 16 == 0`, last-dim row bytes `L*D*2 = 3072 % 16 == 0`) and writes the
concatenated layout straight into the output — no padded-gather scratch, no torch copy.

**Where.** `engram.py`
- `:645-672` toggle `_DSV41_ENGRAM_ROCE_GATHER` + helper `_all_gather_heads(rows, tp_size)`
  (`=0` -> `tensor_model_parallel_all_gather(rows, dim=1)` exactly as before).
- `:1327` `Engram.embed` non-SP path (the hot one): `rows = _all_gather_heads(rows, tp_size)`
  then the unchanged `rows[:, :n_hash_cols]`.
- `:1105` `ParallelEngramEmbedding.forward`: same helper for consistency. Not on the hot path
  here (nothing calls `embed_tokens(...)` directly under the V2 runner; `Engram.embed` reads
  `staged_rows`), so this is a no-op for the live config.
- The SP path (`dim=0` gather, `:1310`) already satisfied the RoCE rule and is untouched.

**Shard size at decode (verified numbers).** `engram_n_heads=8`, `engram_max_ngram_size=4`
-> `n_hash_cols = 24`; TP=4 -> `part_n_hash_cols L = 6`; `head_dim D = 256`; staged rows
are **bf16** (`staged_rows` dtype). Per-rank shard = `T * 6 * 256 * 2 = 3072 B/token`:
C8 x 6 tokens (5 spec + 1) = T=48 -> **144 KiB** (largest captured size, 48). Well under
16 MiB (the RoCE limit is hit only above T=5461 tokens, i.e. large prefill chunks, where
`should_all_gather` returns False and the pynccl fallback yields the identical tensor —
same as today for that regime). No size gate needed. `24 == 4*6`, so the trailing
`[:, :n_hash_cols]` slice is a no-op and `.flatten(-2)` in `Engram.forward` stays a view.

**CUDA-graph safety.** No host sync; shapes depend only on `T` (per capture size); the
only allocation is the gathered output (`torch.empty` in `roce_oneshot.all_gather` direct
path), exactly what the pynccl path also allocated. The RoCE gather launcher key is
dtype-independent (`_gather_launcher_key`) and `prepare()` (called from
`B12xRoceAllReduce.capture()` before every capture, `b12x_roce_all_reduce.py:277-287`)
compiles it — the logits gather `(40, 32320) bf16 along dim 1` (2-D last dim) already runs
through this exact path inside today's FULL graphs (live log: "RoCEnante all-gather is
live"). The padded-scratch path (refused under capture) is never entered because the rows
are 16-byte aligned. `rows.reshape(T, L*D)` is a view (staged_rows[:T] is a contiguous
dim-0 slice); the final `.reshape(T, tp*L, D)` is a view on the contiguous RoCE output and
on the pynccl fallback output (`reshape` there already materialises a contiguous tensor).

**Expected effect.** Two fewer `[T, 24, 256]` bf16 device copies + two NCCL all-gathers
replaced by RoCE one-shot gathers per decode step; the Engram gather joins the other
collectives already on RoCE. Small per-step latency win at C1..C8 (the gather is 3-144 KiB).

**Test result.** In the throwaway container (no GPU): imports OK; with a fake 4-rank
all-gather, `_all_gather_heads` output `torch.equal` to the dim=1 gather for
T in {1, 5, 6, 48, 4096}; the collective was issued on `(T, 1536)` contiguous, `dim=-1`;
with the toggle at 0 the collective was issued on `(T, 6, 256)`, `dim=1` (today's call).
Direct-layout conditions checked for every T. Not tested: the actual RoCE kernel on 4 ranks
(needs GPUs).

**Risk.** Low. Numerics identical by construction. The one thing to watch on the first
real boot: the "RoCEnante all-gather is live" line may now name the Engram shard
`(T, 1536) bfloat16 along dim 1` if it is the first routed gather (it fires before the
logits gather in the forward), which is expected, not a regression. If a padded-gather
error ever appeared under capture it would mean `staged_rows[:T].data_ptr()` was not
16-byte aligned, which cannot happen for a slice at offset 0 of a `torch.zeros` allocation.

---

## CHANGE 2 — DSpark draft reload reads only the `mtp.*` shards (`DSV41_DRAFT_SHARD_FILTER`)

**What.** Verified: `v1/worker/gpu/spec_decode/dspark/utils.py:73-77` calls
`get_model(vllm_config=draft_vllm_config, model_config=draft_model_config)` with the same
checkpoint dir as the target -> a fresh `DefaultModelLoader` walks all **49** shards again
(live log: second "Loading safetensors checkpoint shards ... 49/49 [00:32]", "DSpark draft
model loaded: 97 params", "Loading weights took 32.18 seconds" on the head; workers read
over NFS, ~59 s, head waits). `DSparkDeepseekV4ForCausalLM.load_weights`
(`models/deepseek_v4_1/nvidia/dspark.py:388-516`) consumes only `mtp.<i>.*` names —
`_remap_dspark_name` returns None for everything else (embed/lm_head are aliased from the
target, `has_own_embed_tokens = has_own_lm_head = False`). Per
`model.safetensors.index.json`, all 2401 `mtp.*` tensors live in **3** shards:
`model-00044/45/46-of-00048.safetensors`.

**Fix (smallest correct point).** `default_loader.py` (new to mounts; verbatim from the
image + this patch):
- `:41-70` toggle `_DSV41_DRAFT_SHARD_FILTER`, `_DSPARK_DRAFT_ARCHS =
  {DSparkV41DraftModel, DSparkDraftModel}`, `_draft_shard_prefix(model_config)` -> `"mtp."`
  only when the model config's architectures (or its `hf_config.architectures`) contain a
  DSpark DeepSeek draft arch **and** `hf_config.model_type in (deepseek_v4, deepseek_v41)`
  (that is exactly what `config/speculative.py:1392-1424` sets for the draft). The target
  (`DeepseekV41ForCausalLM`) and any other model get None -> untouched path.
- `:100` new optional `Source.only_shards_with_prefix` (default None, so every other
  `Source(...)` constructor, incl. `secondary_weights`, is unaffected).
- `:289-295` in `_get_weights_iterator`, **after** `_prepare_weights`: restrict
  `hf_weights_files`. Deliberately not via `allow_patterns_overrides`:
  `filter_duplicate_safetensors_files` (`weight_utils.py:580-606`) raises
  `FileNotFoundError` when an index-referenced shard is missing from the candidate list, so
  a pattern override would abort the load.
- `:362-400` `_filter_shards_by_tensor_prefix`: reads the index once, keeps the shards
  holding >= 1 `mtp.*` tensor. No index file or no match -> full list + one warning (safe
  fallback). Logs **one line**: `DSV41_DRAFT_SHARD_FILTER: draft reads 3 of 49 shards
  (those holding 'mtp.*' tensors): model-00044-of-00048.safetensors, ...`.
- `:413` `get_all_weights` passes `only_shards_with_prefix=_draft_shard_prefix(model_config)`.
- `mounts.txt`: added `default_loader.py model_executor/model_loader/default_loader.py`.

**Kept intact.** `weight_utils.py` is byte-identical to sr2roce: the Engram disk skip
(`.engram.embed.weight/.scale` skipped when `DSV41_ENGRAM_DISK=1`) and the EP filter
(`should_skip_weight(name, local_expert_ids)`, `_init_ep_weight_filter`) are per-tensor
checks inside the iterator and still run on the shards that are read. The target load is
unchanged (guard returns None).

**Expected effect.** Draft reload: 3 shards instead of 49 -> roughly 32 s -> ~2-3 s on
the head and ~59 s -> ~4-6 s on the NFS workers (the 3 shards are ~6% of the bytes and
the draft's tqdm will show `3/3`). Boot-to-ready shortens by about the worker figure since
the head waits for the slowest rank.

**Test result.** Throwaway container: import OK; guard returns `"mtp."` for
`DSparkV41DraftModel/deepseek_v41` and `DSparkDraftModel/deepseek_v4`, None for the target
arch, for `Qwen3DSparkModel`, and with the toggle at 0; against the real model dir
(mounted ro) the filter kept exactly 3 of 49 files (44, 45, 46); bogus prefix and missing
index both fall back to the full list; `Source("x", None)` still constructs.

**Risk.** Low. If a future checkpoint moved draft weights outside `mtp.*` the draft's own
`load_weights` would ignore them anyway. If the index were absent the change is a no-op
(warning). `DSV41_DRAFT_SHARD_FILTER=0` restores the 49-shard walk without touching any
other file.

---

## Verification summary

- `python3 -c "import ast; ..."` on `engram.py`, `default_loader.py`: OK.
- Throwaway `vllm-dsv41:exl3b-roce` container (no `--gpus`), all 19 sr4 mounts `:ro`
  per `mounts.txt`, model dir `:ro`, `DSV41_ENGRAM_DISK=1 DSV41_ENGRAM_FAST=1`:
  `python3 -c "import vllm.models.deepseek_v4_1.common.engram,
  vllm.model_executor.model_loader.weight_utils,
  vllm.model_executor.model_loader.default_loader"` OK, plus `/tmp/sr4-work/sr4_test.py`
  (identity + filter tests above) OK.
- `diff -rq` vs sr2roce: only `engram.py`, `mounts.txt` differ; `default_loader.py` new.
====
sparse_swa.py v1/attention/backends/mla/sparse_swa.py
flashinfer_sparse.py models/deepseek_v4_1/nvidia/flashinfer_sparse.py
engram.py models/deepseek_v4_1/common/engram.py
weight_utils.py model_executor/model_loader/weight_utils.py
model_state.py models/deepseek_v4_1/nvidia/model_state.py
sparse_attn_indexer.py model_executor/layers/sparse_attn_indexer.py
attention.py models/deepseek_v4_1/attention.py
virtual_heads.py models/deepseek_v4_1/virtual_heads.py
exl3_config.py /usr/local/lib/python3.12/dist-packages/cuda_exl3/config.py
exl3_moe.py /usr/local/lib/python3.12/dist-packages/cuda_exl3/moe.py
vocab_parallel_embedding.py model_executor/layers/vocab_parallel_embedding.py
vl_model.py models/deepseek_v4_1/nvidia/vl_model.py
dsv4_nvidia_model.py models/deepseek_v4/nvidia/model.py
b12x_roce_all_reduce.py distributed/device_communicators/b12x_roce_all_reduce.py
cuda_communicator.py distributed/device_communicators/cuda_communicator.py
parallel_state.py distributed/parallel_state.py
envs.py envs.py
gpu_worker.py v1/worker/gpu_worker.py
default_loader.py model_executor/model_loader/default_loader.py
