# Drafts for the 11:00 UTC (7 AM ET) issue/PR pass. Re-read each thread for new comments before posting.

Reviewed heads (10:08 UTC), all MERGEABLE, 1 commit each. Merge only if the head is unchanged:
- PR #4 `9dc9d21ad011` (the sha koldfrontier verified), PR #5 `557df4ec43da`, PR #7 `42810ac1ed10`.
- Order: #4, #5 (then a follow-up commit on main: `git rm --cached build/__pycache__/verify5.cpython-314.pyc` plus `__pycache__/` in `.gitignore`), #7. Then comment on #1, #2 and #6.

## Issue #1 (GB10 slow state), comment
Update from a 2026-09-14 overnight run, plus thanks to @tmolteno and @im0xMagnus for the data points.
- **Tonight's measurements:**
  - The fleet (2 FE, 2 GX10) was power-cycled on 2026-09-13.
  - Three nodes report 2411 MHz at idle; the earlier clock lock is no longer in effect.
  - The EXL3 lane's idle test (first request after a long idle vs back to back) shows 99.9 vs 105-107 tok/s on counting. That's a ~5-7% gap, not the 1.5x slow mode we measured on 2026-09-10.
- **Consistent with @im0xMagnus's result:** 0 slow seconds on newer BIOS/driver/kernel, locked or unlocked; `-lgc 0,2200` acts as a power cap, not a fast-state holder.
- **What we'd still like:**
  - One of our FE nodes on BIOS `0ACUM027` + driver 580.173.02 to close this out. (One forum report says 580.173.02 breaks the GPU on OTA2607 firmware, so that needs care.)
- Leaving open until then.

## Issue #2 (garbage with CUDA graphs on a newer vLLM tree), comment (only if nothing new)
- The pin from #3 has been merged for a few days. No confirmation from a rebuild yet.
- @ecohash-co @hyudryu: if either of you rebuilt at `e47aa780b`, please post the result. Otherwise we'll close this as resolved by #3 in a week.
- Separately, from the 2026-09-14 speed run (runs/2026-09-14-speedrun):
  - `NCCL_MAX_NCHANNELS=8` (bot-lab-21) gave +7% KV pool on the EXL3 lane.
  - The b12x RoCE one-shot all-reduce (@original-el8, @lukealonso; local-inference-lab/vllm#597, ported onto dsv41-feat) was the biggest decode win: code +11%, prose +16% vs the same config on NCCL.
  - `expandable_segments:False` (your finding) is being measured on the EXL3 lane tonight (b1-noexpseg); result goes in the run folder.

## Issue #6 (DeepSeek Harness resends old images, 400 at 4 images), reply
- **Diagnosis:** your read is right. The server enforces `--limit-mm-per-prompt {"image":4}` per request. A client that re-sends every image from the session history on each turn crosses the limit even on a text-only turn.
- **Recommended fix: client side.** Drop old `image_url` blocks from history, or keep only the last N images. It keeps prompts smaller and TTFT lower too.
- **Server-side option:** the limit can be raised (e.g. `{"image":8}`) at the cost of a larger vision profiling reserve, which reduces the KV pool. We'd rather not raise it on the default config.
- Leaving open for the Harness maintainers. Suggest filing it there too.
- **New (04:20Z):** koldfrontier ran boot 10 with `{"image":16}`: six images -> 200 and the right answer, vision_tools_demo 7/7, decode unchanged, and the KV pool (1,109,543) was inside their boot-to-boot spread, since the profile run's dummy input is bounded by max_num_batched_tokens rather than the image limit.
- **Updated reply draft:** "Thanks @pavolbiely for the report and @koldfrontier for the 16-image data point. The client-side fix (trim old images before resending) is still the right one, and Open WebUI resends everything too, so a higher server limit is reasonable. We have not changed the default tonight because the speed run kept one change per boot; raising it to 8 or 16 is on the list for the next config boot, measured against the KV pool. Leaving this open until then."

## PR #4 (repo-relative diffs + patch/full + verifier)
- **Verify first:** local vLLM checkout at `e47aa780b`, run `patch/verify_diffs.sh`. If all 7 + 4 pass → merge (squash) with thanks.
- **Note:** `patch/full/engram.diff` targets the boot 10 `engram.py` (c0329107). Tonight's fast Engram path lives in `runs/2026-09-14-speedrun/patches/` and does not change `patch/`.
- **New (04:20Z):** koldfrontier independently verified `patch/full/*.diff` on a fourth 4x GB10 fleet: `git apply --check` clean at `e47aa780b`, results byte-identical to the boot-10 md5 set. Plus a map of what a rebase onto vLLM `main` (`eb42686a30cd`) means: the sparse_attn_indexer fix becomes a flag (`--sparse-indexer-topk-backend per_row`, from #56464), attention.py is a rename to `DeepseekV41IndexerBackend`, and engram.py needs a DISK storage backend behind upstream's new hooks (#56512).
- **Reply draft:** "Merged, thank you @im0xMagnus. And thanks @koldfrontier for the independent verification on a fourth fleet and for the rebase map against main; that is exactly what the next person to rebase needs, and it fits with what we saw tonight (the Engram disk layer is still required on GB10). The speed run's Engram changes live in runs/2026-09-14-speedrun/patches/ and do not touch patch/."

## PR #5 (verify5 false MISS)
- The logic is good and matches what hyudryu saw (MISS on a built module).
- It adds `build/__pycache__/verify5.cpython-314.pyc`. Ask to drop it, or merge and remove it in a follow-up commit plus a `.gitignore` for `__pycache__`.
- Merge if the pyc is gone. Otherwise merge with a follow-up commit removing it (credit im0xMagnus).

## PR #7 (docs: Community ports line for im0xMagnus's 8x Spark TP8 port), new at 03:58Z
- Two added lines in README under Sister repos, no code. Links a public repo that carries this recipe to eight ranks on dealignai's UNCENSORED-FP8.
- **Plan:** merge (squash) with thanks. It credits a community port and matches our credit-everyone rule.
- Link checked 08:29 UTC: `im0xMagnus/deepseek-v4.1-flash-uncensored-8x-dgx-spark` is public (created 2026-09-11, pushed 03:58Z) with LICENSE, README, bench, build, launch, results, tools.
- Reply draft: "Thanks @im0xMagnus, merged. Glad the recipe carried to eight ranks. Tonight's speed run (runs/2026-09-14-speedrun) found a b12x RoCE all-reduce win on 4 nodes that may help TP8 too; details and credits are in that folder."
