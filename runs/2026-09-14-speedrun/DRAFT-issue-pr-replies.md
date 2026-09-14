# Drafts for the 11:00 UTC (7 AM ET) issue/PR pass. Re-read each thread for new comments before posting.

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
- Separately: `expandable_segments:False` (your finding) is on our list. Tonight's run found `NCCL_MAX_NCHANNELS=8` gives +7% KV on the EXL3 lane. (Link the speed-run folder.)

## Issue #6 (DeepSeek Harness resends old images, 400 at 4 images), reply
- **Diagnosis:** your read is right. The server enforces `--limit-mm-per-prompt {"image":4}` per request. A client that re-sends every image from the session history on each turn crosses the limit even on a text-only turn.
- **Recommended fix: client side.** Drop old `image_url` blocks from history, or keep only the last N images. It keeps prompts smaller and TTFT lower too.
- **Server-side option:** the limit can be raised (e.g. `{"image":8}`) at the cost of a larger vision profiling reserve, which reduces the KV pool. We'd rather not raise it on the default config.
- Leaving open for the Harness maintainers. Suggest filing it there too.

## PR #4 (repo-relative diffs + patch/full + verifier)
- **Verify first:** local vLLM checkout at `e47aa780b`, run `patch/verify_diffs.sh`. If all 7 + 4 pass → merge (squash) with thanks.
- **Note:** `patch/full/engram.diff` targets the boot 10 `engram.py` (c0329107). Tonight's fast Engram path lives in `runs/2026-09-14-speedrun/patches/` and does not change `patch/`.

## PR #5 (verify5 false MISS)
- The logic is good and matches what hyudryu saw (MISS on a built module).
- It adds `build/__pycache__/verify5.cpython-314.pyc`. Ask to drop it, or merge and remove it in a follow-up commit plus a `.gitignore` for `__pycache__`.
- Merge if the pyc is gone. Otherwise merge with a follow-up commit removing it (credit im0xMagnus).

## PR #7 (docs: Community ports line for im0xMagnus's 8x Spark TP8 port), new at 03:58Z
- Two added lines in README under Sister repos, no code. Links a public repo that carries this recipe to eight ranks on dealignai's UNCENSORED-FP8.
- **Plan:** merge (squash) with thanks. It credits a community port and matches our credit-everyone rule. Check the link resolves before merging.
