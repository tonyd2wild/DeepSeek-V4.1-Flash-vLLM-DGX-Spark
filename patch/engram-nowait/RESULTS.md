# Page-cache fast path for the blocking Engram reader

This optional experiment patches the recipe's canonical `patch/engram.py` blocking `preadv` fallback. It does not change the repository's deployed snapshot, file layout, dequantization, or the newer FAST path.

## Target and mechanism

`engram-nowait.diff` applies to `patch/engram.py` at recipe commit `d45538f67366c62da668c35ec1afd4fcd0e4c8b4`. Unlike the other recipe subfolder diffs that target the pinned vLLM checkout, this second-stage diff targets the recipe snapshot itself.

For every requested row, the caller first tries `os.preadv(..., os.RWF_NOWAIT)` synchronously. Rows that return `EAGAIN` are tried once more; only residual rows go to the existing blocking pool. Errors treated as unsupported (`EINVAL`, `ENOSYS`, `ENOTSUP`, and `EOPNOTSUPP`) run the established whole-batch reader. Partial reads keep reading. A zero return from NOWAIT is treated as a miss, because Linux 5.9 and 5.10 may return zero before EOF; the blocking reader remains responsible for terminal short-read detection.

The goal is to avoid creating futures and switching threads for rows already in the Linux page cache. This adds no user-space row cache or second copy of the Engram tables.

## Reproduce the forced-path checks

From the repository root, with Python 3.10 or newer:

```bash
git apply --check patch/engram-nowait/engram-nowait.diff
git apply patch/engram-nowait/engram-nowait.diff
python3 patch/engram-nowait/test_engram_nowait.py patch/engram.py
git apply -R patch/engram-nowait/engram-nowait.diff
```

The dependency-free harness extracts the functions from the patched file and exercises cached hit, one-`EAGAIN`, persistent-`EAGAIN`, partial-read, unsupported-operation, NOWAIT-zero fallback, and terminal-short-read behavior. It prints seven `PASS` lines and `7/7 checks passed` on success.

## Measured correctness evidence

The implementation family was tested on four DGX Sparks running Ubuntu 24.04.4 and Linux `6.17.0-1031-nvidia`. Production model rows were node-local. The corrected component harness used real files, offsets, and row sizes, and used `mincore` to classify residency without warming the measured sample.

- 3,000 real-offset reads matched the established reader byte for byte.
- Forced cached-hit, one-`EAGAIN`, persistent-`EAGAIN`, partial-read, unsupported-operation, and terminal-short-read cases passed.
- This upstream artifact adds the post-review NOWAIT-zero fallback described above; it is not claimed to be byte-identical to the deployed source.

The repository does not contain the original raw profiler streams, so the measurements below are reported evidence rather than a fully replayable benchmark archive.

## Component results

| Rank | 50% classified-cold | 100% classified-cold |
| ---: | ---: | ---: |
| 0 | 5.78x | 2.35x |
| 1 | 3.62x | 2.22x |
| 2 | 2.00x | 2.57x |
| 3 | 6.69x | 1.94x |

For an 8,192-token prefill-shaped cached workload (196,608 row reads across four logical jobs), median component time fell from 1,052.327 ms to 126.787 ms (8.30x). In that measured pair, voluntary context switches fell from 296,162 to zero.

## Production evidence and limits

| Fixed workload | Before | After |
| --- | ---: | ---: |
| C1 output rate | 12.07 tok/s | 35.49 tok/s |
| C4 wall aggregate | 28.49 tok/s | 62.96 tok/s |
| 13,316-token prompt TTFT | 5.889 s | 5.989 s |

A 24-hour production soak supplied stability evidence: about 3,921 completed requests, 1.064 billion prompt tokens, and 4.378 million generation tokens, with zero error, abort, preemption, or rank-restart outcomes.

The fixed short-request cases improved, but this does not establish a general production-throughput gain. The long-prefill TTFT pair did not improve, the production windows were not matched for prompt length, output length, cache residency, and concurrency, and the 24-hour window had no matched old-reader control.

## Interaction with the newer FAST path

`DSV41_ENGRAM_FAST=1` uses read-only memmaps plus NumPy gathers and bypasses `_kai_parallel_read`. This experiment intentionally leaves that path unchanged. It applies only when someone chooses to apply this artifact to the blocking `preadv` fallback.
