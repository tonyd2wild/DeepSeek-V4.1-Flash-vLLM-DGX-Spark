# Page-cache fast path for the blocking Engram reader

This experiment accelerates the canonical `patch/engram.py` blocking `preadv` path without changing its file layout, dequantization, or thread-pool fallback.

## Mechanism

For every requested row, the caller first tries `os.preadv(..., os.RWF_NOWAIT)` synchronously. Rows that return `EAGAIN` are tried once more; only the residual rows go to the existing blocking pool. `EINVAL`, `ENOSYS`, `ENOTSUP`, and `EOPNOTSUPP` disable the fast path for that batch and run the unchanged reader. Partial and terminal short reads preserve the existing full-row/error behavior.

The goal is to avoid creating futures and switching threads for rows that Linux already has in the page cache. This is not a user-space row cache and adds no second copy of the Engram tables.

## Correctness checks

The exact implementation in `engram-nowait.diff` was deployed on four DGX Sparks after:

- 3,000 real-offset reads matched the existing reader byte for byte;
- forced cached-hit, one-`EAGAIN`, persistent-`EAGAIN`, partial-read, unsupported-operation, and terminal-short-read paths passed; and
- the old blocking path remained the fallback whenever NOWAIT was unavailable.

## Component results

Corrected residency sampling used `mincore` to classify pages without reading them. Separate offsets and buffers were used for the old and candidate arms.

| Rank | 50% classified-cold | 100% classified-cold |
| ---: | ---: | ---: |
| 0 | 5.78x | 2.35x |
| 1 | 3.62x | 2.22x |
| 2 | 2.00x | 2.57x |
| 3 | 6.69x | 1.94x |

For an 8,192-token prefill-shaped cached workload (196,608 row reads across four logical jobs), median component time fell from 1,052.327 ms to 126.787 ms (8.30x). Voluntary context switches fell from 296,162 to zero in the measured pair.

## Production evidence and limits

After the single reader change was deployed through the normal four-rank service path:

| Fixed workload | Before | After |
| --- | ---: | ---: |
| C1 output rate | 12.07 tok/s | 35.49 tok/s |
| C4 wall aggregate | 28.49 tok/s | 62.96 tok/s |
| 13,316-token prompt TTFT | 5.889 s | 5.989 s |

A 24-hour soak completed about 3,921 requests, 1.064 billion prompt tokens, and 4.378 million generation tokens with zero error, abort, preemption, or rank restart outcomes.

These results prove that the patch is stable and that bounded short-request cases can improve. They do not prove a general production-throughput gain: the long-prefill TTFT pair did not improve, and the production windows were not matched for prompt length, output length, cache residency, and concurrency.

## Interaction with the newer FAST path

`DSV41_ENGRAM_FAST=1` uses read-only memmaps plus NumPy gathers and bypasses `_kai_parallel_read`. This NOWAIT patch intentionally does not compete with or modify that path. It applies to the blocking `preadv` fallback and to deployments that do not enable FAST.

Apply from the repository root with:

```bash
git apply patch/engram-nowait/engram-nowait.diff
```
