# Try 5 vs try 6: worker memory while loading the EXL3 TP3 weights

Stack and memory sampler logs from the vLLM worker on each rank (`KAI_STACK_SAMPLER=1`, patch sets `dsv41-exl3-tp3c` and `dsv41-exl3-tp3d`). Every 5 s the sampler writes one header line, `=== time anon file shmem` in GiB from `/proc/self/status`, followed by the worker's main-thread stack. Times are UTC (ET = UTC - 4).

Worker PIDs repeat from boot to boot (232 on Asusi and Reddie, 330 on Bluey), so each file holds two runs back to back:

- **try 5** (`exl3tp3a5`, stock `vl_model.py`), samples from 17:23 UTC until it was stopped by hand a few minutes later;
- **try 6** (`exl3tp3a6`, streaming `vl_model.py`), from 17:31 UTC: weight load until 17:40:35 (startup complete), then serving the post-boot smokes and benchmark.

Peak worker anonymous memory per phase (GiB):

| rank | try 5 load (stopped early) | try 6 load | try 6 serving |
|---|---|---|---|
| asusi-rank1 | 11.0 | 3.6 | 2.9 |
| bluey-rank0 | 9.7 | 3.5 | 2.9 |
| reddie-rank2 | 21.3 | 3.7 | 2.9 |

Try 5 was stopped before its load finished, so its peak is where it was cut off, not where it would have ended. Tries 3 and 4 ran their loads further and were stopped by the memory guard when a Spark fell to 4 GiB free; try 4's per-process capture had the worker at 24.5 GiB anonymous.
