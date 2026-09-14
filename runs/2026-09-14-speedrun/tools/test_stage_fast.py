#!/usr/bin/env python3
"""Offline check of the Engram FAST path (CPU only, no GPU): for random row sets with duplicates
and unowned rows, the fast gather (memmap + np.take on the pool) plus dequant_rows_torch must give
bf16 rows bit-identical to gather_dequant_many (the preadv path). Also times both on fresh row sets
(decode-sized 288 rows and prefill-sized 98,304 rows). Run inside the exl3a image with the sr1
engram.py mounted over vllm/models/deepseek_v4_1/common/engram.py and /engram-local mounted."""
import os
import time

import torch

os.environ.setdefault("DSV41_ENGRAM_DIR", "/engram-local")
from vllm.models.deepseek_v4_1.common import engram as E  # noqa: E402

ROW_START, NUM_ROWS = int(os.environ["ROW_START"]), int(os.environ["NUM_ROWS"])
t = E.DiskEngramTable("/engram-local", layer_id=1, dim=256, block_size=32, row_start=ROW_START, num_rows=NUM_ROWS)
g = torch.Generator().manual_seed(int(os.environ.get("SEED", "1234")))


def make(R):
    base = torch.randint(0, t.num_rows, (max(1, R // 3),), generator=g)
    rel = torch.cat([base, base[torch.randint(0, base.numel(), (R - base.numel(),), generator=g)]])
    owned = torch.rand(R, generator=g) > 0.1
    return torch.where(owned, rel, torch.zeros_like(rel)), owned


def fast(rel, owned):
    uniq, inv = torch.unique(rel, return_inverse=True)
    U = uniq.numel()
    w = torch.empty((U, t.dim), dtype=torch.uint8)
    s = torch.empty((U, t.sb), dtype=torch.uint8)
    tasks = []
    wm, sm = t.memmaps()
    E._fast_gather(wm, uniq.numpy(), w.numpy(), tasks)
    E._fast_gather(sm, uniq.numpy(), s.numpy(), tasks)
    for x in tasks:
        x.result()
    rows = E.dequant_rows_torch(w, s, t.dim, t.sb).index_select(0, inv)
    rows.masked_fill_(~owned[:, None], 0.0)
    return rows.to(torch.bfloat16)


ok = True
for R in (288, 98304):
    rel, owned = make(R)
    ref = E.gather_dequant_many([(t, rel, owned)])[0]
    new = fast(rel, owned)
    eq = torch.equal(ref, new)
    ok &= eq
    # timing on fresh sets (each path sees rows the other has not touched in this run)
    ra, oa = make(R)
    rb, ob = make(R)
    t0 = time.time(); E.gather_dequant_many([(t, ra, oa)]); t_old = time.time() - t0
    t0 = time.time(); fast(rb, ob); t_new = time.time() - t0
    t0 = time.time(); E.gather_dequant_many([(t, ra, oa)]); t_old_w = time.time() - t0
    t0 = time.time(); fast(rb, ob); t_new_w = time.time() - t0
    print(f"R={R:6d} bit-identical={eq}  cold: preadv+cpu {t_old*1e3:8.1f} ms, fast {t_new*1e3:8.1f} ms | "
          f"warm: preadv+cpu {t_old_w*1e3:8.1f} ms, fast(cpu dequant here) {t_new_w*1e3:8.1f} ms", flush=True)
print("RESULT", "PASS" if ok else "FAIL")
