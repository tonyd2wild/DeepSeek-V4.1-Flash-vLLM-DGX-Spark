#!/usr/bin/env python3
"""test_engram_disk.py <tp_rank> (inside lmsysorg/sglang:dev-dsv41 with our engram.py mounted, one GPU).
Checks that the disk lookup (_disk_owned_rows_impl) returns exactly the rows a direct safetensors read gives,
dequantized with SGLang's math, for random owned rows, both edges of the rank's range (including rows outside
the node-local copy, which come from the model directory), and unowned rows (must be zero). Then times a
decode-sized and a prefill-sized lookup. Model at /models/DeepSeek-V4.1-Flash-Ablit, local rows at /engram."""
import random
import sys
import time

import numpy as np
import safetensors
import torch

from sglang.srt.layers import engram as E
from sglang.srt.layers.attention.dsv4.torch_quant import FP8_BLOCK_SIZE

rank = int(sys.argv[1]); tp = 4; dim = 256
MODEL, LOCAL = "/models/DeepSeek-V4.1-Flash-Ablit", "/engram"
import json
wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
ok = True
for L in (1, 14):
    N = E._dsk_memmaps(MODEL, L)[0].shape[0]
    a = N * rank // tp; b = N * (rank + 1) // tp
    disk = E._DiskEngramRows(L, a, b - a, dim, model_dir=MODEL, local_dir=LOCAL)
    ids = [a, a + 1, b - 2, b - 1] + [random.randrange(a, b) for _ in range(120)]
    if disk.fallback is not None:  # rows outside the local copy
        ids += [r for r in (disk.hi, disk.hi + 1, disk.lo - 1) if a <= r < b]
    ids += [(a - 1) % N, b % N, random.randrange(N)]  # likely unowned
    ids = ids[: len(ids) // 2 * 2]
    idx = torch.tensor(ids, dtype=torch.int64, device="cuda").view(-1, 2)
    got = E._disk_owned_rows_impl(disk, idx).cpu().view(-1, dim)
    with safetensors.safe_open(f"{MODEL}/{wm[f'layers.{L}.engram.embed.weight']}", "pt") as fw, \
         safetensors.safe_open(f"{MODEL}/{wm[f'layers.{L}.engram.embed.scale']}", "pt") as fs:
        sw, ss = fw.get_slice(f"layers.{L}.engram.embed.weight"), fs.get_slice(f"layers.{L}.engram.embed.scale")
        ref = torch.zeros(len(ids), dim, dtype=torch.bfloat16)
        for i, r in enumerate(ids):
            if a <= r < b:
                w = sw[r:r + 1].float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
                s = ss[r:r + 1].float().unsqueeze(-1)
                ref[i] = (w * s).flatten(-2).to(torch.bfloat16)[0]
    eq = torch.equal(got, ref)
    nz = int((got.abs().sum(-1) > 0).sum())
    print(f"layer {L} rank {rank} rows [{a},{b}) local [{disk.lo},{disk.hi}) fallback={disk.fallback is not None}"
          f" ids={len(ids)} nonzero={nz} EQUAL={eq}")
    ok &= eq
    for n in (288, 98304):
        q = torch.randint(0, N, (n,), device="cuda")
        E._disk_owned_rows_impl(disk, q); torch.cuda.synchronize()
        t = time.time(); E._disk_owned_rows_impl(disk, q); torch.cuda.synchronize()
        print(f"  lookup {n} rows: {1000 * (time.time() - t):.1f} ms (second call)")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
