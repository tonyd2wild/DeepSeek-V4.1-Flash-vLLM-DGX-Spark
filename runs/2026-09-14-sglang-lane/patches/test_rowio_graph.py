#!/usr/bin/env python3
"""test_rowio_graph.py <tp_rank> (inside lmsysorg/sglang:dev-dsv41, one GPU, our engram.py mounted,
SGLANG_DSV41_ENGRAM_ROWIO pointing at libdsv41_rowio.so). For Engram layers 1 and 14 it checks that
DiskEngramRows.lookup returns exactly what a direct safetensors read gives (SGLang's dequant math) for
owned ids, and +0.0 for unowned ids:
  1) eagerly, and
  2) inside a captured CUDA graph (the native reader runs as a host node), replayed 3 times with fresh
     ids copied into the static input buffer, which is what the decode graph does every step.
Model folder /models/DeepSeek-V4.1-Flash-Ablit (reference reads), rank rows from /engram."""
import json
import os
import random
import sys
import time

import safetensors
import torch

from sglang.srt.layers import engram as E
from sglang.srt.layers.attention.dsv4.torch_quant import FP8_BLOCK_SIZE

rank = int(sys.argv[1]); dim = 256
MODEL = "/models/DeepSeek-V4.1-Flash-Ablit"
ENGRAM = "/engram"
wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
local = json.load(open(f"{ENGRAM}/engram-local.json"))["layers"] if os.path.exists(f"{ENGRAM}/engram-local.json") else None
ok = True


def reference(L, ids, lo, hi):
    with safetensors.safe_open(f"{MODEL}/{wm[f'layers.{L}.engram.embed.weight']}", "pt") as fw, \
         safetensors.safe_open(f"{MODEL}/{wm[f'layers.{L}.engram.embed.scale']}", "pt") as fs:
        sw, ss = fw.get_slice(f"layers.{L}.engram.embed.weight"), fs.get_slice(f"layers.{L}.engram.embed.scale")
        ref = torch.zeros(len(ids), dim, dtype=torch.bfloat16)
        for i, r in enumerate(ids):
            if lo <= r < hi:
                w = sw[r:r + 1].float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
                s = ss[r:r + 1].float().unsqueeze(-1)
                ref[i] = (w * s).flatten(-2).to(torch.bfloat16)[0]
    return ref


def pick(lo, hi, N, n):
    ids = [lo, hi - 1] + [random.randrange(lo, hi) for _ in range(n - 5)]
    ids += [(lo - 1) % N, hi % N, random.randrange(N)]
    return ids[:n]


for L in (1, 14):
    with safetensors.safe_open(f"{MODEL}/{wm[f'layers.{L}.engram.embed.weight']}", "pt") as fw:
        N = fw.get_slice(f"layers.{L}.engram.embed.weight").get_shape()[0]
    if local is not None:
        lo, hi = local[str(L)]
        root = ENGRAM
    else:
        lo, hi = 0, N // 4  # rank 0 has the full shards; any range works for the check
        root = MODEL
    d = E.DiskEngramRows(root, L, N, dim, lo, hi)
    # 1) eager
    ids = pick(lo, hi, N, 96)
    idx = torch.tensor(ids, dtype=torch.int64, device="cuda").view(-1, 2)
    out = torch.empty(*idx.shape, dim, dtype=torch.bfloat16, device="cuda")
    d.lookup(idx, out); torch.cuda.synchronize()
    eq = torch.equal(out.view(-1, dim).cpu(), reference(L, ids, lo, hi))
    print(f"layer {L} rank {rank} rows [{lo},{hi}) eager EQUAL={eq}")
    ok &= eq
    # 2) CUDA graph with the native reader as a host node
    n = 48
    static_idx = torch.tensor(pick(lo, hi, N, n), dtype=torch.int64, device="cuda").view(-1, 6)
    static_out = torch.empty(*static_idx.shape, dim, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        d.lookup(static_idx, static_out)  # warm-up (eager, also allocates staging)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        d.lookup(static_idx, static_out)
    for rep in range(3):
        ids = pick(lo, hi, N, n)
        static_idx.copy_(torch.tensor(ids, dtype=torch.int64).view(-1, 6))
        t = time.time(); g.replay(); torch.cuda.synchronize(); ms = 1000 * (time.time() - t)
        eq = torch.equal(static_out.view(-1, dim).cpu(), reference(L, ids, lo, hi))
        print(f"  graph replay {rep}: {n} ids {ms:.2f} ms EQUAL={eq}")
        ok &= eq
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
