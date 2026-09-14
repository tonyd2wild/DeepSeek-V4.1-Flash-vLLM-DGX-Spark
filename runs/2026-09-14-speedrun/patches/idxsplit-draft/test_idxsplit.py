#!/usr/bin/env python3
"""Exactness test for DSV41_INDEXER_TP_SPLIT (split vs unsplit indexer prefill top-k).

The GPU modes need the PATCHED sparse_attn_indexer.py mounted over the image's copy,
because they call the patch's own helpers (_dsv41_prefill_rows_local,
_dsv41_scatter_gathered, _dsv41_prefill_chunk_tp_split). Example, on one GB10:

  docker run --rm --gpus all --network none -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -v $PWD/sparse_attn_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro \
    -v $PWD/test_idxsplit.py:/t.py:ro --entrypoint python3 <serving image> /t.py --mode sim
  ... same with  /t.py --mode spawn --tp 4

Modes
  sim        One process. For each simulated rank r, run the patch's per-slice helper on
             rank r's rows, concatenate in rank order (what all_gather(dim=0) returns),
             scatter with the patch's helper, and compare with the unsplit reference.
  spawn      --tp processes sharing cuda:0 (torch.multiprocessing), gloo process group.
             Each rank runs _dsv41_prefill_chunk_tp_split() end to end with a gloo
             all-gather (NCCL refuses several ranks on one GPU), compares with its own
             unsplit reference, and checks that all ranks hold identical bytes.
  cpu-logic  No GPU, no vLLM. Pulls _dsv41_tp_split_bounds / _dsv41_scatter_gathered out
             of the patched file with ast and checks the row partition and reorder with
             fake per-row kernels.

The unsplit reference below is the sr1 loop body (sparse_attn_indexer.py, the prefill
chunk: fp8_fp4_mqa_logits -> candidate select/mask -> top_k_per_row_prefill into a -1
pre-filled view of the shared [T, topk] buffer), copied verbatim in call order and args.

Verdict per comparison
  BYTE-EQUAL                 split output == reference, byte for byte
  SET-EQUAL (ref nondet)     reference differs from itself across two runs (kernel
                             order/tie nondeterminism) and split matches it as per-row sets
  MISMATCH                   anything else -> the split is NOT safe; do not enable it
"""
from __future__ import annotations

import __future__ as _future
import argparse
import ast
import hashlib
import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHED = os.path.join(HERE, "sparse_attn_indexer.py")

# (name, [(prior_ctx_tokens, query_rows), ...], num_decode_tokens, compress_ratio)
CASES = [
    ("first-chunk r1", [(0, 8192)], 0, 1),
    ("late-chunk r2", [(40960, 8192)], 0, 2),
    ("3 reqs + 7 decode r1", [(20000, 3000), (0, 2500), (6000, 2692)], 7, 1),
    ("600 rows r1", [(5000, 600)], 3, 1),
    ("520 rows (empty last rank) r2", [(9000, 520)], 0, 2),
    ("8190 rows r2", [(1, 8190)], 0, 2),
    ("512 rows at MIN r1", [(30000, 512)], 1, 1),
]
LONG_CASE = ("long r1 (logits ~2.9 GB)", [(80000, 8192)], 0, 1)


# --------------------------------------------------------------------------- inputs
def row_bounds(reqs, ratio):
    """cu_seqlen_ks / cu_seqlen_ke exactly as BuildPrefillChunkMetadataKernel, no DCP."""
    ks, ke, start = [], [], 0
    for p, q in reqs:
        for j in range(q):
            ks.append(start)
            ke.append(start + (p + 1 + j) // ratio)
        start += (p + q) // ratio
    return ks, ke, start


def make_case(spec, args, use_fp4, device, seed):
    name, reqs, num_decode, ratio = spec
    torch.manual_seed(seed)
    ks, ke, n_keys = row_bounds(reqs, ratio)
    rows = len(ks)
    T = num_decode + rows
    H, D = args.heads, 128
    if use_fp4:
        if args.ties:
            lo_n = torch.randint(0, 2, (T, H, D // 2), device=device) * 2
            hi_n = torch.randint(0, 2, (T, H, D // 2), device=device) * 2
            q = (lo_n | (hi_n << 4)).to(torch.uint8)
            lo_n = torch.randint(0, 2, (n_keys, D // 2), device=device) * 2
            hi_n = torch.randint(0, 2, (n_keys, D // 2), device=device) * 2
            k_quant = (lo_n | (hi_n << 4)).to(torch.uint8)
            q_scale = torch.full((T, H, D // 32), 127, dtype=torch.uint8, device=device)
            k_scale = torch.full((n_keys, D // 32), 127, dtype=torch.uint8, device=device)
        else:
            q = torch.randint(0, 256, (T, H, D // 2), dtype=torch.uint8, device=device)
            q_scale = torch.randint(120, 131, (T, H, D // 32), dtype=torch.uint8, device=device)
            k_quant = torch.randint(0, 256, (n_keys, D // 2), dtype=torch.uint8, device=device)
            k_scale = torch.randint(120, 131, (n_keys, D // 32), dtype=torch.uint8, device=device)
    else:
        fp8 = torch.float8_e4m3fn
        q_scale = None
        if args.ties:
            q = torch.randint(0, 2, (T, H, D), device=device).float().to(fp8)
            k_quant = torch.randint(0, 2, (n_keys, D), device=device).float().to(fp8)
            k_f32 = torch.ones(n_keys, 1, device=device)
        else:
            q = (torch.randn(T, H, D, device=device) * 2).to(fp8)
            k_quant = (torch.randn(n_keys, D, device=device) * 2).to(fp8)
            k_f32 = torch.exp2(torch.randint(-4, 2, (n_keys, 1), device=device).float())
        k_scale = k_f32.contiguous().view(torch.uint8)  # [N, 4] like the workspace
    if args.ties:
        w = torch.ones(T, H, device=device, dtype=torch.float32)
    else:
        w = torch.randn(T, H, device=device, dtype=torch.float32) * 0.1
    return SimpleNamespace(
        name=name, use_fp4=use_fp4, T=T, t0=num_decode, t1=num_decode + rows,
        q=q, q_scale=q_scale, k_quant=k_quant, k_scale=k_scale, w=w,
        ks=torch.tensor(ks, dtype=torch.int32, device=device),
        ke=torch.tensor(ke, dtype=torch.int32, device=device),
        topk=args.topk, n_keys=n_keys,
    )


# --------------------------------------------------------------- unsplit reference
def unsplit_reference(sai, c, cand_in, cand_mode, cand_block):
    """sr1 prefill chunk body, verbatim in call order and arguments."""
    topk_indices_buffer = torch.full((c.T, c.topk), -1, dtype=torch.int32, device=c.q.device)
    candidate_blocks = cand_in.clone() if cand_in is not None else None
    q_slice = c.q[c.t0:c.t1]
    q_scale_slice = c.q_scale[c.t0:c.t1] if c.q_scale is not None else None
    topk_indices = topk_indices_buffer[c.t0:c.t1, :c.topk]
    if c.use_fp4:
        q_slice_cast = q_slice.view(torch.int8)
        k_quant_cast = c.k_quant.view(torch.int8)
        k_scale_cast = c.k_scale.view(torch.int32).squeeze(-1)
    else:
        q_slice_cast = q_slice
        k_quant_cast = c.k_quant
        k_scale_cast = c.k_scale.view(torch.float32).squeeze(-1)
    logits = sai.fp8_fp4_mqa_logits(
        (q_slice_cast, q_scale_slice),
        (k_quant_cast, k_scale_cast),
        c.w[c.t0:c.t1],
        c.ks,
        c.ke,
        clean_logits=False,
    )
    num_rows = logits.shape[0]
    if cand_mode != "none":
        chunk_candidates = candidate_blocks[c.t0:c.t1]
        if cand_mode == "write":
            sai._select_candidate_blocks(
                logits, c.ks, c.ke, chunk_candidates.shape[1], cand_block, chunk_candidates
            )
        else:
            sai._apply_candidate_mask(logits, c.ks, c.ke, chunk_candidates, cand_block)
    sai.ops.top_k_per_row_prefill(
        logits, c.ks, c.ke, topk_indices, num_rows,
        logits.stride(0), logits.stride(1), c.topk,
    )
    del logits
    return topk_indices_buffer, candidate_blocks


def split_sim(sai, c, world, cand_in, cand_mode, cand_block):
    """Per-rank slices via the patch's helper, cat in rank order, patch's scatter."""
    buf = torch.full((c.T, c.topk), -1, dtype=torch.int32, device=c.q.device)
    cand = cand_in.clone() if cand_in is not None else None
    rows = c.t1 - c.t0
    parts = []
    for r in range(world):
        lo, hi, per = sai._dsv41_tp_split_bounds(rows, world, r)
        sl = slice(c.t0 + lo, c.t0 + hi)
        parts.append(sai._dsv41_prefill_rows_local(
            c.q[sl], c.q_scale[sl] if c.q_scale is not None else None,
            c.k_quant, c.k_scale, c.w[sl], c.ks[lo:hi], c.ke[lo:hi],
            c.topk, per, cand[sl] if cand is not None else None,
            cand_block, cand_mode == "write", c.use_fp4,
        ))
    gathered = torch.cat(parts, 0)
    sai._dsv41_scatter_gathered(gathered, rows, c.t0, buf, c.topk, cand, cand_mode == "write")
    return buf, cand


def verdict(ref, ref2, got):
    if torch.equal(ref, got):
        return "BYTE-EQUAL", True
    ref_det = torch.equal(ref, ref2)
    set_eq = torch.equal(ref.sort(dim=1).values, got.sort(dim=1).values)
    if not ref_det and set_eq:
        return "SET-EQUAL (ref nondet)", True
    bad = (ref.sort(dim=1).values != got.sort(dim=1).values).any(dim=1).nonzero().flatten()
    return f"MISMATCH rows={bad[:8].tolist()} (n={bad.numel()}, ref_det={ref_det})", False


def import_patched():
    import vllm.model_executor.layers.sparse_attn_indexer as sai
    if not hasattr(sai, "_dsv41_prefill_rows_local"):
        sys.exit("The image's sparse_attn_indexer.py is not the patched one; mount it (see docstring).")
    return sai


def cand_modes(args):
    return ["none", "write", "consume"] if args.cand_k > 0 else ["none"]


def specs(args):
    return CASES + ([LONG_CASE] if args.long else [])


def fp4_modes(args):
    return {"fp8": [False], "fp4": [True], "both": [False, True]}[args.quant]


# ------------------------------------------------------------------------ sim mode
def run_sim(args):
    sai = import_patched()
    dev = "cuda"
    worlds = [int(x) for x in args.tp.split(",")]
    ok_all = True
    for si, spec in enumerate(specs(args)):
        for use_fp4 in fp4_modes(args):
            c = make_case(spec, args, use_fp4, dev, seed=1234 + si)
            cand_seed = None
            for mode in cand_modes(args):
                if mode == "none":
                    cand_in = None
                elif mode == "write":
                    cand_in = torch.full((c.T, args.cand_k), -1, dtype=torch.int32, device=dev)
                else:  # consume: candidates as the source layer published them
                    cand_in = cand_seed
                ref, ref_cand = unsplit_reference(sai, c, cand_in, mode, args.cand_block)
                ref2, ref2_cand = unsplit_reference(sai, c, cand_in, mode, args.cand_block)
                if mode == "write":
                    cand_seed = ref_cand
                for world in worlds:
                    got, got_cand = split_sim(sai, c, world, cand_in, mode, args.cand_block)
                    v, ok = verdict(ref, ref2, got)
                    line = f"[sim] {c.name:32s} {'fp4' if use_fp4 else 'fp8'} cand={mode:7s} tp={world}: topk {v}"
                    if mode == "write":
                        vc, okc = verdict(ref_cand, ref2_cand, got_cand)
                        line += f" | candidates {vc}"
                        ok = ok and okc
                    print(line, flush=True)
                    ok_all &= ok
            del c
            torch.cuda.empty_cache()
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------- spawn mode
def _spawn_worker(rank, world, args, port):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.cuda.set_device(0)
    sai = import_patched()
    dev = "cuda"

    def gloo_all_gather(t, dim):
        assert dim == 0
        cpu = t.contiguous().cpu()
        outs = [torch.empty_like(cpu) for _ in range(world)]
        dist.all_gather(outs, cpu)
        return torch.cat(outs, 0).to(t.device)

    failures = 0
    for si, spec in enumerate(specs(args)):
        for use_fp4 in fp4_modes(args):
            c = make_case(spec, args, use_fp4, dev, seed=1234 + si)  # same bytes on every rank
            chunk = SimpleNamespace(token_start=c.t0, token_end=c.t1,
                                    cu_seqlen_ks=c.ks, cu_seqlen_ke=c.ke)
            cand_seed = None
            for mode in cand_modes(args):
                cand_in = (None if mode == "none" else
                           torch.full((c.T, args.cand_k), -1, dtype=torch.int32, device=dev)
                           if mode == "write" else cand_seed)
                ref, ref_cand = unsplit_reference(sai, c, cand_in, mode, args.cand_block)
                ref2, ref2_cand = unsplit_reference(sai, c, cand_in, mode, args.cand_block)
                if mode == "write":
                    cand_seed = ref_cand
                buf = torch.full((c.T, c.topk), -1, dtype=torch.int32, device=dev)
                cand = cand_in.clone() if cand_in is not None else None
                sai._dsv41_prefill_chunk_tp_split(
                    chunk, c.q, c.q_scale, c.w, c.k_quant, c.k_scale, buf, c.topk,
                    cand, args.cand_block, mode == "write", c.use_fp4, world,
                    rank=rank, all_gather_fn=gloo_all_gather,
                )
                v, ok = verdict(ref, ref2, buf)
                if mode == "write":
                    vc, okc = verdict(ref_cand, ref2_cand, cand)
                    v += f" | candidates {vc}"
                    ok = ok and okc
                digest = hashlib.sha256(buf.cpu().numpy().tobytes()).hexdigest()[:16]
                digests = [None] * world
                dist.all_gather_object(digests, digest)
                same = len(set(digests)) == 1
                ok = ok and same
                if rank == 0 or not ok:
                    print(f"[spawn r{rank}] {c.name:32s} {'fp4' if use_fp4 else 'fp8'} cand={mode:7s} "
                          f"tp={world}: {v} | ranks identical={same}", flush=True)
                failures += 0 if ok else 1
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"rank {rank}: {failures} failing comparisons")


def run_spawn(args):
    import torch.multiprocessing as mp
    world = int(args.tp.split(",")[0])
    mp.spawn(_spawn_worker, args=(world, args, args.port), nprocs=world, join=True)
    print("RESULT: PASS")
    return 0


# ------------------------------------------------------------------ cpu-logic mode
def load_pure_helpers(path):
    """Exec only the vLLM-free helpers of the patched file (no vLLM import needed)."""
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_dsv41_tp_split_bounds", "_dsv41_scatter_gathered"
        ):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_DSV41_TP_SPLIT_ROW_ALIGN" for t in node.targets
        ):
            keep.append(node)
    assert len(keep) == 3, f"expected 3 helper nodes in {path}, found {len(keep)}"
    ns = {"torch": torch}
    code = compile(ast.Module(body=keep, type_ignores=[]), path, "exec",
                   flags=_future.annotations.compiler_flag, dont_inherit=True)
    exec(code, ns)
    return SimpleNamespace(bounds=ns["_dsv41_tp_split_bounds"],
                           scatter=ns["_dsv41_scatter_gathered"],
                           align=ns["_DSV41_TP_SPLIT_ROW_ALIGN"])


def run_cpu_logic(args):
    h = load_pure_helpers(args.patched)
    topk, cand_k, t0, sentinel = 16, 4, 5, -7
    n_checks = 0
    for world in (1, 2, 3, 4, 8):
        for rows in (1, 63, 64, 65, 511, 512, 520, 600, 1443, 4095, 8190, 8192):
            bounds = [h.bounds(rows, world, r) for r in range(world)]
            per = bounds[0][2]
            assert all(b[2] == per for b in bounds)
            assert per % h.align == 0 and per * world >= rows and per - h.align < -(-rows // world)
            covered = 0
            for r, (lo, hi, _) in enumerate(bounds):
                assert lo == min(r * per, rows) and lo <= hi <= lo + per
                assert lo == covered or hi == lo, (world, rows, bounds)
                covered = max(covered, hi)
            assert covered == rows
            for write in (False, True):
                parts = []
                for r, (lo, hi, _) in enumerate(bounds):
                    w_cols = topk + (cand_k if write else 0)
                    part = torch.full((per, w_cols), -1, dtype=torch.int32)
                    g = torch.arange(lo, hi, dtype=torch.int32)[:, None]
                    part[: hi - lo] = g * 1000 + torch.arange(w_cols, dtype=torch.int32)
                    parts.append(part)
                gathered = torch.cat(parts, 0)
                T = t0 + rows + 3
                buf = torch.full((T, topk), sentinel, dtype=torch.int32)
                cand = torch.full((T, cand_k), sentinel, dtype=torch.int32)
                h.scatter(gathered, rows, t0, buf, topk, cand, write)
                g = torch.arange(rows, dtype=torch.int32)[:, None]
                assert torch.equal(buf[t0:t0 + rows], g * 1000 + torch.arange(topk, dtype=torch.int32))
                assert (buf[:t0] == sentinel).all() and (buf[t0 + rows:] == sentinel).all()
                if write:
                    exp = g * 1000 + torch.arange(topk, topk + cand_k, dtype=torch.int32)
                    assert torch.equal(cand[t0:t0 + rows], exp)
                    assert (cand[:t0] == sentinel).all() and (cand[t0 + rows:] == sentinel).all()
                else:
                    assert (cand == sentinel).all()
                n_checks += 1
    print(f"cpu-logic: {n_checks} partition/reorder checks passed (align={h.align})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sim", "spawn", "cpu-logic"], default="sim")
    ap.add_argument("--tp", default="4,3,2", help="sim: comma list of world sizes; spawn: first value")
    ap.add_argument("--heads", type=int, default=32, help="config.index_n_heads (attention.py says 32)")
    ap.add_argument("--topk", type=int, default=1152, help="config.index_topk")
    ap.add_argument("--cand-k", type=int, default=64, help="config.candidate_topk_blocks (0 = skip)")
    ap.add_argument("--cand-block", type=int, default=64, help="config.candidate_block_size")
    ap.add_argument("--quant", choices=["fp8", "fp4", "both"], default="both",
                    help="indexer cache format (serving uses dsa_indexer_uses_fp4)")
    ap.add_argument("--ties", action="store_true", help="integer-valued inputs that force exact logit ties")
    ap.add_argument("--long", action="store_true", help="add an 88K-key case")
    ap.add_argument("--port", type=int, default=29533)
    ap.add_argument("--patched", default=PATCHED)
    args = ap.parse_args()
    return {"sim": run_sim, "spawn": run_spawn, "cpu-logic": run_cpu_logic}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
