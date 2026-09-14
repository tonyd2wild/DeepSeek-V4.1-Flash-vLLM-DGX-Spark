#!/usr/bin/env python3
"""RoCEnante (b12x.comm.roce) vs NCCL: correctness + latency on 4 DGX Sparks, TP4 decode shapes.

Written for the DSV4.1 speed run. The runtime and its method names come from
b12x.comm.roce (RoCEnante) by Jason @original-el8 and Luke Alonso @lukealonso,
https://github.com/local-inference-lab/b12x (pinned b58f34ea). For the full
upstream benchmark with receipts, see /opt/b12x-roce/benchmark_roce_oneshot.py
in the image built by Dockerfile.roce.

  preflight (one node, no peers):  python3 test_roce_latency.py --preflight
  4 ranks:  torchrun --nnodes 4 --nproc-per-node 1 --node-rank R \
              --master-addr 192.168.192.2 --master-port 29651 test_roce_latency.py

Timing: CUDA events, median per rank, then the slowest rank (what TP decode pays).
Graph arm captures --graph-ops back-to-back collectives in one CUDA graph.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys

import torch
import torch.distributed as dist


def preflight() -> int:
    ok = True
    try:
        from b12x.comm import roce
        from b12x.comm.roce import _proxy
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL import b12x.comm.roce: {exc}")
        return 2
    print("API_VERSION", roce.API_VERSION, "(adapter needs 1)")
    ok &= roce.API_VERSION == 1
    print("gid_index", roce.default_gid_index())
    print("hcas", roce.discover_hcas())
    try:
        print("proxy .so", _proxy.load())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL proxy build/load: {exc}")
        ok = False
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("gpu", props.name, "integrated", getattr(props, "is_integrated", None))
        print("is_supported", roce.is_supported(0))
        ok &= bool(roce.is_supported(0))
    else:
        print("FAIL no CUDA device")
        ok = False
    for mod in ("b12x.gemm.blockscaled", "b12x._lib.intrinsics", "b12x.moe.fused_moe",
                "b12x.attention.paged", "b12x.gemm.mxfp8_linear", "b12x.gemm.tensor_fp8_linear"):
        try:
            __import__(mod)
            print("import ok", mod)
        except Exception as exc:  # noqa: BLE001
            print(f"import FAIL {mod}: {exc}")
    print("PREFLIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def slowest_median(samples_us: list[float]) -> float:
    med = statistics.median(samples_us)
    out = [0.0] * dist.get_world_size()
    dist.all_gather_object(out, med)
    return max(out)


def time_eager(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    res = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        res.append(a.elapsed_time(b) * 1000.0)
    return res


def time_graph(g: torch.cuda.CUDAGraph, ops: int, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    res = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        res.append(a.elapsed_time(b) * 1000.0 / ops)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    # 48 KB = 6 tok x 4096 x bf16 (C1 DSpark k=5 verify); up to 480 KB seen per step; 1-2 MB = cap.
    ap.add_argument("--sizes", default="8192,49152,98304,196608,294912,491520,1048576,2097152")
    ap.add_argument("--gather-rows", default="6,36")
    ap.add_argument("--gather-cols", type=int, default=129280 // 4)  # DSV4.1 vocab / TP4
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--graph-ops", type=int, default=20)
    ap.add_argument("--no-nccl-graph", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.preflight:
        return preflight()

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=dev)
    rank, world = dist.get_rank(), dist.get_world_size()
    gloo = dist.new_group(backend="gloo")

    from b12x.comm import roce
    assert roce.API_VERSION == 1, roce.API_VERSION
    rt = roce.AllReduce.from_exchange_group(
        exchange_group=gloo, device=dev, max_size=2 << 20, max_gather_bytes=16 << 20)
    if rank == 0:
        print(f"world={world} host={socket.gethostname()} hcas={rt.hca_names} "
              f"spin={os.environ.get('B12X_ROCE_SPIN_LIMIT', 'default')}", flush=True)
    dtype = torch.bfloat16
    rt.prepare((torch.bfloat16, torch.float16, torch.float32))

    rows: list[dict] = []
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)

    # ---- all-reduce ----
    for nbytes in [int(s) for s in a.sizes.split(",")]:
        n = nbytes // 2
        x = torch.randn(n, device=dev, dtype=dtype, generator=gen)
        if not rt.should_allreduce(x):
            if rank == 0:
                print(f"AR {nbytes} B: not eligible, skipped")
            continue
        ref = x.float().clone()
        dist.all_reduce(ref)
        got = rt.all_reduce(x)
        torch.cuda.synchronize()
        rel = ((got.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
        digests = [None] * world
        dist.all_gather_object(digests, hash(got.cpu().view(torch.int16).numpy().tobytes()))
        identical = len(set(digests)) == 1

        nccl_buf = x.clone()
        nccl_e = time_eager(lambda: dist.all_reduce(nccl_buf), a.iters, a.warmup)
        roce_e = time_eager(lambda: rt.all_reduce(x), a.iters, a.warmup)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s), rt.capture(stream=s):
            with torch.cuda.graph(g, stream=s):
                y = x
                for _ in range(a.graph_ops):
                    y = rt.all_reduce(x)
        roce_g = time_graph(g, a.graph_ops, a.iters // 3, 10)
        rt.check_health()

        nccl_g = None
        if not a.no_nccl_graph:
            try:
                g2 = torch.cuda.CUDAGraph()
                s2 = torch.cuda.Stream()
                s2.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s2), torch.cuda.graph(g2, stream=s2):
                    for _ in range(a.graph_ops):
                        dist.all_reduce(nccl_buf)
                nccl_g = time_graph(g2, a.graph_ops, a.iters // 3, 10)
            except Exception as exc:  # noqa: BLE001
                if rank == 0:
                    print(f"NCCL graph arm skipped: {exc}")
        row = {
            "op": "all_reduce", "bytes": nbytes, "max_rel_err_vs_nccl_fp32": rel,
            "ranks_bit_identical": identical,
            "nccl_eager_us": slowest_median(nccl_e), "roce_eager_us": slowest_median(roce_e),
            "roce_graph_us": slowest_median(roce_g),
            "nccl_graph_us": slowest_median(nccl_g) if nccl_g else None,
        }
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)

    # ---- all-gather (logits shard [rows, vocab/tp], last dim) ----
    for r in [int(v) for v in a.gather_rows.split(",")]:
        shard = torch.randn(r, a.gather_cols, device=dev, dtype=dtype, generator=gen)
        if not rt.should_all_gather(shard, -1):
            if rank == 0:
                print(f"AG [{r},{a.gather_cols}]: not eligible, skipped")
            continue
        flat = torch.empty(world * shard.numel(), device=dev, dtype=dtype)
        dist.all_gather_into_tensor(flat, shard.contiguous())
        ref = flat.view(world, r, a.gather_cols).permute(1, 0, 2).reshape(r, world * a.gather_cols)
        got = rt.all_gather(shard, dim=-1)
        torch.cuda.synchronize()
        exact = torch.equal(got, ref)

        def nccl_ag():
            dist.all_gather_into_tensor(flat, shard)
            return flat.view(world, r, a.gather_cols).permute(1, 0, 2).reshape(r, -1)

        nccl_e = time_eager(nccl_ag, a.iters, a.warmup)
        roce_e = time_eager(lambda: rt.all_gather(shard, dim=-1), a.iters, a.warmup)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s), rt.capture(stream=s):
            with torch.cuda.graph(g, stream=s):
                for _ in range(a.graph_ops):
                    rt.all_gather(shard, dim=-1)
        roce_g = time_graph(g, a.graph_ops, a.iters // 3, 10)
        rt.check_health()
        row = {
            "op": "all_gather_lastdim", "shard": [r, a.gather_cols],
            "shard_bytes": shard.numel() * 2, "bit_exact_vs_nccl": exact,
            "nccl_eager_us": slowest_median(nccl_e), "roce_eager_us": slowest_median(roce_e),
            "roce_graph_us": slowest_median(roce_g),
        }
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)

    rt.check_health()
    bad = [r for r in rows if not r.get("ranks_bit_identical", True)
           or not r.get("bit_exact_vs_nccl", True)
           or r.get("max_rel_err_vs_nccl_fp32", 0) > 1e-2]
    if rank == 0:
        verdict = "PASS" if not bad else f"FAIL ({len(bad)} rows)"
        print("CORRECTNESS", verdict)
        if a.out:
            with open(a.out, "w") as fh:
                json.dump({"hcas": list(rt.hca_names), "rows": rows, "verdict": verdict}, fh, indent=1)
    rt.close()
    dist.barrier(group=gloo)
    dist.destroy_process_group()
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
