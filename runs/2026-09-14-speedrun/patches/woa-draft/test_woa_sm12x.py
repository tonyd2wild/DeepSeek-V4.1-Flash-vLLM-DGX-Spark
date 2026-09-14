#!/usr/bin/env python3
"""Standalone correctness + speed test for the DSV41_WOA_B12X wo_a path.

Run on ONE GB10 inside the serving image (vllm-dsv41:exl3b) with the draft
woa_sm12x.py importable, e.g.:

    docker run --rm --gpus all -v $PWD/woa-draft:/w -w /w vllm-dsv41:exl3b \
        bash -c 'cp woa_sm12x.py $(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/models/deepseek_v4_1/ && python3 test_woa_sm12x.py'

What it checks, per backend (b12x, fi) and per token count:
  1. Load-time restore: Emulation's BF16 wo_a -> FP8 is bit-exact vs the
     original checkpoint FP8 bytes.
  2. Numerics: new path vs (A) the production BF16 emulation bmm and
     (B) an FP32 reference that also MXFP8-quantizes the activation (which is
     what the kernels do, same as the SM100 DeepGEMM path).
  3. CUDA-graph safety: capture + replay equals eager (capture fails loudly on
     any host sync / legacy-stream launch).
  4. Speed: eager and graph-replay time vs the current BF16 torch.bmm.
Shapes default to TP4 per rank: 2 local groups x [1024, 4096].
"""

import argparse
import os
import sys
import types

import torch
import torch.nn as nn

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def make_ckpt_mxfp8(n, k, block_rows=32, std=0.02, seed=0):
    """Random weight quantized like the checkpoint: 32x32 ue8m0 blocks,
    expanded to per-row [N, K/32] (what KMxfp8Static's loader produces)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g) * std
    wb = w.view(n // block_rows, block_rows, k // 32, 32)
    amax = wb.abs().amax(dim=(1, 3)).clamp(min=torch.finfo(torch.float32).tiny)
    s = (torch.ceil(torch.log2(amax / FP8_MAX)) + 127).clamp(0, 254)
    descale = torch.exp2(s - 127)[:, None, :, None]
    w_fp8 = (wb / descale).view(n, k).to(FP8)
    scale = s.to(torch.uint8).repeat_interleave(block_rows, dim=0).contiguous()
    return w_fp8.contiguous(), scale


def dequant(w_fp8, s):
    n, k = w_fp8.shape
    d = torch.exp2(s.float() - 127).repeat_interleave(32, dim=1)
    return w_fp8.float() * d


def mxfp8_qdq(x):
    """MXFP8 (1x32, ue8m0 ceil) quantize-dequantize of activations, FP32 out."""
    shp = x.shape
    xb = x.float().view(*shp[:-1], shp[-1] // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)
    s = (torch.ceil(torch.log2(amax / FP8_MAX)) + 127).clamp(0, 254)
    d = torch.exp2(s - 127)
    return ((xb / d).to(FP8).float() * d).view(shp)


def rel_fro(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def time_fn(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        st.record()
        fn()
        en.record()
        en.synchronize()
        ts.append(st.elapsed_time(en) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]  # median, us


def graph_of(fn, x_static):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x_static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(x_static)
    return g, out


def grouped_forward(groups, x_tgd):
    # Same as o_proj.woa_grouped_forward (patched o_proj.py); kept local so the
    # test does not need the patched o_proj mounted.
    try:
        from vllm.models.deepseek_v4.nvidia.ops.o_proj import woa_grouped_forward

        return woa_grouped_forward(groups, x_tgd)
    except ImportError:
        return torch.stack([grp(x_tgd[:, i]) for i, grp in enumerate(groups)], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--tokens", default="1,2,4,6,8,12,16,24,32,64,256,2048")
    ap.add_argument("--backends", default="b12x,fi")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    from vllm.models.deepseek_v4_1 import woa_sm12x as W

    G, R, K = args.groups, args.rank, args.k
    N = G * R
    cap = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}  G={G} R={R} K={K}")
    w_fp8, s = make_ckpt_mxfp8(N, K)
    w_bf16_emul = dequant(w_fp8, s).to(torch.bfloat16)  # == Emulation at load
    w_ref32 = dequant(w_fp8, s)
    failures = []

    # 1. restore exactness
    fake = nn.Module()
    fake.weight = nn.Parameter(w_bf16_emul.clone(), requires_grad=False)
    fake.weight_scale = nn.Parameter(s.clone(), requires_grad=False)
    got = W.restore_fp8_weight(fake)
    ok = got is not None and torch.equal(
        got[0].view(torch.uint8), w_fp8.view(torch.uint8)
    ) and torch.equal(got[1], s)
    print(f"[restore] BF16->FP8 bit-exact: {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("restore")

    toks = [int(t) for t in args.tokens.split(",")]
    wbytes_fp8 = N * K
    wbytes_bf16 = 2 * N * K
    for backend in args.backends.split(","):
        os.environ["DSV41_WOA_B12X"] = backend
        wo_a = nn.Module()
        wo_a.weight = nn.Parameter(w_bf16_emul.clone(), requires_grad=False)
        wo_a.weight_scale = nn.Parameter(s.clone(), requires_grad=False)
        attn = types.SimpleNamespace(wo_a=wo_a, n_local_groups=G, o_lora_rank=R)
        W.maybe_install_woa_sm12x(attn)
        groups = getattr(wo_a, "dsv41_woa_groups", None)
        if groups is None:
            print(f"[{backend}] install declined (see log) -> FAIL")
            failures.append(f"{backend}:install")
            continue
        print(f"\n[{backend}] installed: wo_a.weight now {tuple(wo_a.weight.shape)}")
        print(
            f"{'T':>5} {'relA':>8} {'cosA':>9} {'relB':>8} {'qdq_relA':>9} "
            f"{'bf16_us':>8} {'new_us':>8} {'bf16_g_us':>9} {'new_g_us':>9} {'graph':>6}"
        )
        for T in toks:
            gen = torch.Generator(device="cuda").manual_seed(T)
            # o_proj_input is a [T,G,K] transpose view of a contiguous [G,T,K] buffer
            buf = torch.randn(G, T, K, device="cuda", generator=gen).to(torch.bfloat16)
            x = buf.transpose(0, 1)

            def run_bf16(xx):
                z = torch.empty(xx.shape[0], G, R, device="cuda", dtype=torch.bfloat16)
                torch.bmm(
                    xx.transpose(0, 1),
                    w_bf16_emul.view(G, R, K).transpose(1, 2),
                    out=z.transpose(0, 1),
                )
                return z

            def run_new(xx):
                return grouped_forward(groups, xx)

            z_a = run_bf16(x)
            z_new = run_new(x)
            xq = mxfp8_qdq(buf)  # [G,T,K]
            z_b = torch.einsum("gtk,grk->tgr", xq, w_ref32.view(G, R, K))
            relA, cosA, relB = rel_fro(z_new, z_a), cos(z_new, z_a), rel_fro(z_new, z_b)
            qdq_relA = rel_fro(z_b, z_a)  # inherent activation-quant error
            okT = (
                z_new.shape == (T, G, R)
                and torch.isfinite(z_new).all().item()
                and relB < 1e-2
                and cosA > 0.998
            )
            t_bf16 = time_fn(lambda: run_bf16(x), args.iters)
            t_new = time_fn(lambda: run_new(x), args.iters)
            tg_bf16 = tg_new = float("nan")
            gstat = "skip"
            if not args.no_graph:
                try:
                    # static input with the real layout: [T,G,K] view of [G,T,K]
                    xs = buf.clone().transpose(0, 1)
                    g_new, out_new = graph_of(run_new, xs)
                    g_new.replay()
                    torch.cuda.synchronize()
                    gstat = "PASS" if torch.equal(out_new, run_new(xs)) else "DIFF"
                    g_bf, _ = graph_of(run_bf16, xs)
                    tg_new = time_fn(g_new.replay, args.iters)
                    tg_bf16 = time_fn(g_bf.replay, args.iters)
                except Exception as e:  # capture failure == not graph-safe
                    gstat = "FAIL"
                    print(f"   graph capture error T={T}: {type(e).__name__}: {e}")
            if gstat in ("FAIL", "DIFF"):
                okT = False
            print(
                f"{T:>5} {relA:8.4f} {cosA:9.6f} {relB:8.5f} {qdq_relA:9.4f} "
                f"{t_bf16:8.1f} {t_new:8.1f} {tg_bf16:9.1f} {tg_new:9.1f} {gstat:>6}"
                + ("" if okT else "  <-- FAIL")
            )
            if not okT:
                failures.append(f"{backend}:T={T}")
        t1 = time_fn(lambda: run_bf16(buf[:, :1].transpose(0, 1)), args.iters)
        print(
            f"   T=1 BF16 bmm eff. BW {wbytes_bf16 / t1 / 1e3:.0f} GB/s; "
            f"FP8 bytes would take {wbytes_fp8 / (wbytes_bf16 / t1):.1f} us at same BW"
        )

    # Optional: FlashInfer batched bmm_mxfp8 (option c), informational only.
    try:
        from flashinfer import bmm_mxfp8, mxfp8_quantize

        wq, ws = mxfp8_quantize(
            w_bf16_emul.view(G, R, K).contiguous(), is_sf_swizzled_layout=True
        )
        print("\n[fi-bmm] (experimental, weight re-quantized by FlashInfer)")
        for T in toks:
            buf = torch.randn(G, T, K, device="cuda").to(torch.bfloat16)

            def run_fb(bb=buf):
                xq, xs = mxfp8_quantize(bb, is_sf_swizzled_layout=True)
                return bmm_mxfp8(xq, wq.transpose(-2, -1), xs, ws, torch.bfloat16)

            ref = torch.bmm(buf, w_bf16_emul.view(G, R, K).transpose(1, 2))
            out = run_fb()
            print(
                f"{T:>5} relA={rel_fro(out, ref):.4f} us={time_fn(run_fb, args.iters):.1f}"
            )
    except Exception as e:
        print(f"\n[fi-bmm] skipped: {type(e).__name__}: {e}")

    print("\nRESULT:", "PASS" if not failures else f"FAIL {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    # Allow running with woa_sm12x.py sitting next to this script (not yet
    # mounted into the vllm tree): alias it as vllm.models.deepseek_v4_1.woa_sm12x.
    here = os.path.dirname(os.path.abspath(__file__))
    loader = types.ModuleType("woa_sm12x_loader")
    sys.modules["woa_sm12x_loader"] = loader
    try:
        import vllm.models.deepseek_v4_1.woa_sm12x  # noqa: F401
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "vllm.models.deepseek_v4_1.woa_sm12x", os.path.join(here, "woa_sm12x.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["vllm.models.deepseek_v4_1.woa_sm12x"] = mod
        spec.loader.exec_module(mod)
        import vllm.models.deepseek_v4_1 as pkg

        pkg.woa_sm12x = mod
    main()
