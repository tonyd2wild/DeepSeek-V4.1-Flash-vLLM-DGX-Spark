"""GPU check of TP3 virtual heads THROUGH vLLM's real loaders (the path that killed exl3tp3a).
For each fake TP rank 0..2: build wq_b / wo_a / wo_b exactly like the V4.1 attention does (72 heads, 9 groups,
DeepseekV4FP8Config -> ModelOpt MXFP8 static), install virtual_heads.wrap_attention, load the REAL checkpoint
tensors, compare every loaded value with the reference (checkpoint padded in checkpoint units, sliced per rank),
run process_weights_after_loading, then a forward through vLLM's kernels vs a float reference."""
import json, sys, types, traceback, torch
sys.path.insert(0, "/t")
import virtual_heads as VH
from safetensors import safe_open
M = "/m"
wm = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
def get(n):
    with safe_open(f"{M}/{wm[n]}", framework="pt") as f: return f.get_tensor(n)
cfg = json.load(open(f"{M}/config.json")); oq = cfg["quantization_config"]["original_quantization_config"]
import vllm.model_executor.layers.linear as L
from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config
from vllm.config import VllmConfig, set_current_vllm_config
qc = DeepseekV4FP8Config.from_config(dict(oq))
TP, HD, OL, RG, PG = 3, 512, 1024, 8, 9
def set_rank(r):
    for name, mod in list(sys.modules.items()):
        if name.startswith("vllm") and mod is not None:
            if hasattr(mod, "get_tensor_model_parallel_rank"): mod.get_tensor_model_parallel_rank = (lambda r=r: r)
            if hasattr(mod, "get_tensor_model_parallel_world_size"): mod.get_tensor_model_parallel_world_size = (lambda: TP)
def e8(s): return torch.pow(2.0, s.view(torch.uint8).float() - 127.0)
def deq_ckpt(w, s):  # 32x32 blocks
    return w.float() * e8(s).repeat_interleave(32, 0).repeat_interleave(32, 1)[: w.shape[0], : w.shape[1]]
def deq_loaded(w, s):
    w = w.float().cpu(); s = s.cpu()
    sc = e8(s) if s.element_size() == 1 else s.float()
    if sc.shape[0] == w.shape[0]: sc = sc.repeat_interleave(w.shape[1] // sc.shape[1], 1)
    else: sc = sc.repeat_interleave(w.shape[0] // sc.shape[0], 0).repeat_interleave(w.shape[1] // sc.shape[1], 1)
    return w * sc
cfgobj = types.SimpleNamespace(virtual_heads_from={"num_attention_heads": 64, "o_groups": 8})
try: vc = VllmConfig()
except Exception as e: print("VllmConfig() failed, continuing without:", repr(e)[:120]); vc = None
ok = True; torch.manual_seed(0); RES = []
for pre in ("layers.5.attn", "mtp.0.attn"):
    ck = {n: (get(f"{pre}.{n}.weight"), get(f"{pre}.{n}.scale")) for n in ("wq_b", "wo_a", "wo_b")}
    ref = {"wq_b": deq_ckpt(VH.pad_tensor(ck["wq_b"][0], 0, 72*HD, "dup"), VH.pad_tensor(ck["wq_b"][1], 0, 72*HD//32, "dup")),
           "wo_a": deq_ckpt(VH.pad_tensor(ck["wo_a"][0], 0, PG*OL, "dup"), VH.pad_tensor(ck["wo_a"][1], 0, PG*OL//32, "dup")),
           "wo_b": deq_ckpt(VH.pad_tensor(ck["wo_b"][0], 1, PG*OL, "zero"), VH.pad_tensor(ck["wo_b"][1], 1, PG*OL//32, "zero"))}
    for r in range(TP):
        set_rank(r)
        ctx = set_current_vllm_config(vc) if vc is not None else torch.no_grad()
        with ctx, torch.device("cuda"):
            wq_b = L.ColumnParallelLinear(1280, 72*HD, bias=False, quant_config=qc, return_bias=False, prefix=f"model.{pre}.wq_b", params_dtype=torch.bfloat16)
            wo_a = L.ColumnParallelLinear(72*HD//PG, PG*OL, bias=False, quant_config=qc, return_bias=False, prefix=f"model.{pre}.wo_a", params_dtype=torch.bfloat16)
            wo_a.is_bmm = True; wo_a.bmm_batch_size = PG // TP
            wo_b = L.RowParallelLinear(PG*OL, 5120, bias=False, quant_config=qc, return_bias=False, prefix=f"model.{pre}.wo_b", params_dtype=torch.bfloat16, reduce_results=False)
        n = VH.wrap_attention(types.SimpleNamespace(wq_b=wq_b, wo_a=wo_a, wo_b=wo_b, n_groups=PG), TP, cfgobj)
        for name, lin in (("wq_b", wq_b), ("wo_a", wo_a), ("wo_b", wo_b)):
            ps = dict(lin.named_parameters()); sname = next(k for k in ("weight_scale_inv", "weight_scale") if k in ps)
            try:
                ps["weight"].weight_loader(ps["weight"], ck[name][0]); ps[sname].weight_loader(ps[sname], ck[name][1])
            except Exception:
                print(f"{pre} r{r} {name}: LOADER FAILED"); traceback.print_exc(limit=3); ok = False; continue
            W, S = ps["weight"].data, ps[sname].data
            got = deq_loaded(W, S)
            if name == "wo_b": exp = ref[name][:, r*W.shape[1]:(r+1)*W.shape[1]]
            else: exp = ref[name][r*W.shape[0]:(r+1)*W.shape[0]]
            err = ((got - exp).abs().max() / exp.abs().max()).item() if exp.abs().max() > 0 else (got.abs().max().item())
            good = got.shape == exp.shape and err < 1e-6 and torch.isfinite(got).all().item()
            ok &= good
            RES.append((f"{pre} r{r} {name} load", err, good))
        for name, lin in (("wq_b", wq_b), ("wo_a", wo_a), ("wo_b", wo_b)):
            try: lin.quant_method.process_weights_after_loading(lin)
            except Exception: print(f"{pre} r{r} {name}: POSTPROCESS FAILED"); traceback.print_exc(limit=3); ok = False
        for name, lin, din in (("wq_b", wq_b, 1280), ("wo_b", wo_b, PG*OL//TP)):
            x = torch.randn(4, din, dtype=torch.bfloat16, device="cuda")
            try:
                y = lin(x).float().cpu()
                Wr = ref[name][r*lin.output_size_per_partition:(r+1)*lin.output_size_per_partition] if name == "wq_b" else ref[name][:, r*din:(r+1)*din]
                yr = x.float().cpu() @ Wr.T
                e = ((y - yr).abs().max() / yr.abs().max()).item()
                good = e < 5e-2 and torch.isfinite(y).all().item(); ok &= good
                RES.append((f"{pre} r{r} {name} fwd", e, good))
            except Exception: print(f"{pre} r{r} {name}: FORWARD FAILED"); traceback.print_exc(limit=3); ok = False
for k, e, g in RES: print(f"{k:28s} rel err {e:.1e} {'OK' if g else 'BAD'}")
print(f"{sum(g for _,_,g in RES)}/{len(RES)} checks OK;", "ALL PASS" if ok else "SOME FAILED")
