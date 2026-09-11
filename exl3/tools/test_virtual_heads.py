"""CPU check of TP3 virtual heads on REAL V4.1 attention weights (layer 5 + MTP 0).
Reference = the 64-head / 8-group math; test = the padded 72-head / 9-group tensors sliced
exactly like vLLM's column/row-parallel loaders do at TP3, partial outputs summed (all-reduce).
The per-head "attention" is a fixed nonlinearity on q, which is all the check needs: heads are
independent between wq_b and wo_a."""
import json, sys, torch
from safetensors import safe_open
sys.path.insert(0, "/t")
from virtual_heads import pad_tensor
M = sys.argv[1]
wm = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
def get(n):
    with safe_open(f"{M}/{wm[n]}", framework="pt") as f: return f.get_tensor(n)
def deq(w, s):
    e = s.view(torch.uint8).float() - 127.0
    sc = torch.pow(2.0, e).repeat_interleave(32, 0).repeat_interleave(32, 1)
    return w.float() * sc[: w.shape[0], : w.shape[1]]
torch.manual_seed(0)
worst = 0.0
for pre in ("layers.5.attn.", "mtp.0.attn."):
    wq, sq = get(pre + "wq_b.weight"), get(pre + "wq_b.scale")
    wa, sa = get(pre + "wo_a.weight"), get(pre + "wo_a.scale")
    wb, sb = get(pre + "wo_b.weight"), get(pre + "wo_b.scale")
    Wq, Wa, Wb = deq(wq, sq), deq(wa, sa), deq(wb, sb)
    T, H, HD, G, OL = 5, 64, 512, 8, 1024
    x = torch.randn(T, 1280)
    act = lambda q: torch.tanh(q) * 3.0
    o = act((x @ Wq.T).view(T, H, HD))
    y = torch.cat([o[:, g*8:(g+1)*8].reshape(T, 8*HD) @ Wa[g*OL:(g+1)*OL].T for g in range(G)], 1)
    ref = y @ Wb.T
    TP, HP, GP = 3, 72, 9
    Wq3 = deq(pad_tensor(wq, 0, HP*HD, "dup"), pad_tensor(sq, 0, HP*HD//32, "dup"))
    Wa3 = deq(pad_tensor(wa, 0, GP*OL, "dup"), pad_tensor(sa, 0, GP*OL//32, "dup"))
    Wb3 = deq(pad_tensor(wb, 1, GP*OL, "zero"), pad_tensor(sb, 1, GP*OL//32, "zero"))
    assert torch.isfinite(Wq3).all() and torch.isfinite(Wa3).all() and torch.isfinite(Wb3).all()
    lh, lg = HP // TP, GP // TP
    out = torch.zeros_like(ref)
    for r in range(TP):
        q = (x @ Wq3[r*lh*HD:(r+1)*lh*HD].T).view(T, lh, HD)
        o_r = act(q)
        Wa_r = Wa3[r*lg*OL:(r+1)*lg*OL]
        y_r = torch.cat([o_r[:, g*8:(g+1)*8].reshape(T, 8*HD) @ Wa_r[g*OL:(g+1)*OL].T for g in range(lg)], 1)
        assert torch.isfinite(y_r).all()
        out += y_r @ Wb3[:, r*lg*OL:(r+1)*lg*OL].T
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    worst = max(worst, rel)
    print(f"{pre:14s} padded TP3 vs 64-head reference: max rel err {rel:.2e}, |ref|max {ref.abs().max():.3f}, finite {torch.isfinite(out).all().item()}")
print("PASS" if worst < 1e-5 else "FAIL", f"worst {worst:.2e}")
