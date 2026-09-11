# SPDX-License-Identifier: Apache-2.0
"""Tech2Wild/Kai 2026-09-11: "virtual heads" for DeepSeek-V4.1-Flash at tensor-parallel 3.

V4.1 has 64 attention heads in 8 output groups (8 heads per group). Neither divides by 3, so a
TP3 serving copy of the model declares 72 heads / 9 o_groups in its config.json (whole groups
per rank: 24 heads = 3 groups). The checkpoint still holds 64 heads, so the full wq_b / wo_a /
wo_b tensors (and their block scales) are padded here, before vLLM slices them per rank:

  wq_b  (column-parallel, rows = heads x head_dim)   pad rows by repeating the last real group
  wo_a  (column-parallel, rows = groups x o_lora)    pad rows by repeating the last real group
  wo_b  (row-parallel,    cols = groups x o_lora)    pad cols with ZERO weights (scale 1.0)

Only wo_b's zero columns remove the padded group from the output, so the padded heads see
ordinary (non-zero) activations everywhere upstream: no all-zero q or wo_b-input blocks reach
the FP8 dynamic quantizers. attn_sink pads itself (-inf, already handled by the model loader).
At TP sizes that divide the real head count the checkpoint already matches, so this is inert.
"""
import torch

_E8M0 = getattr(torch, "float8_e8m0fnu", None)


def pad_tensor(t: torch.Tensor, dim: int, full: int, mode: str) -> torch.Tensor:
    real = t.shape[dim]
    n = full - real
    if n <= 0:
        return t
    one_byte = t.element_size() == 1
    base = t.view(torch.uint8) if one_byte else t
    if mode == "dup":
        if n > real:
            raise ValueError(f"virtual heads: cannot repeat {n} rows from {real}")
        pad = base.narrow(dim, real - n, n)
    elif mode == "zero":
        shape = list(t.shape)
        shape[dim] = n
        if one_byte:
            fill = 127 if (_E8M0 is not None and t.dtype == _E8M0) else 0  # e8m0 127 = 2^0
            pad = torch.full(shape, fill, dtype=torch.uint8, device=t.device)
        else:
            pad = torch.zeros(shape, dtype=t.dtype, device=t.device)
    else:
        raise ValueError(mode)
    out = torch.cat([base, pad.to(base.device)], dim=dim)
    return out.view(t.dtype) if one_byte else out


def _virtual_from(config):
    """{"num_attention_heads": 64, "o_groups": 8} from the TP3 config.json, or None."""
    for c in (config, getattr(config, "text_config", None)):
        v = getattr(c, "virtual_heads_from", None) if c is not None else None
        if v:
            return v if isinstance(v, dict) else dict(v)
    return None


def _wrap(param: torch.nn.Parameter, dim: int, real_g: int, pad_g: int, mode: str) -> bool:
    orig = getattr(param, "weight_loader", None)
    if orig is None or dim is None:
        return False

    def loader(p, loaded_weight, *args, **kwargs):
        # Pad in CHECKPOINT units: the incoming tensor holds real_g groups along dim
        # (weights and block scales alike), the model wants pad_g. vLLM may store the
        # param in another layout (e.g. per-row expanded scales), so never size from it.
        if loaded_weight.dim() > dim and loaded_weight.shape[dim] % real_g == 0:
            full = loaded_weight.shape[dim] // real_g * pad_g
            loaded_weight = pad_tensor(loaded_weight, dim, full, mode)
        return orig(p, loaded_weight, *args, **kwargs)

    try:
        param.weight_loader = loader
    except AttributeError:  # vLLM v2 parameters expose weight_loader as a property
        param._weight_loader = loader
    return True


def wrap_attention(attn, tp_size: int, config=None) -> int:
    """Install the padding loaders on attn.wq_b / wo_a / wo_b when the config declares
    virtual heads (virtual_heads_from). Returns the number of params wrapped."""
    src = _virtual_from(config)
    if tp_size <= 1 or not src:
        return 0
    real_g, pad_g = int(src["o_groups"]), int(attn.n_groups)
    if pad_g <= real_g:
        return 0
    wrapped = 0
    for lin, attr, mode in ((attn.wq_b, "output_dim", "dup"),
                            (attn.wo_a, "output_dim", "dup"),
                            (attn.wo_b, "input_dim", "zero")):
        for p in lin.parameters(recurse=False):
            dim = getattr(p, attr, None)
            if dim is None:  # plain Parameter: [out, in] layout
                dim = 0 if attr == "output_dim" else 1
            wrapped += _wrap(p, dim, real_g, pad_g, mode)
    return wrapped


# Tech2Wild/Kai 2026-09-11 (diagnostic, exl3tp3a5): with KAI_STACK_SAMPLER=1, every worker that imports this
# module logs its own RssAnon/RssFile/RssShmem and the main thread's Python stack every 5 s to
# /cache/kai-stack-<host>-<pid>.log, so we can see which code runs while host memory climbs after loading.
def _kai_stack_sampler():
    import os, sys, threading, time, traceback
    path = f"/cache/kai-stack-{os.uname().nodename}-{os.getpid()}.log"
    main = threading.main_thread().ident
    with open(path, "a", buffering=1) as fh:
        while True:
            try:
                st = {}
                for line in open("/proc/self/status"):
                    if line.startswith(("RssAnon:", "RssFile:", "RssShmem:")):
                        k, v = line.split()[:2]; st[k[:-1]] = int(v) / 1048576
                fh.write(f"=== {time.strftime('%H:%M:%S')} anon {st.get('RssAnon', 0):.1f} file {st.get('RssFile', 0):.1f} shmem {st.get('RssShmem', 0):.1f} GiB\n")
                fr = sys._current_frames().get(main)
                for f in (traceback.extract_stack(fr)[-14:] if fr is not None else []):
                    fh.write(f"   {f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}\n")
            except Exception as e:  # never let the sampler hurt the worker
                fh.write(f"sampler error {e!r}\n")
            time.sleep(5)


def _kai_maybe_start_sampler():
    import os, sys, threading
    if os.environ.get("KAI_STACK_SAMPLER") == "1" and not getattr(sys, "_kai_sampler_started", False):
        sys._kai_sampler_started = True
        threading.Thread(target=_kai_stack_sampler, daemon=True, name="kai-stack-sampler").start()


_kai_maybe_start_sampler()
