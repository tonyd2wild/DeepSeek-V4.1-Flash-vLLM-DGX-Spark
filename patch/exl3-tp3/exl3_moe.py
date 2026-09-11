"""EXL3 mixture-of-experts method for vLLM.

Routed experts are executed as a grouped GEMM. vLLM's `moe_align_block_size`
sorts the (token, expert) pairs by expert and pads each expert's run out to a
whole block, so every row block belongs to exactly one expert -- which is what
lets the GEMM offset the trellis by a single expert id per block rather than per
row.

The EXL3 wrinkle is that `suh` is per expert *and* per shard, and it sits inside
the input Hadamard, so the activation transform has to happen after routing:
each routed row is transformed with its own expert's scales. That is what
`exl3_moe_had_in` does, gathering and transforming in one pass.
"""

from __future__ import annotations

import os
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.utils import set_weight_attrs
from cuda_exl3 import env as _env
from cuda_exl3 import ops as _ops

logger = init_logger(__name__)

TILE = 16

# CUDA_EXL3_MOE_SKIP_PAD=0/1 forces the padding-row skip, for measuring both
# arms inside one binary -- which is how the gate above turned out to be wrong.
_SKIP_PAD_ENV = _env.getenv("CUDA_EXL3_MOE_SKIP_PAD")
_SKIP_PAD_OVERRIDE = None if not _SKIP_PAD_ENV else _SKIP_PAD_ENV not in ("0", "", "false")


# vLLM has two MoE weight-loading protocols and which one runs depends on the
# version. Older builds call `layer.load_weights(...)` once with every expert
# tensor; newer ones (RoutedExperts) call a per-parameter `weight_loader` with
# MoE kwargs and only if it advertises `supports_moe_loading`. Missing the second
# one is silent: the parameters simply stay at whatever `torch.empty` left, and
# the model generates fluent nonsense rather than failing. Support both.
_SHARD_TO_PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}


def _exl3_moe_weight_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str | None = None,
    shard_id: str | None = None,
    expert_id: int | None = None,
    return_success: bool = False,
    **kwargs,
):
    method = getattr(param, "_exl3_method", None)
    layer = getattr(param, "_exl3_layer", None)
    if method is None or layer is None:
        return False if return_success else None

    # Models route non-expert tensors through the same weight_loader with no MoE
    # kwargs at all. Nothing of ours belongs to that path, so decline quietly
    # rather than raising a TypeError from a missing argument.
    if shard_id is None or expert_id is None:
        return False if return_success else None

    proj = _SHARD_TO_PROJ.get(str(shard_id))
    if proj is None:
        return False if return_success else None

    local = method._local_expert(layer, int(expert_id))
    if local < 0:                       # this expert lives on another rank
        return False if return_success else None

    # The name handed over is the *parameter* name, because the caller has
    # already rewritten "experts.<n>.gate_proj." to "w13_". Either form ends in
    # the EXL3 suffix (trellis / suh / svh / mcg), so strip the prefix if it
    # survived and take what is left.
    tail = str(weight_name or "").rstrip(".").rsplit(".", 1)[-1]
    for pre in ("w13_", "w2_"):
        if tail.startswith(pre):
            tail = tail[len(pre):]
            break
    ok = method._place(layer, local, proj, tail, loaded_weight) is not None
    return ok if return_success else None


# Model loaders check this before routing expert tensors through weight_loader
# with MoE kwargs; without it the parameter is loaded as a plain dense tensor.
_exl3_moe_weight_loader.supports_moe_loading = True


# -- clamped SwiGLU (DeepSeek V4.1) -------------------------------------------
#
# The reference (inference/model.py, Expert.forward):
#     gate = w1(x).float(); up = w3(x).float()
#     if swiglu_limit > 0:
#         up = clamp(up, -limit, limit); gate = clamp(gate, max=limit)
#     x = silu(gate) * up
# The fused kernel applies exactly this, in fp32, before the down-projection's
# input Hadamard. `limit <= 0` (the default) is the unclamped SwiGLU.


def clamped_swiglu_reference(inter: torch.Tensor, limit: float | None) -> torch.Tensor:
    """silu(min(g, limit)) * clamp(u, -limit, limit) in fp32; `inter` is (rows, 2I)
    with gate in the first half. The math the fused kernel implements."""
    I = inter.shape[-1] // 2
    g = inter[..., :I].float()
    u = inter[..., I:].float()
    if limit is not None and limit > 0:
        g = torch.clamp(g, max=limit)
        u = torch.clamp(u, min=-limit, max=limit)
    return torch.nn.functional.silu(g) * u


_GLU_KERNEL_TAKES_LIMIT: bool | None = None


def _glu_kernel_takes_limit(ops) -> bool:
    """Whether the compiled `exl3_moe_glu_had_in` has the `limit` argument.

    A binary built before the clamp landed accepts the old six arguments only;
    in that case the clamp is applied in torch before the call, so a stale
    build degrades to an extra elementwise pass rather than to wrong numerics.
    """
    global _GLU_KERNEL_TAKES_LIMIT
    if _GLU_KERNEL_TAKES_LIMIT is None:
        try:
            op = ops.exl3_moe_glu_had_in
            schema = str(getattr(op, "default", op)._schema)
            _GLU_KERNEL_TAKES_LIMIT = "limit" in schema
        except Exception:
            _GLU_KERNEL_TAKES_LIMIT = False
        if not _GLU_KERNEL_TAKES_LIMIT:
            logger.warning_once(
                "cuda-exl3: compiled exl3_moe_glu_had_in has no `limit` argument; "
                "swiglu_limit models will clamp in torch before the kernel (slower). "
                "Rebuild the extension."
            )
    return _GLU_KERNEL_TAKES_LIMIT


def _glu_had_in(ops, inter, a2, w2_suh, expert_ids, n_rows, block_m, limit):
    """Fused (clamped) SwiGLU + input Hadamard for the down projection."""
    if limit is None or limit <= 0:
        ops.exl3_moe_glu_had_in(inter, a2, w2_suh, expert_ids, n_rows, block_m)
        return
    if _glu_kernel_takes_limit(ops):
        ops.exl3_moe_glu_had_in(inter, a2, w2_suh, expert_ids, n_rows, block_m,
                                float(limit))
        return
    # Fallback: clamp the GEMM output in place (it is consumed only here), then
    # run the unclamped kernel. silu is monotone-safe under the min, so
    # silu(min(g, L)) == silu(g') with g' = min(g, L).
    I = inter.shape[1] // 2
    inter[:, :I].clamp_(max=limit)
    inter[:, I:].clamp_(min=-limit, max=limit)
    ops.exl3_moe_glu_had_in(inter, a2, w2_suh, expert_ids, n_rows, block_m)


class Exl3MoEMethod(FusedMoEMethodBase):
    """Routed experts with EXL3-quantized w13 / w2."""

    def __init__(self, moe, quant_config, gate_info, up_info, down_info, prefix: str,
                 swiglu_limit: float | None = None):
        super().__init__(moe)
        self.quant_config = quant_config
        self.prefix = prefix
        # DeepSeek V4.1: silu(min(g, L)) * clamp(u, -L, L) with L = 10 (the
        # reference Expert.forward); None keeps the plain silu(g) * u of every
        # other EXL3 model. Read off the layer by Exl3Config.get_quant_method.
        self.swiglu_limit = (
            float(swiglu_limit) if swiglu_limit is not None and float(swiglu_limit) > 0
            else None
        )

        bits = {gate_info.bits, up_info.bits}
        if len(bits) != 1:
            raise ValueError(
                f"EXL3 {prefix}: gate and up are quantized at different bitrates "
                f"{sorted(bits)}; they share one fused trellis and cannot differ."
            )
        self.w13_bits = bits.pop()
        self.w2_bits = down_info.bits
        cbs = {gate_info.cb, up_info.cb, down_info.cb}
        if len(cbs) != 1:
            raise ValueError(f"EXL3 {prefix}: experts use different codebooks")
        self.cb = cbs.pop()
        self.cb_name = {1: "mcg", 2: "mul1"}.get(self.cb)

    # -- weights ---------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        E = num_experts
        H = hidden_size
        # dsv41 P7: exl3_moe_gemm needs every N-shard to be a multiple of its
        # 128-wide block. vLLM hands us I_total/tp, which is not always one
        # (DeepSeek-V4.1-Flash: 2304/4 = 576 = 4.5 blocks; GLM-5.3: 1536/4 =
        # 384 is fine). Split the intermediate dim unevenly on 128 boundaries
        # instead: 2304 = 18 blocks -> [4,5,5,4] over four ranks. The TP
        # reduction sums partial down-projections, so ranks may hold different
        # widths. Extra blocks go to the non-head ranks so rank 0 (API server +
        # head) keeps the most room for KV.
        I, I_off = self._tp_split(intermediate_size_per_partition)
        dev = torch.cuda.current_device()

        for name, dims in [
            ("w13_trellis", (E, H // TILE, 2 * I // TILE, TILE * self.w13_bits)),
            ("w2_trellis", (E, I // TILE, H // TILE, TILE * self.w2_bits)),
        ]:
            p = torch.nn.Parameter(
                torch.empty(dims, dtype=torch.int16, device=dev), requires_grad=False
            )
            layer.register_parameter(name, p)
            set_weight_attrs(p, {**extra_weight_attrs})

        for name, dims in [
            ("w13_suh", (E, 2, H)),      # gate and up have separate input scales
            ("w2_suh", (E, 1, I)),
            ("w13_svh", (E, 2 * I)),
            ("w2_svh", (E, H)),
        ]:
            p = torch.nn.Parameter(
                torch.empty(dims, dtype=torch.half, device=dev), requires_grad=False
            )
            layer.register_parameter(name, p)
            set_weight_attrs(p, {**extra_weight_attrs})

        # The codebook multiplier is a compile-time constant of the codebook id,
        # but it is still a tensor in the checkpoint, so it needs somewhere to go.
        if self.cb_name:
            for name, dims in [(f"w13_{self.cb_name}", (E, 2)),
                               (f"w2_{self.cb_name}", (E, 1))]:
                p = torch.nn.Parameter(
                    torch.empty(dims, dtype=torch.int32, device=dev), requires_grad=False
                )
                layer.register_parameter(name, p)
                set_weight_attrs(p, {**extra_weight_attrs})

        for pname, p in list(layer.named_parameters(recurse=False)):
            p._exl3_name = pname
            p._exl3_method = self
            p._exl3_layer = layer
            # Plain setattr, not set_weight_attrs: extra_weight_attrs has
            # usually already installed vLLM's generic loader and
            # set_weight_attrs asserts rather than overwrite. Ours has to win --
            # it is the one that understands the four EXL3 tensors behind each
            # projection.
            p.weight_loader = _exl3_moe_weight_loader
        self._install_loader(layer)

        layer.exl3_num_experts = E
        layer.exl3_hidden = H
        layer.exl3_inter = I
        layer.exl3_inter_off = I_off
        layer.exl3_cb = self.cb
        layer.exl3_w13_bits = self.w13_bits
        layer.exl3_w2_bits = self.w2_bits

    def _tp_split(self, I_even: int) -> tuple[int, int]:
        """(this rank's intermediate width, its column offset in the full dim).

        Even split when it is already 128-aligned or cannot be fixed; otherwise
        an uneven 128-block split with the remainder on ranks 1.. (never rank 0).
        """
        tp, r = self._tp_size, self._tp_rank
        total = I_even * tp
        if tp <= 1 or I_even % 128 == 0 or total % 128 != 0:
            return I_even, r * I_even
        nblk = total // 128
        base, rem = divmod(nblk, tp)
        counts = [base] * tp
        for i in range(rem):
            counts[1 + i] += 1
        off = sum(counts[:r]) * 128
        I = counts[r] * 128
        logger.info_once(
            "EXL3 %s: I_total/tp = %d is not a multiple of 128; uneven TP split "
            "of the expert intermediate dim %d over %d ranks = %s columns "
            "(this rank: %d at offset %d)",
            self.prefix, I_even, total, tp, str([c * 128 for c in counts]), I, off,
        )
        return I, off

    def _install_loader(self, layer):
        """Take over weight loading for this module.

        vLLM's MoE loader decides `is_fused = loaded_weight.dim() == 3`, i.e. it
        reads a 3-D tensor as "all experts stacked". An EXL3 trellis is
        genuinely 3-D *per expert* (k/16, n/16, 16*bits), so that heuristic
        slices it apart along the wrong axis. There is no hook before that
        decision, so this module loads its own weights.
        """
        method = self

        def load_weights(weights):
            loaded: set[str] = set()
            for name, w in weights:
                parts = name.split(".")
                if len(parts) < 3 or not parts[0].isdigit():
                    continue                      # not a per-expert tensor
                expert_id, proj, suffix = int(parts[0]), parts[1], parts[-1]
                # DeepSeek checkpoints name the projections w1/w3/w2; _place
                # speaks gate_proj/up_proj/down_proj.
                proj = _SHARD_TO_PROJ.get(proj, proj)
                local = method._local_expert(layer, expert_id)
                if local < 0:
                    continue                      # expert lives on another rank
                pname = method._place(layer, local, proj, suffix, w)
                if pname:
                    loaded.add(pname)
            return loaded

        layer.load_weights = load_weights

    @staticmethod
    def _local_expert(layer, expert_id: int) -> int:
        emap = getattr(layer, "expert_map", None)
        if emap is None:
            return expert_id
        return int(emap[expert_id].item()) if expert_id < emap.numel() else -1

    def _place(self, layer, e: int, proj: str, suffix: str, w: torch.Tensor):
        """Copy one checkpoint tensor into the fused per-expert parameter.

        gate/up occupy the two halves of w13 along the output dim, matching how
        vLLM lays out w13 everywhere else. Tensor-parallel slicing is on the
        intermediate dim: the output dim for gate/up, the input dim for down.
        """
        r = self._tp_rank
        # column offset of this rank's slice of the intermediate dim (P7: may be
        # an uneven split; falls back to r * width for older layers)
        off = getattr(layer, "exl3_inter_off", None)
        if off is None:
            off = r * layer.exl3_inter
        if proj == "down_proj":
            if suffix == "trellis":
                p = layer.w2_trellis
                part = p.shape[1]
                p.data[e].copy_(w.narrow(0, off // TILE, part))
            elif suffix == "suh":
                p = layer.w2_suh
                part = p.shape[2]
                p.data[e][0].copy_(w.narrow(0, off, part))
            elif suffix == "svh":
                p = layer.w2_svh
                p.data[e].copy_(w)
            else:
                p = getattr(layer, f"w2_{self.cb_name}", None)
                if p is None:
                    return None
                p.data[e][0] = int(w.reshape(-1)[0])
            return p._exl3_name

        half = 0 if proj == "gate_proj" else 1
        if suffix == "trellis":
            p = layer.w13_trellis
            part = p.shape[2] // 2
            p.data[e].narrow(1, half * part, part).copy_(w.narrow(1, off // TILE, part))
        elif suffix == "suh":
            p = layer.w13_suh
            p.data[e][half].copy_(w)
        elif suffix == "svh":
            p = layer.w13_svh
            part = p.shape[1] // 2
            p.data[e].narrow(0, half * part, part).copy_(w.narrow(0, off, part))
        else:
            p = getattr(layer, f"w13_{self.cb_name}", None)
            if p is None:
                return None
            p.data[e][half] = int(w.reshape(-1)[0])
        return p._exl3_name

    @property
    def _tp_size(self) -> int:
        return int(getattr(self.moe, "tp_size", 1) or 1)

    @property
    def _tp_rank(self) -> int:
        # Defaulting a missing rank to 0 would make every rank load shard 0:
        # wrong output on every rank but the first, with no error anywhere. Only
        # tolerate that when there is genuinely one shard.
        r = getattr(self.moe, "tp_rank", None)
        if r is None:
            if self._tp_size > 1:
                raise RuntimeError(
                    f"EXL3 {self.prefix}: cannot determine the tensor-parallel rank "
                    f"(tp_size={self._tp_size}); vLLM's FusedMoEConfig no longer "
                    "exposes tp_rank. Refusing to load, as guessing rank 0 would "
                    "silently give every rank the same shard."
                )
            return 0
        return int(r)

    def process_weights_after_loading(self, layer):
        # The expert gemm splits k when its grid is too small to fill the GPU,
        # which needs an fp32 accumulator sized from the routed row count -- and
        # that is not known until the forward runs. The workspace refuses to grow
        # once CUDA graphs are being captured, so claim the ceiling now. The gemm
        # caps its own split at this many elements and runs unsplit above it, so
        # reserving the cap makes growth impossible rather than merely unlikely.
        try:
            cap = int(torch.ops.cuda_exl3_C.exl3_get_moe_acc_cap())
            if cap <= 0:
                return          # split-k off: no accumulator will ever be needed
            torch.ops.cuda_exl3_C.exl3_reserve_acc(layer.w13_svh.data, cap)
        except Exception as e:  # pragma: no cover - lazy sizing is still correct
            logger.debug("EXL3 %s: MoE accumulator pre-reserve skipped (%s)",
                         self.prefix, e)

    def get_fused_moe_quant_config(self, layer):
        # Not using vLLM's modular kernel framework; apply() runs the whole thing.
        return None

    # -- execution -------------------------------------------------------

    @staticmethod
    def _block_m(rows: int, num_experts: int) -> int:
        """Row-block size, which is also the alignment granularity.

        Must be one of the GEMM's BM tiers. Larger blocks amortize the weight
        read over more rows, but each expert's run is padded up to a whole block,
        so with few tokens per expert the padding dominates.
        """
        import os

        forced = _env.getenv("CUDA_EXL3_MOE_BLOCK_M")
        if forced:
            return int(forced)
        # Measured: doubling the block costs 8-25% at low concurrency and up to
        # 2x at block=128, because every expert's run is padded up to a whole
        # block. Weight traffic is unchanged (the same experts are read either
        # way), which is why the cost is sublinear rather than proportional --
        # but it is not free, so use the smallest block until there are enough
        # tokens per expert to fill a larger one.
        per_expert = rows / max(num_experts, 1)
        if per_expert < 16:
            return 16
        if per_expert < 48:
            return 32
        return 64 if per_expert < 96 else 128

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        if shared_experts is not None:
            # This method is not a modular kernel, so it cannot overlap the
            # shared expert; SharedExperts.forward no-ops unless the order it
            # picked matches. Call it defensively (as the modular path does) and
            # return only the routed output -- the caller collects the shared
            # result from the SharedExperts object itself.
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
                SharedExpertsOrder,
            )

            shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)

        M, H = x.shape
        T = topk_ids.shape[1]
        E = layer.exl3_num_experts
        I = layer.exl3_inter
        out_dtype = x.dtype

        # Under expert parallel this layer holds only its own slice of the
        # experts, but topk_ids stay global, and moe_align_block_size wants the
        # global count: it buckets by global id and only then maps through
        # expert_map, marking every block this rank does not own with -1. Pass
        # the local count and the global ids land in the wrong buckets; pass no
        # map and the -1 never appears, so the kernels' `e < 0` guards -- which
        # exist precisely for this -- never fire. Without EP the map is None and
        # the global count is the local one, which is why this hid for so long.
        emap = getattr(layer, "expert_map", None)
        e_global = int(emap.numel()) if emap is not None else E
        # The block size wants the global count for the same reason. `rows` here
        # is every (token, expert) pair, but only E_local/E_global of them are
        # this rank's, so local occupancy is rows*(E_local/E_global)/E_local =
        # rows/E_global -- the local count would overstate it by the EP factor
        # and buy a block one or two tiers too large, which is pure padding.
        block_m = self._block_m(M * T, e_global)
        # pad_sorted_ids makes sorted_ids a whole number of blocks; without it
        # expert_ids covers more blocks than sorted_ids has entries and the
        # gather reads past the end.
        sorted_ids, expert_ids, n_rows = moe_align_block_size(
            topk_ids, block_m, e_global, expert_map=emap, pad_sorted_ids=True
        )
        sorted_ids = sorted_ids.int()
        expert_ids = expert_ids.int()
        n_rows = n_rows.int()
        rows = min(expert_ids.numel() * block_m, sorted_ids.numel())
        expert_ids = expert_ids[: rows // block_m]

        xc = x.contiguous()
        ops = torch.ops.cuda_exl3_C

        # A routed block is block_m rows but decode routes about one row per
        # expert, so most rows are padding carrying zeros. The gemm skips
        # fetching them -- cp.async zero-fills a row it is not given -- and the
        # transform then need not write them.
        #
        # This was gated on grid size for a while, on the strength of a 3.6% loss
        # at one token. That number came from comparing two separately built
        # packages; forcing both arms inside one binary shows the skip winning
        # there too. Whole MoE layer, TP=4 shapes, 42 layers:
        #   M=1 +2.5%   M=2 +0.4%   M=4 +1.1%   M=8 +6.0%
        #   M=16 +9.7%  M=32 +9.7%  M=64 +11.1%  M=256 +5.5%
        # and on a 48-SM part (#1): +1.9 to +3.2% at every M, no crossover. So
        # the gate only ever turned off a win, on both parts. One fewer constant
        # fitted to one card.
        skip_pad = True
        if _SKIP_PAD_OVERRIDE is not None:
            skip_pad = _SKIP_PAD_OVERRIDE

        # gate/up: two transforms per routed row, one per shard's suh
        a13 = torch.empty((2, rows, H), dtype=torch.half, device=x.device)
        ops.exl3_moe_had_in(xc, a13, layer.w13_suh.data, sorted_ids, expert_ids,
                            n_rows, block_m, T, M * T, skip_pad)
        inter = ops.exl3_moe_gemm(a13, layer.w13_trellis.data, layer.w13_suh.data,
                                  layer.w13_svh.data, expert_ids, n_rows, [I, I],
                                  layer.exl3_cb, block_m, out_dtype,
                                  sorted_ids if skip_pad else None, None, M, T)

        # down: SwiGLU folded into the input transform. Doing it separately
        # materialised a (rows, I) tensor that the transform then read straight
        # back -- a full round trip through memory for no reuse. The rows are
        # already in routed order here, so the gather is the identity.
        a2 = torch.empty((1, rows, I), dtype=torch.half, device=x.device)
        _glu_had_in(ops, inter, a2, layer.w2_suh.data, expert_ids, n_rows, block_m,
                    self.swiglu_limit)
        # The down projection finishes the MoE: its epilogue scales each routed
        # row by the routing weight and accumulates it into the token's row, so
        # there is no (rows, H) tensor and no combine kernel. That kernel ran one
        # block per token -- eight blocks at decode -- and was pure latency.
        #
        # That epilogue accumulates with atomics, so the order in which a
        # token's top-k rows are summed varies run to run and the output is
        # reproducible in value but not bit-exact -- the same trade split-k
        # makes on the dense path. CUDA_EXL3_DETERMINISTIC promises bit-exact
        # everywhere, so under it the down projection writes routed rows and
        # the separate combine sums them in a fixed k order instead.
        if _ops.DETERMINISTIC:
            rows_out = ops.exl3_moe_gemm(a2, layer.w2_trellis.data,
                                         layer.w2_suh.data, layer.w2_svh.data,
                                         expert_ids, n_rows, [H], layer.exl3_cb,
                                         block_m, out_dtype, sorted_ids, None,
                                         M, T)
            return ops.exl3_moe_combine(rows_out, sorted_ids, topk_weights, M,
                                        expert_ids, block_m)

        out = ops.exl3_moe_gemm(a2, layer.w2_trellis.data, layer.w2_suh.data,
                                layer.w2_svh.data, expert_ids, n_rows, [H],
                                layer.exl3_cb, block_m, out_dtype,
                                sorted_ids, topk_weights, M, T)

        return out
