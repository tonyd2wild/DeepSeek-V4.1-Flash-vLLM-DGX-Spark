# SPDX-License-Identifier: Apache-2.0
"""Tech2Wild/Kai 2026-09-14: native MXFP8 wo_a on SM12x (GB10).

Why: wo_a is a grouped (``is_bmm``) MXFP8 linear. After load, ModelOpt
re-selects its kernel from [DeepGemmMxfp8Bmm (SM100 only), Emulation], so on
SM12x it lands on Emulation, which dequantizes wo_a to BF16 at load and
o_proj runs ``torch.bmm`` over 2x the weight bytes on every step.

What: with ``DSV41_WOA_B12X`` set, the attention layer's post-load hook (runs
after every quant method) splits wo_a into one stand-alone MXFP8 linear per
local group and packs each with a stock vLLM SM12x MXFP8 kernel:

    DSV41_WOA_B12X=1 | b12x         B12xMxfp8LinearKernel (b12x.gemm.mxfp8_linear)
    DSV41_WOA_B12X=fi | flashinfer  FlashInferCutlassMxfp8LinearKernel (mm_mxfp8)
    unset / 0                       no change (Emulation BF16 bmm)

Reusing the vLLM kernel classes means we inherit their weight packing, their
torch.compile custom-op boundaries, and (b12x) their JIT warm-up: the group
modules carry ``b12x_warmup_provider`` and sit under the model, so
``b12x_warmup`` compiles them for every CUDA-graph capture size before capture.

o_proj.py checks ``wo_a.dsv41_woa_groups`` first (see woa_grouped_forward).
Any precondition failure logs a warning and leaves the default path untouched.
"""

import os

import torch
import torch.nn as nn
from torch.nn import Parameter

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "DSV41_WOA_B12X"


def woa_backend() -> str | None:
    v = os.environ.get(ENV, "").strip().lower()
    if v in ("", "0", "false", "off", "no"):
        return None
    if v in ("1", "true", "on", "yes", "b12x"):
        return "b12x"
    if v in ("fi", "flashinfer", "cutlass"):
        return "flashinfer"
    raise ValueError(f"{ENV}={v!r}: expected 0, 1/b12x or fi/flashinfer")


def _kernel_cls(backend: str):
    if backend == "b12x":
        from vllm.model_executor.kernels.linear.mxfp8.b12x import (
            B12xMxfp8LinearKernel,
        )

        return B12xMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )

    return FlashInferCutlassMxfp8LinearKernel


class WoaGroup(nn.Module):
    """One local wo_a group, [o_lora_rank, K] FP8 + [o_lora_rank, K/32] E8M0,
    driven by a stock vLLM MXFP8 linear kernel."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, kernel):
        super().__init__()
        self.weight = Parameter(weight, requires_grad=False)
        self.weight_scale = Parameter(weight_scale, requires_grad=False)
        self.kernel = kernel  # plain object attribute, not a submodule
        kernel.process_weights_after_loading(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.apply_weights(self, x)


@torch.no_grad()
def restore_fp8_weight(wo_a: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return wo_a's MXFP8 (weight [N,K] e4m3, scale [N,K/32] uint8), or None.

    Emulation (VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=1, the default) replaced the
    weight with fp8 * 2^(s-127) in BF16 but kept the E8M0 scale. Each e4m3 value
    times a power of two is exact in BF16, so dividing the scale back out gives
    the checkpoint bytes; the round trip is verified (load time, host sync OK).
    """
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE,
        MXFP8_SCALE_DTYPE,
        MXFP8_VALUE_DTYPE,
        dequant_mxfp8_to_bf16,
    )

    w = wo_a.weight.data
    s = getattr(wo_a, "weight_scale", None)
    if s is None or w.ndim != 2 or s.ndim != 2 or s.dtype != MXFP8_SCALE_DTYPE:
        return None
    n, k = w.shape
    if k % MXFP8_BLOCK_SIZE:
        return None
    s = s.data[:n, : k // MXFP8_BLOCK_SIZE].contiguous()
    if w.dtype == MXFP8_VALUE_DTYPE:
        return w.contiguous(), s
    if w.dtype not in (torch.bfloat16, torch.float16):
        return None
    descale = torch.exp2(s.float() - 127.0).repeat_interleave(MXFP8_BLOCK_SIZE, dim=1)
    w_fp8 = (w.float() / descale).to(MXFP8_VALUE_DTYPE)
    del descale
    if not torch.equal(dequant_mxfp8_to_bf16(w_fp8, s).to(w.dtype), w):
        return None
    return w_fp8, s


def maybe_install_woa_sm12x(attn) -> None:
    """Post-load: replace the emulated wo_a with per-group native MXFP8 linears.

    ``attn`` needs ``wo_a``, ``n_local_groups`` and ``o_lora_rank``.
    """
    backend = woa_backend()
    if backend is None:
        return
    wo_a = attn.wo_a
    if getattr(wo_a, "dsv41_woa_groups", None) is not None:
        return  # already installed (e.g. second post-load pass)

    from vllm.platforms import current_platform

    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    ):
        logger.warning_once("%s set but device is not SM12x; wo_a unchanged.", ENV)
        return
    kernel_cls = _kernel_cls(backend)
    ok, why = kernel_cls.is_supported()
    if not ok:
        logger.warning_once("%s: %s unsupported (%s); wo_a unchanged.", ENV,
                            kernel_cls.__name__, why)
        return
    got = restore_fp8_weight(wo_a)
    if got is None:
        logger.warning_once(
            "%s: wo_a weight not recoverable as MXFP8 (dtype=%s shape=%s); "
            "wo_a unchanged.", ENV, wo_a.weight.dtype, tuple(wo_a.weight.shape))
        return
    w, s = got
    groups, rank = int(attn.n_local_groups), int(attn.o_lora_rank)
    if w.shape[0] != groups * rank:
        logger.warning_once("%s: wo_a rows %d != %d groups x %d; wo_a unchanged.",
                            ENV, w.shape[0], groups, rank)
        return

    from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig

    # Row slices of a contiguous [N,K] are contiguous views: no extra copy; the
    # per-group kernels own the (single) FP8 buffer from here on.
    mods = nn.ModuleList(
        WoaGroup(
            w[g * rank : (g + 1) * rank],
            s[g * rank : (g + 1) * rank].contiguous(),
            kernel_cls(Mxfp8LinearLayerConfig()),
        )
        for g in range(groups)
    )
    old_w = wo_a.weight
    # Drop the BF16 copy (the 2x bytes). Empty params keep attribute access
    # (e.g. dtype checks) working; weight *reload* of wo_a is not supported here.
    wo_a.weight = Parameter(old_w.data.new_empty((0,)), requires_grad=False)
    wo_a.weight_scale = Parameter(s.new_empty((0,)), requires_grad=False)
    del old_w
    wo_a.dsv41_woa_groups = mods  # registers as a submodule (seen by b12x warmup)
    logger.info_once(
        "DSV41 wo_a: native MXFP8 via %s, %d groups x [%d, %d] per rank "
        "(replaces BF16 emulation).", kernel_cls.__name__, groups, rank, w.shape[1])
