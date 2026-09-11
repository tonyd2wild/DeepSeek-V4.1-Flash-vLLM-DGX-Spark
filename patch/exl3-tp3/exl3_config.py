"""vLLM quantization config for the EXL3 (ExLlamaV3) format."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from cuda_exl3 import env as _env

logger = init_logger(__name__)

# Codebook ids, matching exllamav3's `cb` template parameter.
CB_3INST = 0
CB_MCG = 1
CB_MUL1 = 2

# CUDA_EXL3_DEBUG_NAMES=1 logs every module's resolution. At warning level on
# purpose: it is opt-in, and logger.info does not reach vLLM's default log
# level, so at info it printed nothing and looked broken (reported in #5). The
# running tallies are the point -- a mapping that silently leaves half the
# attention stack in bf16 shows up as a bf16 count that keeps climbing.
_DEBUG = bool(_env.getenv("CUDA_EXL3_DEBUG_NAMES"))
_resolved: list[str] = []
_unresolved: list[str] = []
_delegated: list[str] = []

# Wrapper segments that different stacks nest in different orders.
_WRAPPER_SEGMENTS = frozenset({"model", "language_model"})

# Per-expert projection names, in (gate, up, down) order, that a checkpoint may
# use under `<experts>.<e>.`. exllamav3 quantizes HF-style models under the
# first triple; DeepSeek V4/V4.1 checkpoints (and vLLM's DeepSeek loader, which
# maps them with shard ids "w1"/"w3"/"w2") use the second.
_EXPERT_PROJ_TRIPLES: tuple[tuple[str, str, str], ...] = (
    ("gate_proj", "up_proj", "down_proj"),
    ("w1", "w3", "w2"),
)

# vLLM (>= the RoutedExperts refactor) registers the expert parameters on a
# `routed_experts` child of the MoE runner; the quant method is asked for with
# the runner's prefix, but be tolerant of being asked with the child's.
_ROUTED_EXPERTS_SUFFIX = ".routed_experts"

# Hybrid checkpoints: EXL3 routed experts on top of a natively quantized model.
# The base format is read from `original_quantization_config` (exllamav3's
# compile.py nests the prior config under this key) or forced with this env.
_BASE_QUANT_ENV = "CUDA_EXL3_BASE_QUANT"
_BASE_QUANT_OFF = ("", "0", "none", "off", "false")

# The block DeepSeek ships in config.json for V4.1-Flash. Used only when
# CUDA_EXL3_BASE_QUANT=deepseek_v4_fp8 is forced on a checkpoint that lost its
# `original_quantization_config`.
_DEEPSEEK_V4_FP8_DEFAULT = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


def _normalize(name: str) -> str:
    return ".".join(p for p in name.split(".") if p not in _WRAPPER_SEGMENTS)


def _is_shared_expert(prefix: str) -> bool:
    """`shared_experts` anywhere on the module path (DeepSeek naming)."""
    return "shared_experts" in prefix.split(".")


def _base_config_class(name: str):
    """The vLLM QuantizationConfig class registered under `name`.

    `get_quantization_config` is the registry vLLM itself uses (on CUDA it
    returns the V4.1-aware DeepseekV4FP8Config for "deepseek_v4_fp8"); the
    direct imports are fallbacks for builds where the registry entry is absent.
    """
    from vllm.model_executor.layers.quantization import get_quantization_config

    try:
        return get_quantization_config(name)
    except Exception:  # pragma: no cover - depends on the vLLM build
        pass
    if name in ("fp8", "deepseek_v4_fp8"):
        for mod in ("vllm.models.deepseek_v4_1.quant_config",
                    "vllm.models.deepseek_v4.quant_config"):
            try:
                import importlib

                return getattr(importlib.import_module(mod), "DeepseekV4FP8Config")
            except Exception:
                continue
    raise ValueError(f"EXL3: no vLLM quantization config registered as {name!r}")


class Exl3ModuleInfo:
    """Static description of one EXL3-quantized tensor from the checkpoint."""

    __slots__ = ("name", "in_features", "out_features", "bits", "cb", "cb_mult", "has_bias")

    def __init__(self, name, in_features, out_features, bits, cb, cb_mult, has_bias):
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.cb = cb
        self.cb_mult = cb_mult
        self.has_bias = has_bias

    def __repr__(self):
        return (
            f"Exl3ModuleInfo({self.name}, {self.in_features}x{self.out_features}, "
            f"bits={self.bits}, cb={self.cb})"
        )


# Layers left alone by online quantization. The router decides which experts
# run at all, so its error is not averaged away like a projection's, and the
# head is the last thing before the logits -- EXL3 checkpoints conventionally
# keep both in high precision (this one records head_bits: 16).
_ONLINE_EXCLUDE = ("lm_head", "gate", "router", "e_score_correction", "shared_head")


def _is_online_excluded(prefix: str) -> bool:
    tail = prefix.rsplit(".", 1)[-1]
    if any(tail == e or tail.startswith(e) for e in _ONLINE_EXCLUDE):
        return True
    return "lm_head" in prefix


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """EXL3 trellis quantization.

    The checkpoint stores, per quantized tensor:
      ``trellis``  int16 (in/16, out/16, 16*bits)  -- packed trellis codes
      ``suh``      fp16  (in,)                     -- input scales x sign flips
      ``svh``      fp16  (out,)                    -- output scales x sign flips
      ``mcg``/``mul1`` int32 scalar                -- procedural codebook multiplier

    ``bits`` varies per tensor (5/6/7 in practice), so the per-tensor table from
    ``quantization_config.json`` is required to size parameters at build time --
    the summary block inlined into ``config.json`` does not carry it.

    Hybrid checkpoints (routed experts in EXL3, everything else in the model's
    native format -- DeepSeek V4.1-Flash: fp8/ue8m0 [32, 32] dense + MXFP4 MTP
    experts) carry the native format's block as ``original_quantization_config``.
    Layers with no EXL3 tensors are then handed to that format's vLLM config,
    and the attributes the model reads off ``quant_config`` (``weight_block_size``,
    ``expert_dtype``...) resolve on it. ``CUDA_EXL3_BASE_QUANT=<name>`` forces
    the base format; ``CUDA_EXL3_BASE_QUANT=none`` disables delegation.
    """

    def __init__(self, full_config: dict[str, Any]):
        super().__init__()
        self.full_config = full_config
        self.version = full_config.get("version", "unknown")
        self.default_bits = full_config.get("bits")
        self.codebook_name = full_config.get("codebook", "mul1")

        # Fusions the model class does not declare, supplied by the checkpoint.
        # See _candidate_names. Read defensively: a malformed entry should not
        # stop a checkpoint loading, it should just not resolve.
        self.extra_packed_mapping: dict[str, list[str]] = {}
        raw = full_config.get("packed_modules_mapping")
        # A published checkpoint cannot be edited, and whether the standalone
        # quantization_config.json is even read depends on whether the inlined
        # copy carried tensor_storage -- so the same mapping can also be given
        # as JSON in CUDA_EXL3_PACKED_MAPPING, which always applies and needs no
        # write access to the weights.
        env_raw = _env.getenv("CUDA_EXL3_PACKED_MAPPING")
        if env_raw:
            try:
                parsed = json.loads(env_raw)
                if isinstance(parsed, dict):
                    raw = {**(raw if isinstance(raw, dict) else {}), **parsed}
            except ValueError as e:
                logger.warning("EXL3: CUDA_EXL3_PACKED_MAPPING is not valid JSON "
                               "(%s); ignoring it", e)
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, (list, tuple)) \
                        and all(isinstance(x, str) for x in v):
                    self.extra_packed_mapping[k] = list(v)

        storage = full_config.get("tensor_storage")
        if not storage:
            # vLLM hands us the summary block inlined into config.json, which
            # omits the per-tensor table (bitrates differ per layer, so we need
            # it to size parameters). Pull in the standalone file written next
            # to the weights.
            full_config = self._merge_tensor_storage(full_config)
            self.full_config = full_config
            storage = full_config.get("tensor_storage")
        if not storage:
            raise ValueError(
                "EXL3: could not find `tensor_storage`. cuda-exl3 needs the full "
                "`quantization_config.json` written by exllamav3 (the copy inlined "
                "into config.json is only a summary). Pass it explicitly with "
                "--hf-overrides '{\"quantization_config_file\": \"/path/to/"
                "quantization_config.json\"}' if it is not next to the weights."
            )
        self.modules: dict[str, Exl3ModuleInfo] = {}
        for name, entry in storage.items():
            info = self._parse_module(name, entry)
            if info is not None:
                self.modules[name] = info

        # vLLM's module paths do not always match the checkpoint's. Qwen3.5, for
        # instance, is `model.language_model.layers.N...` on disk but
        # `language_model.model.layers.N...` in vLLM. Index on a form with the
        # `model`/`language_model` wrapper segments dropped so either spelling
        # resolves.
        self.modules_norm: dict[str, Exl3ModuleInfo] = {}
        for name, info in self.modules.items():
            key = _normalize(name)
            if key in self.modules_norm:
                logger.warning(
                    "EXL3: normalized name collision on %r (%s vs %s); "
                    "falling back to exact matching for these.",
                    key, self.modules_norm[key].name, name,
                )
                self.modules_norm[key] = None  # type: ignore[assignment]
            else:
                self.modules_norm[key] = info
        # Some checkpoints omit modules from quantization_config.json even
        # though the tensors are present -- Qwen3.5 ships an EXL3-quantized MTP
        # head that the config never mentions. Recover those from the
        # safetensors headers so speculative decoding can use them.
        recovered = self._augment_from_checkpoint()

        # Shared experts must never be EXL3 (P5). DeepSeek V4/V4.1's vLLM
        # loader runs every `.shared_experts.` tensor through
        # `_pad_shared_expert_weight` whenever the quant config exposes a
        # `weight_block_size` -- which a hybrid config does, see below. That pad
        # zero-extends the *intermediate* axis of a dense [I, H] / [H, I] weight
        # so TP slices it in whole fp8 blocks; applied to a trellis
        # (in/16, out/16, 16*bits) it concatenates zero tiles onto the wrong
        # axis and the layer decodes garbage. Refuse the artifact up front
        # rather than fail inside the loader.
        bad = sorted(n for n in self.modules if _is_shared_expert(n))
        if bad:
            raise ValueError(
                f"EXL3: {len(bad)} quantized tensor(s) live under `shared_experts` "
                f"(e.g. {bad[0]}). Shared experts must stay in the base format: "
                "vLLM's DeepSeek loader pads shared-expert weights on their "
                "intermediate axis for block-quantized configs, which corrupts an "
                "EXL3 trellis. Re-cook with the shared experts excluded."
            )

        # `ignored_layers` / `modules_to_not_convert`: vLLM names for layers the
        # checkpoint left unquantized. Honored for the delegated base method
        # (passed through) and for our own fallback path.
        ignored = full_config.get("ignored_layers") or full_config.get(
            "modules_to_not_convert"
        )
        self.ignored_layers: list[str] = [
            x for x in (ignored or []) if isinstance(x, str)
        ]

        # Hybrid checkpoint (P2): everything that is not an EXL3 tensor is
        # served by the checkpoint's original quantization method, e.g.
        # DeepSeek V4.1's fp8/ue8m0 [32, 32] dense layers via vLLM's MXFP8 path.
        self.base_quant_config: QuantizationConfig | None = self._build_base_quant_config(
            full_config
        )
        if self.base_quant_config is not None:
            # Whatever vLLM has already handed us (nothing yet, in practice --
            # configure_quant_config runs after construction and goes through
            # the property setter below).
            self.base_quant_config.packed_modules_mapping = self._packed_modules_mapping

        logger.info(
            "EXL3: %d quantized tensors (format v%s, codebook=%s)%s%s",
            len(self.modules),
            self.version,
            self.codebook_name,
            f", {recovered} recovered from checkpoint headers" if recovered else "",
            f", other layers delegated to {self.base_quant_config.get_name()}"
            if self.base_quant_config is not None else "",
        )

    # -- hybrid base config (P2) -------------------------------------------

    @staticmethod
    def _base_quant_name(full_config: dict[str, Any]) -> str | None:
        env = _env.getenv(_BASE_QUANT_ENV)
        if env is not None:
            env = env.strip().lower()
            return None if env in _BASE_QUANT_OFF else env
        orig = full_config.get("original_quantization_config")
        if isinstance(orig, dict):
            name = orig.get("quant_method")
            return str(name).lower() if name else None
        return None

    def _build_base_quant_config(self, full_config: dict[str, Any]):
        name = self._base_quant_name(full_config)
        if name is None:
            return None
        orig = full_config.get("original_quantization_config")
        orig = dict(orig) if isinstance(orig, dict) else {}

        if name in ("fp8", "deepseek_v4_fp8"):
            # DeepSeek V4.1: `{"quant_method": "fp8", "activation_scheme":
            # "dynamic", "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
            # "expert_dtype": "fp4"}`. DeepseekV4FP8Config.from_config is
            # Fp8Config.from_config, which requires `activation_scheme` and
            # keys the MXFP8 dense path off weight_block_size == [32, 32].
            if not orig:
                logger.warning(
                    "EXL3: %s=%s but the checkpoint has no "
                    "`original_quantization_config`; assuming DeepSeek V4.1's "
                    "shipped block %s", _BASE_QUANT_ENV, name, _DEEPSEEK_V4_FP8_DEFAULT,
                )
                orig = dict(_DEEPSEEK_V4_FP8_DEFAULT)
            orig.setdefault("quant_method", "fp8")
            orig.setdefault("activation_scheme", "dynamic")
            cls = _base_config_class("deepseek_v4_fp8")
        else:
            if not orig:
                raise ValueError(
                    f"EXL3: {_BASE_QUANT_ENV}={name} needs an "
                    "`original_quantization_config` block to build from."
                )
            cls = _base_config_class(name)

        if self.ignored_layers:
            merged = list(orig.get("ignored_layers") or orig.get("modules_to_not_convert") or [])
            merged += [x for x in self.ignored_layers if x not in merged]
            orig["ignored_layers"] = merged
        base = cls.from_config(orig)
        logger.info("EXL3: hybrid checkpoint, non-EXL3 layers -> %s (%s)",
                    cls.__name__, {k: v for k, v in orig.items() if k != "ignored_layers"})
        return base

    # vLLM assigns `quant_config.packed_modules_mapping = model.packed_modules_mapping`
    # after construction (model_loader/utils.py: configure_quant_config). The
    # delegate needs the same mapping for its own is_layer_skipped checks, so
    # mirror every assignment.
    @property
    def packed_modules_mapping(self) -> dict[str, list[str]]:
        return self.__dict__.get("_packed_modules_mapping", {})

    @packed_modules_mapping.setter
    def packed_modules_mapping(self, value: dict[str, list[str]]) -> None:
        self.__dict__["_packed_modules_mapping"] = value
        base = self.__dict__.get("base_quant_config")
        if base is not None:
            base.packed_modules_mapping = value

    def __getattr__(self, name: str):
        # Attributes the model reads off `vllm_config.quant_config` but which
        # belong to the base format: DeepSeek V4.1 reads `weight_block_size`
        # (MLP intermediate rounding, shared-expert padding, and
        # `_linear_scale_param_name`, which decides that the checkpoint's
        # `.scale` tensors load as ModelOpt's `weight_scale`), `expert_dtype`,
        # `is_scale_e8m0`, `moe_quant_algo`, `ignored_layers_match_mode`...
        # Only reached for attributes this object does not have; a pure-EXL3
        # checkpoint (no base) keeps raising AttributeError so `getattr(cfg,
        # "weight_block_size", None)` stays None exactly as before.
        if name.startswith("_"):
            raise AttributeError(name)
        base = self.__dict__.get("base_quant_config")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)
        if self.base_quant_config is not None:
            self.base_quant_config.apply_vllm_mapper(hf_to_vllm_mapper)

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None) -> None:
        if self.base_quant_config is not None:
            try:
                self.base_quant_config.maybe_update_config(
                    model_name, hf_config=hf_config, revision=revision
                )
            except TypeError:  # older signature without `revision`
                self.base_quant_config.maybe_update_config(model_name, hf_config)

    @staticmethod
    def _parse_module(name: str, entry: dict) -> Exl3ModuleInfo | None:
        if entry.get("quant_format") != "exl3":
            return None
        tensors = entry.get("stored_tensors", {})
        trellis = tensors.get(f"{name}.trellis")
        if trellis is None:
            return None
        k_tiles, n_tiles, packed = trellis["shape"]
        bits = packed // 16

        if f"{name}.mul1" in tensors:
            cb, cb_mult = CB_MUL1, entry.get("mul1_multiplier", 0)
        elif f"{name}.mcg" in tensors:
            cb, cb_mult = CB_MCG, entry.get("mcg_multiplier", 0)
        else:
            cb, cb_mult = CB_3INST, 0

        return Exl3ModuleInfo(
            name=name,
            in_features=k_tiles * 16,
            out_features=n_tiles * 16,
            bits=bits,
            cb=cb,
            cb_mult=cb_mult,
            has_bias=f"{name}.bias" in tensors,
        )

    # -- locating the full per-tensor config -----------------------------

    # Stashed by override_quantization_method(), which vLLM calls with the HF
    # config before it builds the quantization config.
    _model_path_hint: str | None = None

    @classmethod
    def _merge_tensor_storage(cls, config: dict[str, Any]) -> dict[str, Any]:
        # dsv41 P6: the hint is stashed by override_quantization_method() in the
        # process that built the target ModelConfig. A spawned worker that
        # rebuilds a quant config from the inlined summary (the DSpark drafter
        # via get_draft_quant_config) starts with the hint empty, so fall back
        # to an explicit path (env CUDA_EXL3_MODEL_PATH or a
        # `quantization_config_file` key in the summary) and remember it.
        hint = (cls._model_path_hint
                or _env.getenv("CUDA_EXL3_MODEL_PATH")
                or config.get("quantization_config_file"))
        raw = cls._load_config_file(hint)
        if raw is None:
            return config
        if not cls._model_path_hint and hint:
            cls._model_path_hint = hint if os.path.isdir(hint) else os.path.dirname(hint)
        merged = dict(raw)
        merged.update({k: v for k, v in config.items() if k != "tensor_storage"})
        return merged

    @staticmethod
    def _load_config_file(model_path: str | None) -> dict[str, Any] | None:
        if not model_path:
            return None
        fname = "quantization_config.json"
        try:
            if os.path.isfile(model_path):
                with open(model_path) as f:
                    return json.load(f)
            if os.path.isdir(model_path):
                path = os.path.join(model_path, fname)
                if not os.path.exists(path):
                    return None
                with open(path) as f:
                    return json.load(f)
            # Remote repo id: the file sits next to the weights on the Hub.
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=model_path, filename=fname)
            with open(path) as f:
                return json.load(f)
        except Exception as e:  # pragma: no cover - diagnostic path
            logger.warning("EXL3: could not load %s from %s: %s", fname, model_path, e)
            return None

    @classmethod
    def from_config_file(cls, path: str) -> "Exl3Config":
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # We never override another method; this hook just gives us the one
        # chance to see where the model was loaded from.
        if hf_config is not None:
            path = getattr(hf_config, "_name_or_path", None)
            if path:
                cls._model_path_hint = path
        return None


    def _augment_from_checkpoint(self) -> int:
        """Add EXL3 modules present in the weights but missing from the config.

        Reads only safetensors headers (shape/dtype), never tensor data.
        """
        path = self._model_path_hint
        if not path or not os.path.isdir(path):
            return 0
        try:
            index_path = os.path.join(path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]
            elif os.path.exists(os.path.join(path, "model.safetensors")):
                weight_map = None
            else:
                return 0

            from safetensors import safe_open

            if weight_map is None:
                shards = {"model.safetensors": None}
                with safe_open(os.path.join(path, "model.safetensors"),
                               framework="pt") as f:
                    weight_map = {k: "model.safetensors" for k in f.keys()}

            # Group by the module each tensor belongs to.
            by_module: dict[str, set[str]] = {}
            for key in weight_map:
                mod, _, leaf = key.rpartition(".")
                if leaf in ("trellis", "suh", "svh", "mcg", "mul1", "bias"):
                    by_module.setdefault(mod, set()).add(leaf)

            missing = [
                m for m, leaves in by_module.items()
                if "trellis" in leaves and m not in self.modules
            ]
            if not missing:
                return 0

            handles: dict[str, Any] = {}
            added = 0
            for mod in missing:
                key = f"{mod}.trellis"
                fname = weight_map[key]
                if fname not in handles:
                    handles[fname] = safe_open(os.path.join(path, fname), framework="pt")
                shape = handles[fname].get_slice(key).get_shape()
                if len(shape) != 3:
                    continue
                k_tiles, n_tiles, packed = shape
                leaves = by_module[mod]
                if "mul1" in leaves:
                    cb = CB_MUL1
                elif "mcg" in leaves:
                    cb = CB_MCG
                else:
                    cb = CB_3INST
                info = Exl3ModuleInfo(
                    name=mod,
                    in_features=k_tiles * 16,
                    out_features=n_tiles * 16,
                    bits=packed // 16,
                    cb=cb,
                    cb_mult=0,
                    has_bias="bias" in leaves,
                )
                self.modules[mod] = info
                self.modules_norm.setdefault(_normalize(mod), info)
                added += 1
            return added
        except Exception as e:  # pragma: no cover - diagnostic path
            logger.warning("EXL3: could not scan checkpoint headers: %s", e)
            return 0

    # -- QuantizationConfig interface ------------------------------------

    def get_name(self):
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # The trellis codebook decodes natively to fp16 and every EXL3 kernel is
        # fp16; bf16 activations are converted at the layer boundary.
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # mma.m16n8k16 + cp.async
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        # NOT config.json: only the standalone file carries `tensor_storage`.
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(config)

    # -- module lookup ---------------------------------------------------

    def _candidate_names(self, prefix: str) -> list[list[str]]:
        """Checkpoint module names that could back a vLLM module `prefix`.

        Returns a list of candidate *groups*; a group has more than one entry
        when vLLM fuses several checkpoint tensors into one layer (qkv_proj,
        gate_up_proj). Ordered most-likely first.
        """
        out: list[list[str]] = [[prefix]]

        # vLLM fills packed_modules_mapping from the *model class*, so a fusion
        # the model does not declare -- or declares under a different name than
        # the checkpoint used -- cannot be resolved. A checkpoint may therefore
        # carry supplementary entries of its own in its quantization config:
        #
        #   "packed_modules_mapping": {"in_proj_qkvbfg_a": ["qkv_proj", ...]}
        #
        # so a fusion peculiar to one model travels with the weights instead of
        # requiring a patch to the model file. Checkpoint entries are merged
        # under, never over, whatever the model declared.
        base, _, last = prefix.rpartition(".")
        merged = dict(self.extra_packed_mapping)
        merged.update(self.packed_modules_mapping)
        for packed_name, sub_names in merged.items():
            if last == packed_name:
                out.append([f"{base}.{s}" if base else s for s in sub_names])

        # vLLM sometimes drops or adds a leading "model." relative to the
        # checkpoint (e.g. lm_head, or multimodal wrappers).
        extra = []
        for group in out:
            if all(g.startswith("model.") for g in group):
                extra.append([g[len("model.") :] for g in group])
            else:
                extra.append([f"model.{g}" for g in group])
        out.extend(extra)
        return out

    def resolve(self, prefix: str) -> list[Exl3ModuleInfo] | None:
        """Resolve a vLLM module prefix to its EXL3 shards, or None."""
        groups = self._candidate_names(prefix)
        for lookup in (self.modules, self.modules_norm):
            key = (lambda n: n) if lookup is self.modules else _normalize
            for group in groups:
                infos = [lookup.get(key(n)) for n in group]
                if all(i is not None for i in infos):
                    return infos  # type: ignore[return-value]
        return None

    def resolve_moe(self, prefix: str, ckpt_names=None) -> tuple | None:
        """Resolve a routed-experts module to expert 0's gate/up/down tensors.

        All experts of a layer share a bitrate and codebook (checked at load), so
        one expert is enough to size the fused weights.

        `ckpt_names`, if given, is the layer's own (gate, up, down) checkpoint
        naming and is tried first; the known triples (`gate_proj/up_proj/
        down_proj`, DeepSeek's `w1/w3/w2`) follow. A mixed triple is never
        accepted: all three projections must resolve under the same naming.
        """
        prefixes = [prefix]
        if prefix.endswith(_ROUTED_EXPERTS_SUFFIX):
            prefixes.append(prefix[: -len(_ROUTED_EXPERTS_SUFFIX)])

        triples: list[tuple[str, str, str]] = []
        if ckpt_names:
            t = tuple(str(n) for n in ckpt_names)
            if len(t) == 3 and all(t):
                triples.append(t)  # type: ignore[arg-type]
        triples += [t for t in _EXPERT_PROJ_TRIPLES if t not in triples]

        for p in prefixes:
            for triple in triples:
                out = []
                for proj in triple:
                    info = self.resolve(f"{p}.0.{proj}")
                    if info is None or len(info) != 1:
                        break
                    out.append(info[0])
                if len(out) == 3:
                    return tuple(out)
        return None

    @staticmethod
    def _layer_ckpt_names(layer) -> tuple[str, str, str] | None:
        """(gate, up, down) checkpoint names a vLLM RoutedExperts layer declares.

        Advisory only: DeepSeek V4's model builds its experts with the factory
        defaults (gate_proj/down_proj/up_proj) while its own expert mapping uses
        w1/w2/w3, so the attributes can disagree with the checkpoint.
        """
        g = getattr(layer, "ckpt_gate_proj_name", None)
        u = getattr(layer, "ckpt_up_proj_name", None)
        d = getattr(layer, "ckpt_down_proj_name", None)
        if g and u and d:
            return (str(g), str(u), str(d))
        return None

    @staticmethod
    def _layer_swiglu_limit(layer) -> float | None:
        """DeepSeek V4.1's clamped SwiGLU limit (10.0), or None for plain SwiGLU.

        vLLM passes `swiglu_limit` from the HF config through FusedMoEFactory
        into both the RoutedExperts layer and its FusedMoEConfig.
        """
        for src in (layer, getattr(layer, "moe_config", None)):
            v = getattr(src, "swiglu_limit", None)
            if v is not None:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                return v if v > 0 else None
        return None

    def _has_exl3_under(self, prefix: str) -> bool:
        """Any EXL3 tensor nested under this module prefix (e.g. MoE experts)."""
        key = _normalize(prefix) + "."
        return any(n.startswith(key) for n in self.modules_norm)

    def _is_ignored(self, prefix: str) -> bool:
        if not self.ignored_layers:
            return False
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            is_layer_skipped,
        )

        return is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        )

    def _delegate(self, layer, prefix: str, what: str) -> QuantizeMethodBase | None:
        """Hand a non-EXL3 layer to the base format's config (hybrid checkpoints)."""
        base = self.base_quant_config
        if base is None:
            return None
        method = base.get_quant_method(layer, prefix)
        if _DEBUG:
            _delegated.append(prefix)
            logger.warning("EXL3: %s -> %s via %s (%s)  [%d exl3 / %d bf16 / %d delegated]",
                           prefix, type(method).__name__ if method is not None else "None",
                           base.get_name(), what,
                           len(_resolved), len(_unresolved), len(_delegated))
        return method

    @staticmethod
    def _is_moe_layer(layer) -> bool:
        cls_name = type(layer).__name__
        return cls_name in ("RoutedExperts", "FusedMoE", "SharedFusedMoE") or (
            "MoE" in cls_name and "Method" not in cls_name
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from cuda_exl3.linear import Exl3LinearMethod

        # Shared experts (P5): never EXL3, whatever the checkpoint says -- the
        # constructor already refused EXL3 tensors under this name, so this is
        # just the routing: they belong to the base format (or bf16).
        if _is_shared_expert(prefix):
            if self.base_quant_config is not None:
                return self._delegate(layer, prefix, "shared_experts")
            return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None

        infos = self.resolve(prefix)

        # Mixture-of-experts layers. vLLM silently falls back to
        # UnquantizedFusedMoEMethod when get_quant_method returns None, which
        # then fails deep inside weight loading with an unrelated-looking error,
        # so when the experts are EXL3 but cannot be resolved, fail here with
        # the actual reason. Routed experts run as a grouped GEMM (one trellis
        # per expert, selected per token), a separate kernel from the dense path.
        if self._is_moe_layer(layer):
            moe_infos = self.resolve_moe(prefix, self._layer_ckpt_names(layer))
            if moe_infos is not None:
                from cuda_exl3.moe import Exl3MoEMethod

                moe_cfg = getattr(layer, "moe_config", None)
                limit = self._layer_swiglu_limit(layer)
                if _DEBUG:
                    _resolved.append(prefix)
                    logger.warning("EXL3: %s -> EXL3 MoE (w13 %db / w2 %db, swiglu_limit=%s)",
                                   prefix, moe_infos[0].bits, moe_infos[2].bits, limit)
                return Exl3MoEMethod(moe_cfg, self, *moe_infos, prefix=prefix,
                                     swiglu_limit=limit)
            if infos is not None or self._has_exl3_under(prefix):
                raise NotImplementedError(
                    f"EXL3: {prefix} has quantized experts that could not be "
                    "resolved to per-expert gate/up/down tensors (tried "
                    f"{[t for t in _EXPERT_PROJ_TRIPLES]})."
                )
            # Experts in the base format (DeepSeek V4.1: the MTP/DSpark head's
            # MXFP4 experts) or genuinely unquantized.
            return self._delegate(layer, prefix, "non-EXL3 experts")

        if isinstance(layer, (LinearBase, VocabParallelEmbedding)):
            if infos is not None:
                if _DEBUG:
                    _resolved.append(prefix)
                    logger.warning("EXL3: %s -> EXL3 (%d shard%s)  [%d exl3 / %d bf16 / %d delegated]",
                                   prefix, len(infos), "" if len(infos) == 1 else "s",
                                   len(_resolved), len(_unresolved), len(_delegated))
                return Exl3LinearMethod(self, infos, prefix)
            # Hybrid checkpoint: a linear with no EXL3 tensors is in the base
            # format (DeepSeek V4.1: fp8/ue8m0 [32,32] dense, Engram wkv,
            # indexer, compressor, MTP linears -> vLLM's MXFP8 path). The base
            # config applies `ignored_layers` itself.
            if self.base_quant_config is not None:
                return self._delegate(layer, prefix, "non-EXL3 linear")
            if isinstance(layer, LinearBase) and self._is_ignored(prefix):
                return UnquantizedLinearMethod()
            # A linear layer with no EXL3 tensors is genuinely unquantized in
            # this checkpoint (e.g. the bf16 vision tower, or a checkpoint that
            # only quantized its routed experts). Optionally encode it here
            # instead of leaving it in bf16 -- see cuda_exl3.online.
            if isinstance(layer, LinearBase):
                from cuda_exl3.online import (
                    Exl3OnlineLinearMethod,
                    online_bits,
                    shape_supported,
                )

                bits = online_bits()
                if bits is not None and not _is_online_excluded(prefix):
                    k = getattr(layer, "input_size_per_partition", None)
                    n = sum(getattr(layer, "output_partition_sizes", []) or [0])
                    if k and n and shape_supported(k, n):
                        logger.info_once(
                            "EXL3: quantizing unquantized linears online at "
                            "%d bits (CUDA_EXL3_ONLINE_BITS)", bits
                        )
                        return Exl3OnlineLinearMethod(self, prefix, bits)
                    if _DEBUG:
                        logger.warning("EXL3 online: %s shape %sx%s unsupported",
                                    prefix, k, n)
                if _DEBUG:
                    _unresolved.append(prefix)
                    logger.warning("EXL3: %s -> unquantized  [%d exl3 / %d bf16]",
                                   prefix, len(_resolved), len(_unresolved))
                return UnquantizedLinearMethod()
            return None

        # Anything else (vLLM's Attention asks for a KV-cache scale method, for
        # one) is the base format's business on a hybrid checkpoint.
        return self._delegate(layer, prefix, type(layer).__name__)
