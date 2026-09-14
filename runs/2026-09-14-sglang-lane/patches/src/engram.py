# Original file: python/sglang/srt/layers/engram.py from SGLang (lmsysorg/sglang:dev-dsv41,
# commit da64c5cb), Copyright the SGLang Team, licensed under the Apache License 2.0.
# Modifications Copyright (c) 2026 Tech2wild: disk-backed Engram rows, enabled by
# SGLANG_DSV41_ENGRAM_DISK=1; every change is marked "[DSV41 disk]". The modifications are
# also offered under the MIT License; this file as a whole remains under Apache-2.0.
"""Engram: gated n-gram hash memory added to the hc residual stream.

Predecessors come from the live batch and per-request history of the n - 1
previous tokens. The token map, hash multipliers and prime layout must match
sglang.kernels.ops.embeddings.engram_hash to produce identical hash ids.
"""

from __future__ import annotations

import concurrent.futures  # [DSV41 disk]
import ctypes
import glob
import json  # [DSV41 disk]
import logging
import mmap
import os
import re
import struct  # [DSV41 disk]
import time
from typing import Optional

import msgspec
import numpy as np
import torch
from torch import nn

from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.kernels.ops.embeddings.engram_hash import (
    MODE_DECODE,
    MODE_EXTEND,
    MODE_VERIFY,
    engram_commit_history,
    engram_hash_ids,
)
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.torch_quant import FP8_BLOCK_SIZE
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_attention_dp_size,
    get_global_dp_buffer_len,
    is_dp_gatherv_active,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_model, get_parallel, get_serving
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


def _find_next_prime(start: int, seen_primes: set[int]) -> int:
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Token id -> compressed id, plus the compressed vocab size. The size feeds every
    hash multiplier, so a mismatch rehashes the whole table."""
    from tokenizers import Regex, normalizers

    # A private-use sentinel keeps a lone-space token from collapsing to "" under Strip().
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token has nothing to normalize; key it by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, vocab_size: int
) -> torch.Tensor:
    """One odd multiplier per (layer, lookback) from a per-layer RNG, bounded so that
    token_id * multiplier fits in int64."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class EngramLayout(msgspec.Struct, frozen=True):
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head]
    n_heads: int
    head_dim: int

    @classmethod
    def build(
        cls,
        layer_ids: tuple[int, ...],
        num_embeddings: tuple[int, ...],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        vocab_size: int,
    ) -> EngramLayout:
        """Primes are drawn in (layer, n-gram size, head) order from one shared
        ascending sequence starting above vocab_size - 1."""
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], vocab_size - 1
                for _ in range(n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=num_embeddings,
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=head_dim,
        )


def build_engram_layout(config) -> Optional[EngramLayout]:
    layer_ids = tuple(config.engram_layer_ids)
    if not layer_ids:
        return None
    return EngramLayout.build(
        layer_ids=layer_ids,
        num_embeddings=tuple(config.engram_num_embeddings),
        max_ngram_size=config.engram_max_ngram_size,
        n_heads=config.engram_n_heads,
        head_dim=config.engram_head_dim,
        vocab_size=config.engram_vocab_size,
    )


def compute_engram_hash_ids(
    tokens: torch.Tensor,
    blocked: torch.Tensor,
    pad_id: int,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """tokens [T, n]: column 0 is the token itself, column s its s-th predecessor;
    blocked [T, n] marks look-back that ran off the sequence start. Returns
    [T, n_engram_layers, (n - 1) * n_heads] row ids into each layer's table."""
    compressed = torch.where(blocked, pad_id, token_map[tokens])
    products = compressed.unsqueeze(1) * multipliers
    # XOR the multiplied ids one look-back at a time: after step i the running value
    # is the (i + 1)-gram hash, bucketed by that n-gram size's primes.
    rolling, hashes = products[..., 0], []
    for i in range(1, tokens.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


class EngramHasher(nn.Module):
    """Hash ids for every token of a forward batch, [T, n_engram_layers, n_hash_cols]."""

    def __init__(
        self,
        layout: EngramLayout,
        tokenizer,
        pad_id: int,
        compressed_vocab_size: int,
    ):
        super().__init__()
        self.max_ngram_size = layout.max_ngram_size
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == compressed_vocab_size, (
            f"the tokenizer normalizes to {vocab_size} distinct tokens but the config "
            f"expects {compressed_vocab_size}; every hash multiplier depends on it"
        )
        self.pad_id = token_map[pad_id]
        flat = [
            [p for per_ngram in layer for p in per_ngram] for layer in layout.primes
        ]
        offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        self.image_token_id: Optional[int] = None
        self.history: Optional[torch.Tensor] = None
        self.pad_row = 0

    def init_history(self, num_req_slots: int, device) -> None:
        """Allocate oldest-first history with a spare row for graph padding.

        Extend reads scheduler-provided predecessors after prefix hits or
        retraction. Decode commits in forward; verify commits after acceptance.
        """
        self.history = torch.zeros(
            num_req_slots + 1,
            self.max_ngram_size - 1,
            dtype=torch.int32,
            device=device,
        )
        self.pad_row = num_req_slots

    @classmethod
    def from_config(cls, config, layout: EngramLayout) -> EngramHasher:
        # The compressed token map is built with the HF normalizers, so the HF
        # tokenizer backend is used here whatever the serving backend is.
        tokenizer = get_tokenizer(
            get_serving().tokenizer_path,
            tokenizer_mode=get_serving().tokenizer_mode,
            trust_remote_code=get_model().trust_remote_code,
            revision=get_model().revision,
            tokenizer_backend="huggingface",
        )
        result = cls(
            layout,
            tokenizer,
            config.engram_pad_token_id,
            config.engram_compressed_vocab_size,
        )
        result.image_token_id = (
            config.image_token_id
            if config.model_type == "deepseek_v41" and config.vision_n_layers > 0
            else None
        )
        return result

    def forward(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        assert self.history is not None, "EngramHasher.init_history was not called"
        n = self.max_ngram_size
        num_tokens = input_ids.shape[0]
        if num_tokens == 0:
            return torch.empty(
                (0, self.primes.shape[0], self.offsets.shape[1]),
                dtype=torch.int64,
                device=input_ids.device,
            )
        mode = forward_batch.forward_mode
        req_slots = forward_batch.req_pool_indices.to(torch.int64)
        bs = req_slots.shape[0]
        device = input_ids.device
        # Which request each token belongs to and where its run starts; tokens at or
        # past num_real are graph padding. History rows come from self.history via
        # req_slots unless the scheduler supplied this extend's predecessors.
        num_real, block, row, starts = num_tokens, 1, None, None
        history, hist_via_slots = self.history, True
        if mode.is_decode():
            kmode = MODE_DECODE
            commit_rows, commit_last = req_slots, None
        elif mode.is_target_verify():
            block = int(forward_batch.spec_info.draft_token_num)
            assert num_tokens == bs * block, (
                "engram target-verify expects one equal block per request, got "
                f"{num_tokens} tokens for {bs} requests of {block}"
            )
            kmode = MODE_VERIFY
            commit_rows = commit_last = None
        else:
            assert mode.is_extend(), (
                f"engram serves extend, target-verify and decode, not {mode}"
            )
            lens = forward_batch.extend_seq_lens.to(torch.int64)
            starts = forward_batch.extend_start_loc.to(torch.int64)
            row = torch.repeat_interleave(torch.arange(bs, device=device), lens)
            num_real = row.shape[0]
            kmode = MODE_EXTEND
            if forward_batch.ngram_history is not None:
                history, hist_via_slots = forward_batch.ngram_history, False
            commit_rows = torch.where(lens > 0, req_slots, self.pad_row)
            commit_last = (starts + lens - 1).clamp(0, num_tokens - 1)

        if input_ids.is_cuda and torch.version.cuda is not None:
            hash_ids, tokens = engram_hash_ids(
                input_ids,
                forward_batch.positions,
                mode=kmode,
                history=history,
                token_map=self.token_map,
                multipliers=self.multipliers,
                primes=self.primes,
                offsets=self.offsets,
                pad_id=self.pad_id,
                num_real=num_real,
                req_slots=req_slots if hist_via_slots else None,
                block=block,
                row=row,
                starts=starts,
                image_token_id=self.image_token_id,
                mm_pad_shift=MM_PAD_SHIFT_VALUE,
            )
        else:
            hash_ids, tokens = self._torch_hash_ids(
                input_ids,
                forward_batch.positions,
                kmode,
                history[req_slots] if hist_via_slots else history,
                num_real,
                block,
                row,
                starts,
            )
        if commit_rows is not None:
            # Padded rows must not overwrite a live request's history.
            last_tokens = tokens if commit_last is None else tokens[commit_last]
            out_loc = forward_batch.out_cache_loc
            if out_loc is not None:
                if commit_last is not None:
                    out_loc = out_loc[commit_last]
                commit_rows = torch.where(out_loc == 0, self.pad_row, commit_rows)
            self.history[commit_rows] = (
                last_tokens[:, : n - 1].flip(-1).to(self.history.dtype)
            )
        return hash_ids

    def _torch_hash_ids(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kmode: int,
        history: torch.Tensor,
        num_real: int,
        block: int,
        row: Optional[torch.Tensor],
        starts: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-CUDA fallback of the hash kernel: predecessor table [T, n] and hash ids.
        ``history`` is already the per-request [bs, n - 1] rows of this batch."""
        n = self.max_ngram_size
        num_tokens = input_ids.shape[0]
        device = input_ids.device
        input_ids = input_ids.to(torch.int64)
        shifts = torch.arange(n, device=device)
        if kmode == MODE_DECODE:
            tokens = torch.cat(
                [input_ids.unsqueeze(-1), history.to(torch.int64).flip(-1)], dim=-1
            )
        else:
            t = torch.arange(num_real, device=device)
            if kmode == MODE_VERIFY:
                row = t // block
                offset = t - row * block
            else:
                offset = t - starts[row]
            # Predecessor at shift s of token t: an earlier token of the same run
            # when s <= offset, else history[row, -(s - offset)].
            from_batch = shifts.unsqueeze(0) <= offset.unsqueeze(-1)
            in_batch = input_ids[(t.unsqueeze(-1) - shifts).clamp_min(0)]
            hist_col = (n - 2 - (shifts.unsqueeze(0) - offset.unsqueeze(-1) - 1)).clamp(
                0, n - 2
            )
            from_hist = history.to(torch.int64)[row].gather(1, hist_col)
            tokens = torch.where(from_batch, in_batch, from_hist)
        if num_real < num_tokens:
            tokens = torch.cat([tokens, tokens.new_zeros(num_tokens - num_real, n)])
        positions = positions.to(torch.int64)
        blocked = positions.unsqueeze(-1) < shifts
        if num_real < num_tokens:
            blocked[num_real:] = True
        if self.image_token_id is not None:
            # Scheduler-provided history still carries the multimodal pad ids.
            tokens = tokens.masked_fill(
                tokens >= MM_PAD_SHIFT_VALUE, self.image_token_id
            )
            # Once a lookback hits an image, every older predecessor is PAD.
            blocked = (
                (blocked | (tokens == self.image_token_id))
                .to(torch.int32)
                .cummax(-1)
                .values.bool()
            )
        hash_ids = compute_engram_hash_ids(
            tokens,
            blocked,
            self.pad_id,
            self.token_map,
            self.multipliers,
            self.primes,
            self.offsets,
        )
        return hash_ids, tokens

    def commit_after_verify(
        self,
        verify_ids_2d: torch.Tensor,
        req_pool_indices: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit anchor + accepted drafts; the bonus is the next block's anchor."""
        assert self.history is not None, "EngramHasher.init_history was not called"
        if self.history.is_cuda:
            engram_commit_history(
                self.history, verify_ids_2d, req_pool_indices, commit_lens
            )
            return
        n1 = self.max_ngram_size - 1
        req = req_pool_indices.to(torch.int64)
        window = torch.cat(
            [self.history[req], verify_ids_2d.to(self.history.dtype)], dim=1
        )
        cols = commit_lens.to(torch.int64).unsqueeze(-1) + torch.arange(
            n1, device=window.device
        )
        self.history[req] = window.gather(1, cols)


_THP_DIR = "/sys/kernel/mm/transparent_hugepage"


def _thp_mode(knob: str) -> str:
    """Active mode of a transparent_hugepage sysfs knob ("" if unreadable)."""
    try:
        with open(f"{_THP_DIR}/{knob}") as f:
            m = re.search(r"\[(\w+)\]", f.read())
        return m.group(1) if m else ""
    except OSError:
        return ""


def _huge_pages_backing(addr: int) -> tuple[int, int]:
    """(mapped_kB, huge_kB) of the VMA holding addr, from /proc/self/smaps.
    The only evidence that the kernel really handed out huge pages."""
    mapped = huge = 0
    inside = False
    try:
        with open("/proc/self/smaps") as f:
            for line in f:
                m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
                if m:
                    if inside:
                        break
                    inside = int(m.group(1), 16) <= addr < int(m.group(2), 16)
                elif inside:
                    key, _, rest = line.partition(":")
                    if key == "Rss":
                        mapped = int(rest.split()[0])
                    elif key in ("AnonHugePages", "ShmemPmdMapped", "FilePmdMapped"):
                        huge += int(rest.split()[0])
    except OSError:
        pass
    return mapped, huge


_page_cache_dropped = False


def drop_checkpoint_page_cache(model_path: Optional[str] = None) -> tuple[int, int]:
    """Drop checkpoint page cache with posix_fadvise(DONTNEED); return (files, bytes).

    Cached checkpoint pages can prevent 512 MiB huge-page allocation,
    so drop them before pre-faulting private host tables.
    """
    if model_path is None:
        try:
            model_path = get_model().model_path
        except (ValueError, AttributeError):
            # No published runtime context (unit tests, offline tools): nothing to drop.
            return 0, 0
    files, nbytes = 0, 0
    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        try:
            fd = os.open(f, os.O_RDONLY)
        except OSError:
            continue
        try:
            nbytes += os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            files += 1
        finally:
            os.close(fd)
    return files, nbytes


def _drop_page_cache_once(reason: str) -> None:
    global _page_cache_dropped
    if _page_cache_dropped or not envs.SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE.get():
        return
    _page_cache_dropped = True
    files, nbytes = drop_checkpoint_page_cache()
    logger.info(
        "engram host table: dropped the page cache of %d checkpoint files (%.0f GiB) %s",
        files,
        nbytes / 2**30,
        reason,
    )


class _HostTable:
    """Host-memory backing for one engram table.

    Layouts:
      shared   one memfd holding every row, mapped by all ranks of the group;
               rank 0 creates it, the others open it through /proc/<pid>/fd.
               No all-reduce. Huge pages need transparent_hugepage/shmem_enabled.
      private  one anonymous mapping per rank holding only its own rows, so the
               lookup keeps the sharded all-reduce. Huge pages come from
               transparent_hugepage/enabled (madvise or always).
    """

    def __init__(self, layout: str, nbytes: int, name: str, group, pin: bool):
        assert layout in ("shared", "private"), layout
        self.layout = layout
        self.nbytes = nbytes
        self.group = group
        self.dirty = False
        self.registered = False
        if layout == "shared":
            self.fd = self._open_shared_fd(nbytes, name)
            self.mm = mmap.mmap(
                self.fd,
                nbytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        else:
            self.fd = None
            self.mm = mmap.mmap(
                -1,
                nbytes,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        # Advisory before the first touch: pages are allocated huge at fault time.
        self.mm.madvise(mmap.MADV_HUGEPAGE)
        self.bytes = torch.frombuffer(self.mm, dtype=torch.uint8)
        if layout == "private":
            # Fault the shard in now, on a host whose page cache has just been
            # emptied: cached checkpoint pages left by a previous server, or by
            # the loader itself, make the 512 MiB huge-page faults fall back.
            _drop_page_cache_once("before pre-faulting the private shard")
            np.frombuffer(self.mm, dtype=np.uint8)[:: mmap.PAGESIZE] = 0
        if layout == "shared":
            # Every rank holds the fd before rank 0 continues; the /proc path only
            # resolves while rank 0 keeps its descriptor.
            group.barrier()
        if pin:
            err = torch.cuda.cudart().cudaHostRegister(self.bytes.data_ptr(), nbytes, 0)
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister({nbytes} bytes) failed: {err}")
            self.registered = True

    @staticmethod
    def choose_layout(requested: str) -> str:
        if requested != "auto":
            return requested
        if _thp_mode("shmem_enabled") in ("advise", "always", "within_size", "force"):
            return "shared"
        if _thp_mode("enabled") in ("madvise", "always"):
            return "private"
        return "shared"

    def _open_shared_fd(self, nbytes: int, name: str) -> int:
        owner = None
        if self.group.rank_in_group == 0:
            fd = os.memfd_create(name, 0)
            os.ftruncate(fd, nbytes)
            owner = (os.getpid(), fd)
        pid, owner_fd = self.group.broadcast_object(owner, src=0)
        if self.group.rank_in_group == 0:
            return fd
        try:
            return os.open(f"/proc/{pid}/fd/{owner_fd}", os.O_RDWR)
        except OSError as e:
            raise RuntimeError(
                "engram host table: cannot open rank 0's memfd through /proc; the "
                "TP ranks must share a PID namespace"
            ) from e

    def _collapse(self, tries: int = 3) -> None:
        """madvise(MADV_COLLAPSE): synchronously fold whatever is still on base pages
        into huge pages. Anonymous memory only; shmem obeys shmem_enabled and
        refuses. EAGAIN means compaction ran out of time, so it is retried a few
        times; anything else is logged and left."""
        MADV_COLLAPSE = 25  # Linux >= 6.1; not in Python's mmap module
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        for attempt in range(tries):
            rc = libc.madvise(
                ctypes.c_void_p(self.bytes.data_ptr()),
                ctypes.c_size_t(self.nbytes),
                MADV_COLLAPSE,
            )
            if rc == 0:
                return
            err = ctypes.get_errno()
            if (
                err != 11 or attempt == tries - 1
            ):  # EAGAIN is the only one worth retrying
                logger.info("engram host table: MADV_COLLAPSE errno %d", err)
                return
            time.sleep(1.0)

    def finish_load(self, label: str):
        if not self.dirty:
            return
        self.dirty = False
        if self.layout == "shared":
            self.group.barrier()
        mapped_kb, huge_kb = _huge_pages_backing(self.bytes.data_ptr())
        if self.layout == "private" and huge_kb < mapped_kb * 0.98:
            # The loader's own reads refilled the page cache; empty it again so the
            # collapse can find contiguous memory.
            if envs.SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE.get():
                drop_checkpoint_page_cache()
            self._collapse()
            mapped_kb, huge_kb = _huge_pages_backing(self.bytes.data_ptr())
        pct = 100.0 * huge_kb / mapped_kb if mapped_kb else 0.0
        msg = (
            f"engram host table {label}: layout={self.layout}, "
            f"{mapped_kb / 2**10:.0f} MiB resident, {huge_kb / 2**10:.0f} MiB in huge pages "
            f"({pct:.0f}%){', pinned' if self.registered else ', unpinned (ATS)'}"
        )
        if huge_kb == 0:
            knob = "shmem_enabled" if self.layout == "shared" else "enabled"
            logger.warning(
                "%s. No huge pages: expect ~10x slower lookups (one TLB miss per row); "
                "transparent_hugepage/%s is '%s'",
                msg,
                knob,
                _thp_mode(knob) or "unreadable",
            )
        else:
            logger.info(msg)


# ---------------------------------------------------------------------------
# [DSV41 disk] Disk-backed Engram rows (SGLANG_DSV41_ENGRAM_DISK=1).
#
# On GB10 (DGX Spark) host and device memory are one 121.7 GiB pool, so neither
# the device-sharded table (about 47 GiB per rank at TP4) nor the host table fits
# beside about 77 GiB of weights. In disk mode a rank keeps no rows in memory: it
# reads the rows it owns from the checkpoint's safetensors shards (full shards,
# or a node-local sparse copy holding only its rows at the original offsets) and
# returns what the device-sharded path returns: the dequantized row for an owned
# id, zeros for any other id, then the same TP all-reduce.
#
# Ownership is by whole hash heads (the prime-sized sub-tables in hash-column
# order), ceil(H / tp) heads per rank. Any partition is exact under the
# all-reduce; this one matches our vLLM lane, whose sparse copies of shards
# 47/48 hold exactly these rows. The default floor(N * r / tp) split differs by
# about 1,000 rows at each boundary and would read holes of those copies.
# ---------------------------------------------------------------------------
_ENGRAM_DISK = os.environ.get("SGLANG_DSV41_ENGRAM_DISK", "0") == "1"


def engram_disk_enabled() -> bool:
    return _ENGRAM_DISK


def engram_head_sizes(layout: "EngramLayout", layer_hash_index: int) -> tuple[int, ...]:
    """Rows of every hash column of one table, in the order EngramHasher's offsets
    concatenate them."""
    return tuple(p for per_ngram in layout.primes[layer_hash_index] for p in per_ngram)


def head_aligned_row_range(head_sizes, tp_rank: int, tp_size: int) -> tuple[int, int]:
    """Global rows [lo, hi) of the whole hash heads that TP rank `tp_rank` owns."""
    per = -(-len(head_sizes) // tp_size)
    first = min(tp_rank * per, len(head_sizes))
    last = min(first + per, len(head_sizes))
    return sum(head_sizes[:first]), sum(head_sizes[:last])


def _safetensors_entry(path: str, name: str) -> tuple[int, str, list]:
    """(absolute byte offset, dtype, shape) of one tensor of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(n))[name]
    return 8 + n + meta["data_offsets"][0], meta["dtype"], list(meta["shape"])


def _engram_disk_root(layer_id: int, lo: int, hi: int) -> str:
    """SGLANG_DSV41_ENGRAM_DIR, else the model path. A directory with an
    engram-local.json (a node-local sparse copy) must cover rows [lo, hi)."""
    root = os.environ.get("SGLANG_DSV41_ENGRAM_DIR") or get_model().model_path
    meta_path = os.path.join(root, "engram-local.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            have = json.load(f)["layers"].get(str(layer_id))
        if have is None or not (have[0] <= lo and hi <= have[1]):
            raise RuntimeError(
                f"engram disk: {meta_path} holds layer {layer_id} rows {have}, this rank "
                f"needs [{lo}, {hi}). Node rank and TP rank must match the copy."
            )
    elif os.environ.get("SGLANG_DSV41_ENGRAM_REQUIRE_LOCAL", "0") == "1":
        raise RuntimeError(
            f"engram disk: SGLANG_DSV41_ENGRAM_REQUIRE_LOCAL=1 but {root} has no engram-local.json"
        )
    return root


_DISK_POOL: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _disk_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _DISK_POOL
    if _DISK_POOL is None:
        _DISK_POOL = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_DSV41_ENGRAM_THREADS", "32")),
            thread_name_prefix="engram-disk",
        )
    return _DISK_POOL


def _parallel_take(jobs) -> None:
    """For each (src, rows, out): out[i] = src[rows[i]]. Large gathers are split
    over the shared pool (np.take releases the GIL), so cold rows fault in from
    disk in parallel. Only the calling thread submits."""
    pool = _disk_pool()
    futures = []
    for src, rows, out in jobs:
        n = rows.shape[0]
        step = min(1024, max(16, -(-n // pool._max_workers)))
        if n <= step:
            np.take(src, rows, axis=0, out=out)
            continue
        for a in range(0, n, step):
            futures.append(pool.submit(np.take, src, rows[a : a + step], 0, out[a : a + step]))
    for f in futures:
        f.result()


class _RowioJob(ctypes.Structure):
    # Must match `struct Job` in dsv41_rowio.cpp.
    _fields_ = [
        ("table", ctypes.c_void_p),
        ("ids", ctypes.c_void_p),
        ("w_out", ctypes.c_void_p),
        ("s_out", ctypes.c_void_p),
        ("n", ctypes.c_uint64),
    ]


_ROWIO = None


def _rowio():
    """(lib, host-function pointer, cuLaunchHostFunc) of the native reader named by
    SGLANG_DSV41_ENGRAM_ROWIO, or None when unset."""
    global _ROWIO
    if _ROWIO is None:
        path = os.environ.get("SGLANG_DSV41_ENGRAM_ROWIO", "")
        if not path:
            _ROWIO = False
        else:
            lib = ctypes.CDLL(path)
            lib.rowio_open.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_char_p]
            lib.rowio_open.argtypes += [ctypes.c_uint64] * 5 + [ctypes.c_int]
            lib.rowio_open.restype = ctypes.c_void_p
            cuda = ctypes.CDLL("libcuda.so.1")
            launch = cuda.cuLaunchHostFunc
            launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            launch.restype = ctypes.c_int
            _ROWIO = (lib, ctypes.cast(lib.rowio_lookup, ctypes.c_void_p).value, launch)
    return _ROWIO or None


class DiskEngramRows:
    """Rows [lo, hi) of one engram table, read on demand from safetensors shards.

    Eager lookups (prefill, eager decode): de-duplicate the owned ids, gather their
    raw fp8 rows and e8m0 scale rows from read-only memmaps over a thread pool,
    copy the raw bytes to the GPU and dequantize there with the upstream
    engram_gather kernel. A lookup inside CUDA-graph capture cannot sync with the
    host, so it enqueues the native reader (SGLANG_DSV41_ENGRAM_ROWIO, built from
    dsv41_rowio.cpp) as a CUDA host function that the graph runs at replay. Both
    paths hand engram_gather the same bytes, so the values equal the
    device-sharded path's bit for bit (unowned ids: fp8 0 x e8m0 0 = +0.0).
    """

    def __init__(self, root: str, layer_id: int, num_embeddings: int, dim: int, lo: int, hi: int):
        self.layer_id, self.dim, self.sb = layer_id, dim, dim // FP8_BLOCK_SIZE
        self.lo, self.hi = lo, hi
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        w_name = f"layers.{layer_id}.engram.embed.weight"
        s_name = f"layers.{layer_id}.engram.embed.scale"
        self.w_path = os.path.join(root, weight_map[w_name])
        self.s_path = os.path.join(root, weight_map[s_name])
        self.w_off, w_dtype, w_shape = _safetensors_entry(self.w_path, w_name)
        self.s_off, s_dtype, s_shape = _safetensors_entry(self.s_path, s_name)
        got = (w_dtype, w_shape, s_dtype, s_shape)
        want = ("F8_E4M3", [num_embeddings, dim], "F8_E8M0", [num_embeddings, self.sb])
        if got != want:
            raise ValueError(f"engram disk: layer {layer_id} tensors are {got}, expected {want}")
        rows = max(hi - lo, 1)
        # Read-only maps of the owned rows only; pages fault in on first use.
        self.w_map = np.memmap(self.w_path, np.uint8, "r", self.w_off + lo * dim, (rows, dim))
        self.s_map = np.memmap(self.s_path, np.uint8, "r", self.s_off + lo * self.sb, (rows, self.sb))
        for m in (self.w_map, self.s_map):
            try:
                m._mmap.madvise(mmap.MADV_RANDOM)
            except (AttributeError, OSError, ValueError):
                pass
        self._check_rows()
        self._graph = None  # native reader state, created on the first eager lookup
        self._jobs: dict = {}

    def _check_rows(self, samples: int = 16) -> None:
        """A sparse copy that does not hold these rows reads zeros. An e8m0 byte 0
        is 2**-127, so a sampled scale row of all zeros means a hole."""
        if self.hi <= self.lo:
            return
        n = self.hi - self.lo
        probe = np.unique(np.append(np.arange(0, n, max(1, n // samples))[:samples], n - 1))
        holes = int((self.s_map[probe] == 0).all(axis=1).sum())
        if holes:
            raise RuntimeError(
                f"engram disk: layer {self.layer_id}: {holes} of {probe.size} sampled rows of "
                f"[{self.lo}, {self.hi}) in {self.s_path} read as zeros (sparse-file hole): "
                "this directory does not hold this rank's rows"
            )

    def gather_raw(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Host gather: uint8 rows [n, dim] and scale rows [n, sb] for global ids,
        zeros for ids outside [lo, hi). Each distinct row is read once."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        owned = (ids >= self.lo) & (ids < self.hi)
        uniq, inverse = np.unique(ids[owned] - self.lo, return_inverse=True)
        w_u = np.empty((uniq.size, self.dim), np.uint8)
        s_u = np.empty((uniq.size, self.sb), np.uint8)
        _parallel_take([(self.w_map, uniq, w_u), (self.s_map, uniq, s_u)])
        w = np.zeros((ids.size, self.dim), np.uint8)
        s = np.zeros((ids.size, self.sb), np.uint8)
        w[owned] = w_u[inverse.reshape(-1)]
        s[owned] = s_u[inverse.reshape(-1)]
        return w, s

    def lookup(self, indices: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """out [..., dim] bf16 on the GPU <- dequantized owned rows, zeros elsewhere."""
        flat = indices.reshape(-1)
        if flat.numel() == 0:
            return out
        if torch.cuda.is_current_stream_capturing():
            return self._lookup_in_graph(flat, out)
        if self._graph is None and _rowio() is not None:
            self._init_graph_staging(out.device)  # pinned allocations stay outside capture
        owned = (flat >= self.lo) & (flat < self.hi)
        local = torch.where(owned, flat - self.lo, torch.full_like(flat, -1))
        # Unowned ids become -1, which sorts first: row 0 of the staged block is
        # then an all-zero row that engram_gather turns into +0.0.
        uniq, inverse = torch.unique(local, return_inverse=True)
        uniq_host = uniq.cpu().numpy()
        skip = int(uniq_host.size > 0 and uniq_host[0] < 0)
        rows = uniq_host[skip:]
        w = torch.zeros((uniq_host.size, self.dim), dtype=torch.uint8, pin_memory=True)
        s = torch.zeros((uniq_host.size, self.sb), dtype=torch.uint8, pin_memory=True)
        if rows.size:
            _parallel_take([(self.w_map, rows, w.numpy()[skip:]), (self.s_map, rows, s.numpy()[skip:])])
        w_dev = w.to(out.device, non_blocking=True)
        s_dev = s.to(out.device, non_blocking=True)
        engram_gather(
            w_dev.data_ptr(), s_dev.data_ptr(), inverse.reshape(-1),
            out.view(-1, self.dim), self.dim, FP8_BLOCK_SIZE,
        )
        return out

    def _init_graph_staging(self, device) -> None:
        lib, _, _ = _rowio()
        cap = int(os.environ.get("SGLANG_DSV41_ENGRAM_GRAPH_MAX_ROWS", "16384"))
        direct = int(os.environ.get("SGLANG_DSV41_ENGRAM_DIRECT", "0") == "1")
        table = lib.rowio_open(
            self.w_path.encode(), self.w_off, self.s_path.encode(), self.s_off,
            self.lo, self.hi, self.dim, self.sb, direct,
        )
        if not table:
            raise RuntimeError(f"engram disk: dsv41_rowio cannot open {self.w_path}")
        self._graph = (
            table,
            torch.empty(cap, dtype=torch.int64, pin_memory=True),
            torch.empty((cap, self.dim), dtype=torch.uint8, pin_memory=True),
            torch.empty((cap, self.sb), dtype=torch.uint8, pin_memory=True),
            torch.arange(cap, dtype=torch.int64, device=device),
        )

    def _lookup_in_graph(self, flat: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        native = _rowio()
        if native is None or self._graph is None:
            raise RuntimeError(
                "engram disk: CUDA-graph capture reached the Engram lookup; set "
                "SGLANG_DSV41_ENGRAM_ROWIO to the built dsv41_rowio.so, or launch with "
                "--disable-cuda-graph"
            )
        _, host_fn, launch = native
        table, ids_host, w_host, s_host, seq = self._graph
        n = flat.numel()
        if n > ids_host.shape[0]:
            raise RuntimeError(
                f"engram disk: {n} ids in a captured batch exceed "
                f"SGLANG_DSV41_ENGRAM_GRAPH_MAX_ROWS={ids_host.shape[0]}"
            )
        job = self._jobs.get(n)
        if job is None:  # kept alive: the captured host node points at it
            job = _RowioJob(table, ids_host.data_ptr(), w_host.data_ptr(), s_host.data_ptr(), n)
            self._jobs[n] = job
        ids_host[:n].copy_(flat, non_blocking=True)
        rc = launch(torch.cuda.current_stream().cuda_stream, host_fn, ctypes.addressof(job))
        if rc != 0:
            raise RuntimeError(f"engram disk: cuLaunchHostFunc returned {rc}")
        # The GPU reads the pinned rows in place (UVA); ids outside [lo, hi) were zero-filled.
        engram_gather(
            w_host.data_ptr(), s_host.data_ptr(), seq[:n],
            out.view(-1, self.dim), self.dim, FP8_BLOCK_SIZE,
        )
        return out


class EngramEmbedding(nn.Module):
    """One layer's fp8 hash table with e8m0 block scales, dequantized on lookup.

    Default: rows sharded over the TP group in device memory; each rank gathers
    the rows it owns, zeroes the rest and the all-reduce reassembles the lookup.
    With SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE the table lives in host memory and
    the GPU gathers rows over the CPU link -- either one shared copy with no
    all-reduce, or one private shard per rank (see _HostTable). Loading is
    sharded in every layout: a rank writes only its own row range.
    """

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        layer_id: int,
        head_sizes: Optional[tuple[int, ...]] = None,  # [DSV41 disk]
    ):
        super().__init__()
        self.dim = dim
        self.tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        self.row_start = num_embeddings * tp_rank // self.tp_size
        row_end = num_embeddings * (tp_rank + 1) // self.tp_size
        self.rows = row_end - self.row_start
        self.host_table: Optional[_HostTable] = None
        self.disk: Optional[DiskEngramRows] = None  # [DSV41 disk]
        if engram_disk_enabled():  # [DSV41 disk]
            self._init_disk(num_embeddings, dim, layer_id, head_sizes, tp_rank)
            return
        if envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
            self._init_host_table(num_embeddings, dim, layer_id)
        else:
            self.weight = nn.Parameter(
                torch.empty(self.rows, dim, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(
                    self.rows, dim // FP8_BLOCK_SIZE, dtype=torch.float8_e8m0fnu
                ),
                requires_grad=False,
            )
        self.weight.weight_loader = self._load_rows
        self.scale.weight_loader = self._load_rows

    def _init_disk(self, num_embeddings, dim, layer_id, head_sizes, tp_rank):
        """[DSV41 disk] Own whole hash heads and read their rows from disk."""
        assert head_sizes is not None and sum(head_sizes) == num_embeddings, (
            "engram disk mode needs the table's hash-column sizes"
        )
        lo, hi = head_aligned_row_range(head_sizes, tp_rank, self.tp_size)
        self.row_start, self.rows = lo, hi - lo
        root = _engram_disk_root(layer_id, lo, hi)
        self.disk = DiskEngramRows(root, layer_id, num_embeddings, dim, lo, hi)
        # Empty placeholders registered as buffers, not parameters: no memory is
        # reserved and the checkpoint rows have nothing to load into
        # (DeepseekV4ForCausalLM.load_weights skips them in disk mode).
        self.register_buffer("weight", torch.empty(0, dtype=torch.float8_e4m3fn), persistent=False)
        self.register_buffer("scale", torch.empty(0, dtype=torch.float8_e8m0fnu), persistent=False)
        logger.info(
            "engram disk: layer %d rank %d rows [%d, %d) from %s (%s)",
            layer_id, tp_rank, lo, hi, self.disk.w_path,
            "graph reader " + os.environ["SGLANG_DSV41_ENGRAM_ROWIO"]
            if os.environ.get("SGLANG_DSV41_ENGRAM_ROWIO") else "eager only",
        )

    def _init_host_table(self, num_embeddings: int, dim: int, layer_id: int):
        layout = _HostTable.choose_layout(
            envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get()
        )
        n = num_embeddings if layout == "shared" else self.rows
        w_bytes = n * dim
        s_bytes = n * (dim // FP8_BLOCK_SIZE)
        self.host_table = _HostTable(
            layout,
            max(1, w_bytes + s_bytes),  # mmap requires storage even for an empty shard.
            f"sglang_engram_{layer_id}",
            get_tp_group(),
            pin=envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN.get(),
        )
        raw = self.host_table.bytes[: w_bytes + s_bytes]
        weight = raw[:w_bytes].view(torch.float8_e4m3fn).view(n, dim)
        scale = raw[w_bytes:].view(torch.float8_e8m0fnu).view(n, dim // FP8_BLOCK_SIZE)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)

    @property
    def _shared(self) -> bool:
        return self.host_table is not None and self.host_table.layout == "shared"

    def _load_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        rows = slice(self.row_start, self.row_start + self.rows)
        if self._shared:
            param.data[rows].copy_(loaded_weight[rows])
        else:
            param.data.copy_(loaded_weight[rows])
        if self.host_table is not None:
            self.host_table.dirty = True

    def finish_load(self, label: str = ""):
        """Barrier (shared layout) once every rank has written its rows; log how
        the table ended up backed."""
        if self.host_table is not None:
            self.host_table.finish_load(label)

    def forward(
        self,
        indices: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        *,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        if self._shared:
            if indices.shape[0] == 0:
                return self._empty(indices)
            out = self._empty(indices)
            engram_gather(
                self.weight.data_ptr(),
                self.scale.data_ptr(),
                indices.reshape(-1),
                out.view(-1, self.dim),
                self.dim,
                FP8_BLOCK_SIZE,
            )
            return out
        if cp_all_tokens and self.tp_size > 1:
            # Prefill CP: gather the hash ids over the CP group first so every
            # TP rank looks up the same indices, then keep this rank's slice.
            parallel = get_parallel()
            local_rows = indices.shape[0]
            all_indices = indices.new_empty(
                (parallel.attn_cp_size * local_rows, *indices.shape[1:])
            )
            attn_cp_all_gather_into_tensor(all_indices, indices.contiguous())
            start = parallel.attn_cp_rank * local_rows
            return self._lookup(all_indices)[start : start + local_rows]
        if self.tp_size > 1 and get_attention_dp_size() > 1:
            return self._dp_sharded_lookup(indices, forward_batch)
        return self._lookup(indices)

    def _lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup when every TP rank holds the same indices: the device and
        private host shards zero unowned rows and the all-reduce reassembles."""
        if indices.shape[0] == 0:
            return self._empty(indices)
        values = self._owned_rows(indices)
        if self.tp_size > 1:
            values = tensor_model_parallel_all_reduce(values)
        return values

    def _empty(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            *indices.shape, self.dim, dtype=torch.bfloat16, device=indices.device
        )

    def _owned_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """Rows of `indices` this rank's shard holds, zero for the rest."""
        if self.disk is not None:  # [DSV41 disk]
            return self.disk.lookup(indices, self._empty(indices))
        if self.rows == 0:
            return self._empty(indices).zero_()
        if self.host_table is None and (
            not indices.is_cuda or torch.version.cuda is None
        ):
            local = indices - self.row_start
            owned = (local >= 0) & (local < self.rows)
            local = local.masked_fill(~owned, 0)
            rows = self.weight[local].float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
            values = (rows * self.scale[local].float().unsqueeze(-1)).flatten(-2)
            return values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
        out = self._empty(indices)
        engram_gather(
            self.weight.data_ptr(),
            self.scale.data_ptr(),
            indices.reshape(-1),
            out.view(-1, self.dim),
            self.dim,
            FP8_BLOCK_SIZE,
            row_lo=self.row_start,
            row_hi=self.row_start + self.rows,
        )
        return out

    def _dp_sharded_lookup(
        self, indices: torch.Tensor, forward_batch: Optional[ForwardBatch]
    ) -> torch.Tensor:
        """Gather DP ranks' indices before looking up TP-sharded rows.

        Every rank joins, including idle ranks with zero rows; the reduced result
        is sliced back to each rank's local tokens.
        """
        assert forward_batch is not None, "the DP engram lookup needs the batch"
        rows = get_global_dp_buffer_len()
        ids_global = torch.empty(
            (rows, *indices.shape[1:]), dtype=indices.dtype, device=indices.device
        )
        # The MAX_LEN gather may zero its local input in place, hence the clone.
        dp_gather_replicate(ids_global, indices.clone(), forward_batch)
        if rows == 0:
            return self._empty(indices)
        values = self._owned_rows(ids_global).view(rows, -1)
        local = torch.empty(
            (indices.shape[0], values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        # MAX_LEN or gatherv-sized slices use reduce-scatter;
        # otherwise scatter the summed buffer to the DP ranks.
        padding = forward_batch.dp_padding_mode
        if (
            padding is not None
            and padding.is_max_len()
            and self.tp_size == get_attention_dp_size()
            and rows == self.tp_size * local.shape[0]
        ) or is_dp_gatherv_active():
            dp_reduce_scatter_tensor(local, values)
        else:
            dp_scatter(local, tensor_model_parallel_all_reduce(values), forward_batch)
        return local.view(*indices.shape, self.dim)


def engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
) -> torch.Tensor:
    """x [T, hc_mult, dim]; kv [T, (hc_mult + 1) * dim] holds one key per hc copy
    followed by the shared value. Adds the gated value to every copy.

    The fused kernel serves every token count on CUDA; the torch path below is the
    non-CUDA fallback and materializes fp32 copies of x, key and value."""
    if (
        x.is_cuda
        and torch.version.cuda is not None
        and x.ndim == 3
        and kv.shape == (x.shape[0], (x.shape[1] + 1) * x.shape[2])
        and x.dtype == kv.dtype
        and x.dtype in (torch.bfloat16, torch.float32)
        and q_weight.dtype in (torch.bfloat16, torch.float32)
        and k_weight.dtype in (torch.bfloat16, torch.float32)
        and all(t.is_contiguous() for t in (x, kv, q_weight, k_weight))
    ):
        return fused_engram_gate(x, kv, q_weight, k_weight, eps, clamp_value)
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    # Normalized per (token, hc copy) over dim, not jointly over the copies.
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    # Signed square root before the sigmoid, matching the training kernel.
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class Engram(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        layout: EngramLayout,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        dim, hc_mult = config.hidden_size, config.hc_mult
        self.embed = EngramEmbedding(
            layout.num_embeddings[self.layer_hash_index],
            layout.head_dim,
            layer_id,
            head_sizes=engram_head_sizes(layout, self.layer_hash_index),  # [DSV41 disk]
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            dim * (hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wkv", prefix),
        )
        self.q_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        *,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        """x [T, hc_mult, dim]; hash_ids [T, n_hash_cols] for this layer."""
        # The lookup runs first even for an idle DP-attention batch: under DP
        # attention it is a collective every rank has to join.
        emb = self.embed(hash_ids, forward_batch, cp_all_tokens=cp_all_tokens)
        if x.shape[0] == 0:
            # Nothing to gate, and the MXFP8 quantize behind wkv rejects an
            # empty M.
            return x
        kv, _ = self.wkv(emb.flatten(-2))
        return self.apply_gate(x, kv)

    def project(
        self, hash_ids: torch.Tensor, *, cp_all_tokens: bool = False
    ) -> torch.Tensor:
        kv, _ = self.wkv(self.embed(hash_ids, cp_all_tokens=cp_all_tokens).flatten(-2))
        return kv

    def apply_gate(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        return engram_gate(
            x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value
        )
