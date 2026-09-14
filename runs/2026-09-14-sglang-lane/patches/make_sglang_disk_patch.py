#!/usr/bin/env python3
"""make_sglang_disk_patch.py <sglang-src-root> <out-dir>

Builds our SGLang patch set `sglang-dsv41-disk` from the files of lmsysorg/sglang:dev-dsv41 (sglang da64c5cb):
  engram.py        (srt/layers/engram.py)          disk-backed Engram rows (SGLANG_DSV41_ENGRAM_DISK=1)
  weight_utils.py  (srt/model_loader/weight_utils.py) honor model.safetensors.index.json per tensor, and skip
                                                    the Engram embed tables in disk mode
Every edit is an anchored string replacement that must match exactly once. Original SGLang code: Apache-2.0.
Our additions: MIT, Tech2Wild 2026-09-14 (ported from our own vLLM patches, Kai 2026-09-10 and the speed run).
"""
import difflib
import os
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def patch(rel, edits, out_name):
    path = os.path.join(SRC, rel)
    text = open(path).read()
    orig = text
    for old, new in edits:
        n = text.count(old)
        assert n == 1, f"{rel}: anchor found {n} times:\n{old[:200]}"
        text = text.replace(old, new)
    open(os.path.join(OUT, out_name), "w").write(text)
    diff = difflib.unified_diff(orig.splitlines(True), text.splitlines(True), f"a/{rel}", f"b/{rel}")
    open(os.path.join(OUT, out_name + ".diff"), "w").writelines(diff)
    print(f"{out_name}: {len(edits)} edits")


DISK_BLOCK = '''
# ---------------------------------------------------------------------------
# Tech2Wild 2026-09-14 (MIT): disk-backed Engram rows for DGX Spark (GB10).
# Ported from our vLLM patches (tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark): Kai's DiskEngramTable
# (2026-09-10) and the speed-run fast staging (2026-09-14: numpy memmap gather split over a thread pool,
# raw fp8 rows and e8m0 scales dequantized on the GPU). On GB10 the GPU and the host share one 121.7 GiB
# pool, so neither layout above (device rows or a host table) leaves room for the ~47 GiB per-rank Engram
# shard next to the weights. With SGLANG_DSV41_ENGRAM_DISK=1 a rank keeps read-only memmaps of the
# checkpoint's Engram tensors and gathers only the rows a step needs; the page cache holds the hot rows.
# SGLANG_DSV41_ENGRAM_DIR may point at a node-local sparse copy of the Engram shards (engram-local.json
# names the row range per layer that exists there); rows outside that range come from the model directory.
# The lookup runs as a graph break (eager_on_graph), so the decode CUDA graph backend must be "breakable".
# ---------------------------------------------------------------------------
_DSV41_ENGRAM_DISK = os.environ.get("SGLANG_DSV41_ENGRAM_DISK", "0") == "1"
_DSK_THREADS = int(os.environ.get("SGLANG_DSV41_ENGRAM_DISK_THREADS", "128"))
_DSK_POOL = None
_DSK_LOOKUP = None


def _dsk_pool():
    global _DSK_POOL
    if _DSK_POOL is None:
        import concurrent.futures

        _DSK_POOL = concurrent.futures.ThreadPoolExecutor(
            max_workers=_DSK_THREADS, thread_name_prefix="engram-disk"
        )
    return _DSK_POOL


def _dsk_memmaps(model_dir: str, layer_id: int):
    """Read-only memmaps over one Engram layer's full embed.weight [N, dim] and
    embed.scale [N, dim // 32] tensors (uint8) in the safetensors shards."""
    import json
    import struct

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    maps = []
    for suffix in ("weight", "scale"):
        name = f"layers.{layer_id}.engram.embed.{suffix}"
        path = os.path.join(model_dir, weight_map[name])
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))[name]
        off = 8 + n + meta["data_offsets"][0]
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=tuple(meta["shape"]))
        try:
            mm._mmap.madvise(mmap.MADV_RANDOM)
        except Exception:  # noqa: BLE001
            pass
        maps.append(mm)
    return maps[0], maps[1]


def _dsk_take(mm, idx, out, futs: list) -> None:
    """out[i] = mm[idx[i]]; large gathers are split over the pool so cold rows
    fault in parallel (numpy's take loop releases the GIL)."""
    n = idx.shape[0]
    step = min(1024, max(16, -(-n // _DSK_THREADS)))
    if n <= step:
        np.take(mm, idx, axis=0, out=out)
        return
    pool = _dsk_pool()
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        futs.append(pool.submit(np.take, mm, idx[lo:hi], 0, out[lo:hi]))


class _DiskEngramRows:
    """This rank's Engram rows [row_start, row_start + rows), read on demand."""

    def __init__(self, layer_id, row_start, rows, dim, model_dir=None, local_dir=None):
        import json

        self.layer_id = layer_id
        self.row_start = row_start
        self.rows = rows
        self.dim = dim
        self.sb = dim // FP8_BLOCK_SIZE
        if model_dir is None:
            model_dir = get_model().model_path
        if local_dir is None:
            local_dir = os.environ.get("SGLANG_DSV41_ENGRAM_DIR", "") or model_dir
        lo, hi = 0, None
        info = os.path.join(local_dir, "engram-local.json")
        if local_dir != model_dir and os.path.exists(info):
            with open(info) as f:
                lo, hi = json.load(f)["layers"][str(layer_id)]
        self.w, self.s = _dsk_memmaps(local_dir, layer_id)
        n = self.w.shape[0]
        assert self.w.shape[1] == dim and tuple(self.s.shape) == (n, self.sb), (self.w.shape, self.s.shape)
        self.lo, self.hi = lo, (n if hi is None else hi)
        a, b = row_start, row_start + rows
        self.fallback = None
        if rows and not (self.lo <= a and b <= self.hi):
            self.fallback = _dsk_memmaps(model_dir, layer_id)
        missing = (max(0, min(self.lo, b) - a) + max(0, b - max(self.hi, a))) if rows else 0
        self._w_buf = self._s_buf = None
        logging.getLogger(__name__).info(
            "Engram DISK layer %d: rows [%d, %d) from %s (rows [%d, %d) there); %d rows from %s",
            layer_id, a, b, local_dir, self.lo, self.hi, missing,
            model_dir if self.fallback is not None else "-",
        )

    def _buffers(self, u: int):
        if self._w_buf is None or self._w_buf.shape[0] < u:
            cap = max(u, 4096)
            pin = torch.cuda.is_available()
            self._w_buf = torch.empty((cap, self.dim), dtype=torch.uint8, pin_memory=pin)
            self._s_buf = torch.empty((cap, self.sb), dtype=torch.uint8, pin_memory=pin)
        return self._w_buf[:u], self._s_buf[:u]

    def gather(self, u):
        """uint8 rows [U, dim] and scales [U, sb] (pinned host) for global row ids u (int64 numpy)."""
        w_t, s_t = self._buffers(u.shape[0])
        w_np, s_np = w_t.numpy(), s_t.numpy()
        futs: list = []
        parts = []
        if self.fallback is None:
            _dsk_take(self.w, u, w_np, futs)
            _dsk_take(self.s, u, s_np, futs)
        else:
            here = (u >= self.lo) & (u < self.hi)
            for mask, (wm, sm) in ((here, (self.w, self.s)), (~here, self.fallback)):
                pos = np.nonzero(mask)[0]
                if pos.size == 0:
                    continue
                tw = np.empty((pos.size, self.dim), np.uint8)
                ts = np.empty((pos.size, self.sb), np.uint8)
                _dsk_take(wm, u[pos], tw, futs)
                _dsk_take(sm, u[pos], ts, futs)
                parts.append((pos, tw, ts))
        for fut in futs:
            fut.result()
        for pos, tw, ts in parts:
            w_np[pos] = tw
            s_np[pos] = ts
        return w_t, s_t


def _disk_owned_rows_impl(disk, indices: torch.Tensor) -> torch.Tensor:
    """Same result as the device-sharded path: this rank's rows dequantized to bf16,
    zero for rows another rank owns (the caller's all-reduce sums the ranks)."""
    flat = indices.reshape(-1).to(torch.int64)
    owned = (flat >= disk.row_start) & (flat < disk.row_start + disk.rows)
    safe = torch.where(owned, flat, torch.full_like(flat, disk.row_start))
    uniq, inverse = torch.unique(safe, return_inverse=True)
    w_t, s_t = disk.gather(uniq.cpu().numpy())
    dev = indices.device
    w = w_t.to(dev, non_blocking=True).view(torch.float8_e4m3fn)
    s = s_t.to(dev, non_blocking=True).view(torch.float8_e8m0fnu)
    rows = w.float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
    vals = (rows * s.float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    out = vals.index_select(0, inverse).masked_fill(~owned.unsqueeze(-1), 0)
    return out.view(*indices.shape, disk.dim)


def _disk_lookup(disk, indices: torch.Tensor) -> torch.Tensor:
    global _DSK_LOOKUP
    if _DSK_LOOKUP is None:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            eager_on_graph,
        )

        _DSK_LOOKUP = eager_on_graph(True)(_disk_owned_rows_impl)
    return _DSK_LOOKUP(disk, indices)


class EngramEmbedding(nn.Module):
'''

ENGRAM_EDITS = [
    ("\nclass EngramEmbedding(nn.Module):\n", DISK_BLOCK),
    (
        "        self.host_table: Optional[_HostTable] = None\n"
        "        if envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():\n",
        "        self.host_table: Optional[_HostTable] = None\n"
        "        self.disk: Optional[_DiskEngramRows] = None\n"
        "        if _DSV41_ENGRAM_DISK:\n"
        "            # Tech2Wild 2026-09-14: rows stay on disk. No weight/scale parameters, and the\n"
        "            # patched weight iterator skips these tensors, so nothing is loaded for them.\n"
        "            self.disk = _DiskEngramRows(layer_id, self.row_start, self.rows, dim)\n"
        "            self.weight = None\n"
        "            self.scale = None\n"
        "            return\n"
        "        if envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():\n",
    ),
    (
        '        """Rows of `indices` this rank\'s shard holds, zero for the rest."""\n'
        "        if self.rows == 0:\n",
        '        """Rows of `indices` this rank\'s shard holds, zero for the rest."""\n'
        "        if self.disk is not None:\n"
        "            return _disk_lookup(self.disk, indices)\n"
        "        if self.rows == 0:\n",
    ),
]

KEEP_BLOCK = '''# ---------------------------------------------------------------------------
# Tech2Wild 2026-09-14 (MIT): yield a tensor only from the file model.safetensors.index.json maps it to,
# so an overlay file that repoints some tensors (our uncensored wo_b layers 10-35) wins on every rank no
# matter how SGLANG_SORT_WEIGHT_FILES orders or staggers the files. With SGLANG_DSV41_ENGRAM_DISK=1 the
# Engram embed tables (read from disk at run time) are skipped before they are ever materialized.
# ---------------------------------------------------------------------------
_DSV41_INDEX_CACHE: dict = {}
_DSV41_SKIP_ENGRAM = os.environ.get("SGLANG_DSV41_ENGRAM_DISK", "0") == "1"
_DSV41_ENGRAM_RE = re.compile(r"\\.engram\\.embed\\.(weight|scale)$")


def _dsv41_keep(st_file: str, name: str) -> bool:
    if _DSV41_SKIP_ENGRAM and _DSV41_ENGRAM_RE.search(name):
        return False
    folder = os.path.dirname(st_file)
    weight_map = _DSV41_INDEX_CACHE.get(folder)
    if weight_map is None:
        weight_map = {}
        index_path = os.path.join(folder, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {}) or {}
        _DSV41_INDEX_CACHE[folder] = weight_map
    owner = weight_map.get(name)
    return owner is None or owner == os.path.basename(st_file)


def safetensors_weights_iterator(
'''

WEIGHT_EDITS = [
    ("def safetensors_weights_iterator(\n", KEEP_BLOCK),
    (
        "                for name in sorted(result.keys()):\n"
        "                    yield name, result[name]\n",
        "                for name in sorted(result.keys()):\n"
        "                    if _dsv41_keep(st_file, name):\n"
        "                        yield name, result[name]\n",
    ),
    (
        "            with safetensors.safe_open(st_file, framework=\"pt\", device=\"cpu\") as f:\n"
        "                for name in f.keys():\n"
        "                    yield name, f.get_tensor(name)\n",
        "            with safetensors.safe_open(st_file, framework=\"pt\", device=\"cpu\") as f:\n"
        "                for name in f.keys():\n"
        "                    if _dsv41_keep(st_file, name):\n"
        "                        yield name, f.get_tensor(name)\n",
    ),
    (
        "                result = {k: f.get_tensor(k) for k in f.keys()}\n"
        "        return result\n",
        "                result = {k: f.get_tensor(k) for k in f.keys() if _dsv41_keep(st_file, k)}\n"
        "        return {k: v for k, v in result.items() if _dsv41_keep(st_file, k)}\n",
    ),
]

EXTRA_SPLIT_HELPER = '''def _split_extra_kv_to_64(extra_u8, extra_idx):
    """Tech2Wild 2026-09-14 (MIT): FlashInfer's SM120 sparse MLA reads the extra
    (low-ratio) KV source at page_block_size 64, like the main pool, but the paths
    below passed it through unsplit. Split its pages the same way, into a separate
    persistent buffer (the main pool's split buffer is still in use this step).
    Pass-through when the extra pool already pages at 64, or at a size that is not a
    multiple of 64 (unchanged from upstream in that case)."""
    if extra_u8 is None or extra_u8.ndim < 3:
        return extra_u8
    pbs = extra_u8.shape[1]
    if pbs <= _PBS_DST or pbs % _PBS_DST != 0:
        return extra_u8
    return _split_kv_pages_to_64(extra_u8, pbs, touched_indices=extra_idx, buf_tag="_extra")


def _flash_mla_flashinfer(
'''

FLASH_EDITS = [
    (
        "    touched_indices: Optional[torch.Tensor] = None,\n"
        ") -> torch.Tensor:\n"
        '    """Split pbs=N footer-format pages into pbs=64 footer-format pages.\n',
        "    touched_indices: Optional[torch.Tensor] = None,\n"
        '    buf_tag: str = "",\n'
        ") -> torch.Tensor:\n"
        '    """Split pbs=N footer-format pages into pbs=64 footer-format pages.\n',
    ),
    (
        '    key = f"flash_mla_sm120_split:{dev}"\n',
        '    key = f"flash_mla_sm120_split{buf_tag}:{dev}"\n',
    ),
    ("def _flash_mla_flashinfer(\n", EXTRA_SPLIT_HELPER),
    (
        "        extra_kv_cache=extra_kv_u8,\n",
        "        extra_kv_cache=_split_extra_kv_to_64(extra_kv_u8, extra_idx),\n",
    ),
    (
        "        extra_kv_cache=extra_kv_64,\n",
        "        extra_kv_cache=_split_extra_kv_to_64(extra_kv_64, extra_idx),\n",
    ),
]

BACKEND_EDITS = [
    (
        "            force_deep_gemm_metadata=(\n"
        "                self.enable_deepseek_v4_fp4_indexer and get_platform().is_sm120\n"
        "            ),\n",
        "            # Tech2Wild 2026-09-14 (MIT): force DeepGEMM's own planner for the V4.1\n"
        "            # ratio-1/2 sources on SM120 too; it matches DeepGEMM's SM120 kernels\n"
        "            # (split_kv=128), the generic JIT planner encodes split_kv=256.\n"
        "            force_deep_gemm_metadata=(\n"
        "                (self.enable_deepseek_v4_fp4_indexer or compress_ratio in (1, 2))\n"
        "                and get_platform().is_sm120\n"
        "            ),\n",
    ),
    (
        "            use_topk_v2=False,\n"
        "            use_prefill_cuda_graph=True,\n",
        "            use_topk_v2=False,\n"
        "            force_deep_gemm_metadata=get_platform().is_sm120,  # Tech2Wild 2026-09-14 (MIT)\n"
        "            use_prefill_cuda_graph=True,\n",
    ),
]

# engram.py v2 (14:15): the design agent's full version (our code, marked "[DSV41 disk]": whole-hash-head
# ownership that matches the node-local copies, an eager memmap path, and a CUDA-graph path that runs the
# native reader dsv41_rowio.cpp as a host node) replaces the anchored-edit version above, which needed
# breakable decode graphs that upstream V4.1 does not support. Made from the image's original (md5 checked).
import hashlib
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
orig = os.path.join(SRC, "sglang/srt/layers/engram.py")
assert hashlib.md5(open(orig, "rb").read()).hexdigest() == "3d2006a91c019c29c2ba81fa2c0296e7", "image engram.py changed"
shutil.copy(os.path.join(HERE, "src", "engram.py"), os.path.join(OUT, "engram.py"))
diff = difflib.unified_diff(open(orig).read().splitlines(True), open(os.path.join(OUT, "engram.py")).read().splitlines(True),
                            "a/sglang/srt/layers/engram.py", "b/sglang/srt/layers/engram.py")
open(os.path.join(OUT, "engram.py.diff"), "w").writelines(diff)
print("engram.py: agent version (v2) copied from src/engram.py")
shutil.copy(os.path.join(HERE, "src", "libdsv41_rowio.so"), os.path.join(OUT, "libdsv41_rowio.so"))
shutil.copy(os.path.join(HERE, "src", "dsv41_rowio.cpp"), os.path.join(OUT, "dsv41_rowio.cpp"))
patch("sglang/srt/model_loader/weight_utils.py", WEIGHT_EDITS, "weight_utils.py")
patch("sglang/kernels/ops/attention/flash_mla_sm120.py", FLASH_EDITS, "flash_mla_sm120.py")
patch("sglang/srt/layers/attention/deepseek_v4_backend.py", BACKEND_EDITS, "deepseek_v4_backend.py")
open(os.path.join(OUT, "mounts.txt"), "w").write(
    "engram.py srt/layers/engram.py\nweight_utils.py srt/model_loader/weight_utils.py\n"
    "flash_mla_sm120.py kernels/ops/attention/flash_mla_sm120.py\n"
    "deepseek_v4_backend.py srt/layers/attention/deepseek_v4_backend.py\n"
    "libdsv41_rowio.so srt/layers/libdsv41_rowio.so\n"
)
print("mounts.txt written")
