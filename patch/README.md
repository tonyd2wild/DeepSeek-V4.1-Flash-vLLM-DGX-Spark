# patch/: the files bind-mounted over the vLLM image

The top-level files here are byte-identical to what the serving boot (boot 10) mounts from `~/patches/dsv41-boot10/`
on every node. Boots 3-9 mounted `~/patches/dsv41-boot3/`: the same files, except `engram.py` was md5 0ae8f1a5 there. `mounts.txt` maps each file to its site-packages path (`/usr/local/lib/python3.12/dist-packages/vllm/<path>`).

| file | md5 | mounted over | fix |
|---|---|---|---|
| `engram.py` | c0329107 | `models/deepseek_v4_1/common/engram.py` | Engram tables on disk, rank-offset fix, one shared parallel read pool, `EngramDiskStager`, optional node-local rows (`DSV41_ENGRAM_DIR`, used only when the copied row range covers the rank) |
| `model_state.py` | 0a14bee6 | `models/deepseek_v4_1/nvidia/model_state.py` | Engram rows staged in `prepare_inputs`, before the (CUDA-graph captured) forward |
| `weight_utils.py` | 7e1027f1 | `model_executor/model_loader/weight_utils.py` | loader skips the two Engram tables |
| `attention.py` | da9ef196 | `models/deepseek_v4_1/attention.py` | SM12x page sizes (Kai) + indexer cache pages of 64 states |
| `flashinfer_sparse.py` | af0f8447 | `models/deepseek_v4_1/nvidia/flashinfer_sparse.py` | SM12x 64-state compressed pages, 64-token SWA backend (Kai) |
| `sparse_swa.py` | cc419353 | `v1/attention/backends/mla/sparse_swa.py` | `get_swa_block_size()` hook (Kai) |
| `sparse_attn_indexer.py` | a9b73756 | `model_executor/layers/sparse_attn_indexer.py` | SM12x decode top-k uses `top_k_per_row_decode` (same gate as Kai's `38d9c39`; comment differs) |
| `mounts.txt` | 79a774bc | | manifest (7 lines) |

All diffs are `diff -u` with repo-relative paths (`a/vllm/...`) against vLLM `dsv41-feat` @ `e47aa780b`, so `git apply` works
from a checkout at that commit; `full/` has one diff per mounted file and `verify_diffs.sh` proves they reproduce the mounted
files byte for byte. The subfolders hold each fix's diff, offline test and notes:

| folder | contents |
|---|---|
| `engram-offset-fix/` | rank-offset fix, diff, harness (ranks 1-3 read rank 0's rows before) |
| `engram-parallel-reads/` | first parallel-read patch (superseded by `cudagraph-prestage/`) and its latency harness |
| `engram-nowait/` | optional `RWF_NOWAIT` page-cache fast path for the blocking `preadv` fallback, exact diff and production evidence |
| `cudagraph-prestage/` | prestage diffs for engram.py and model_state.py, `test_engram_prestage.py` (33 checks, bit-exact) |
| `sm12x-pages/` | Kai's SM12x page-size diffs, the indexer 64-state diff, `test_indexer64.py` |
| `sm12x-indexer-topk/` | top-k diff, GB10 correctness and timing tests, `RESULTS.md` |
| `full/` | one diff per mounted file from the pinned vLLM commit to the boot-10 file; `git apply patch/full/*.diff` |

`apply_engram_patch.py`, `engram_helper.py` and `test_engram_disk.py` are from the first Engram-on-disk scaffold (unit test of the
original reader).
