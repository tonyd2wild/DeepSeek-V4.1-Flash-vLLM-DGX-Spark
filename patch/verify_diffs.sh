#!/bin/bash
# verify_diffs.sh <vllm-checkout-at-e47aa780b> - apply patch/full/*.diff to a scratch copy of the seven files and compare
# the results with the mounted files in patch/ (byte-identical = OK). Also re-plays the per-fix diffs in order.
set -u
V="${1:?usage: verify_diffs.sh <vllm checkout at e47aa780b>}"; P="$(cd "$(dirname "$0")" && pwd)"; W=$(mktemp -d); rc=0
while read -r f rel; do
  [ -z "$f" ] && continue
  mkdir -p "$W/$(dirname "$rel")"; cp "$V/vllm/$rel" "$W/$rel"
  if patch -s -p2 -d "$W" "$rel" < "$P/full/${f%.py}.diff" 2>/dev/null && cmp -s "$W/$rel" "$P/$f"; then echo "ok   full/${f%.py}.diff -> $f"
  else echo "FAIL full/${f%.py}.diff -> $f"; rc=1; fi
done < "$P/mounts.txt"
# per-fix chains
chain() { # chain <rel> <final-file> <diff>...
  local rel="$1" final="$2"; shift 2; local t="$W/chain-$(basename "$rel")"; cp "$V/vllm/$rel" "$t"
  for d in "$@"; do patch -s -p2 "$t" < "$d" 2>/dev/null || { echo "FAIL chain $(basename "$d")"; rc=1; return; }; done
  cmp -s "$t" "$final" && echo "ok   chain -> $(basename "$final") via $#" || { echo "FAIL chain result != $(basename "$final")"; rc=1; }
}
chain models/deepseek_v4_1/attention.py "$P/attention.py" "$P/sm12x-pages/attention.py.diff" "$P/sm12x-pages/indexer-64state.diff"
chain models/deepseek_v4_1/common/engram.py "$P/cudagraph-prestage/engram.py" "$P/engram-offset-fix/engram-offset-fix.diff" "$P/engram-parallel-reads/engram-parallel-reads.diff" "$P/cudagraph-prestage/engram-prestage.diff"
chain models/deepseek_v4_1/nvidia/model_state.py "$P/model_state.py" "$P/cudagraph-prestage/model_state-prestage.diff"
chain model_executor/layers/sparse_attn_indexer.py "$P/sparse_attn_indexer.py" "$P/sm12x-indexer-topk/sparse_attn_indexer-sm12x-topk.diff"
rm -rf "$W"; exit $rc
