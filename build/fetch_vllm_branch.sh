#!/bin/bash
# fetch_vllm_branch.sh [dir] - check out vLLM at the commit the patches in patch/ target.
# dsv41-feat was force-pushed on 2026-09-11 00:49 UTC, merged into main (PR #56214) and deleted at 09:11 UTC the same day.
# The pinned commit is still served by GitHub by full sha (and branch dsv41-optimized pointed at it on 2026-09-12).
# See docs/RECIPE.md "Pin the branch commit".
set -e
SHA="${VLLM_BRANCH_SHA:-e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba}"
DIR="${1:-vllm}"
[ -d "$DIR/.git" ] || git clone https://github.com/vllm-project/vllm.git "$DIR"
git -C "$DIR" fetch origin "$SHA"
git -C "$DIR" checkout -q "$SHA"
echo "vllm @ $(git -C "$DIR" rev-parse HEAD)  ($(git -C "$DIR" log -1 --format=%cd --date=iso))"
