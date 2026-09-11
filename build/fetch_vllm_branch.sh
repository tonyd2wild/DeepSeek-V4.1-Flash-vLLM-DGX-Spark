#!/bin/bash
# fetch_vllm_branch.sh [dir] - check out vLLM branch dsv41-feat at the commit the patches in patch/ target.
# The branch was force-pushed on 2026-09-11; GitHub still serves the old commit by full sha. See docs/RECIPE.md "Pin the branch commit".
set -e
SHA="${VLLM_BRANCH_SHA:-e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba}"
DIR="${1:-vllm}"
[ -d "$DIR/.git" ] || git clone --branch dsv41-feat --single-branch https://github.com/vllm-project/vllm.git "$DIR"
git -C "$DIR" fetch origin "$SHA"
git -C "$DIR" checkout -q "$SHA"
echo "vllm dsv41-feat @ $(git -C "$DIR" rev-parse HEAD)  ($(git -C "$DIR" log -1 --format=%cd --date=iso))"
