# patch/full: one diff per mounted file, against vLLM `dsv41-feat` @ `e47aa780b`

Each file here is `diff -u` from the vLLM tree at commit `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba` to the mounted file in
`patch/` (the boot-10 set, md5s in `patch/README.md`). Paths are repo-relative (`a/vllm/...`), so from a vLLM checkout at that
commit (`build/fetch_vllm_branch.sh`):

```
git apply --check ../DeepSeek-V4.1-Flash-vLLM-DGX-Spark/patch/full/*.diff   # dry run
git apply         ../DeepSeek-V4.1-Flash-vLLM-DGX-Spark/patch/full/*.diff
```

`patch/verify_diffs.sh <vllm-checkout>` applies them to a scratch copy and checks that every result is byte-identical to the
mounted file. The per-fix diffs in the sibling folders are the same changes split by fix (`engram.py` is three layers:
offset fix, parallel reads, prestage; the boot-10 node-local-rows layer is only in this folder's full diff).
