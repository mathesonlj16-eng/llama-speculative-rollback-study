# v1.5.1–v1.5.2: operator-level reproduction of the FA batch-invariance gap (2026-11-25)

**Where things are:**
- Per-case results: [`fa-padkv-test.csv`](../../results/v1.5/fa-padkv-test.csv)
- Test patch: [`test-backend-ops-fa-padkv-vs-pr-head.patch`](../../patches/v1.5/test-backend-ops-fa-padkv-vs-pr-head.patch), which applies cleanly to PR #32081 head `3f25ea199`
- Runner: `bash scripts/v1.5/run.sh padkv`
- Raw logs are archived outside this repository.

**Prior art.** The FA fix these tests exercise follows makoni's earlier patch in [issue #25618](https://github.com/ggml-org/llama.cpp/issues/25618); see [FINDINGS.md](FINDINGS.md). That patch reports bitwise per-operation checks at fixed KV lengths but contains no test code. The `test-backend-ops` cases here, which vary the padded length and the stream count, are this study's.

## Why the PR's own test misses the gap

`test_flash_attn_ext_batch_inv` gives the batch call and the single-token calls the same K/V tensors, so `kv` never differs between them. Only the number of query tokens changes.

In llama.cpp, the KV cache view is padded to a multiple of 256 cells, counted from the furthest position written (`llama_kv_cache::get_n_kv`). So one token at position p sees `GGML_PAD(p+1, 256)` cells. A batch ending at position p0+nb−1 sees `GGML_PAD(p0+nb, 256)` cells. v1.4.0 traced the remaining model-level discrepancy to exactly this difference.

## The test

`test_flash_attn_ext_batch_inv_padkv` is the PR's test with the padded lengths llama.cpp produces:
- **Batch.** `nb` queries at positions p0…p0+nb−1 see K/V of length `GGML_PAD(p0+nb, 256)`.
- **Single tokens.** Each single-token call `j` sees a view of the first `GGML_PAD(p0+j+1, 256)` cells of the same tensors.
- **Mask.** Causal: query `j` sees cells 0…p0+j. The unmasked cells are identical in both calls.

**Cases** (40 in total, counting the PR's 16 FA cases):
- KV types: F16 and Q8_0.
- Head shapes (hs, nh_kv, gqa): (256, 2, 4), which is Qwen3.5-2B's attention, and (128, 8, 4), the PR's own.
- (p0, nb) pairs that cross a 256-cell boundary: (250, 8), (255, 2), (500, 16), (1020, 8).
- Controls that stay within one padded length: (100, 8), (300, 8).

**Build.** The test compiled once and was linked against the two v1.5.0 builds:
- `pr-bi`: the PR head
- `pr-bi-fafix`: the PR head plus [`fa-kv-split-invariant-vs-pr-head.patch`](../../patches/v1.5/fa-kv-split-invariant-vs-pr-head.patch)

Compile and link took 36 s; each run takes seconds.

## Results

`test-backend-ops batch -o FLASH_ATTN_EXT -b CUDA0`, on a GTX 1650 (sm_75) with CUDA 13.2:

| Cases | PR head | PR head + FA patch |
|---|---|---|
| PR's `test_flash_attn_ext_batch_inv` (16) | 16 OK | 16 OK |
| padkv controls (8) | 8 OK | 8 OK |
| padkv, boundary crossed (16) | **7 FAIL**, 9 OK | **16 OK** |
| Total | 33/40, exit 1 | **40/40**, exit 0 |

**Failures on the PR head:**
- Qwen shape (256, 2, 4) at p0 = 250, 255 and 500, with F16 and Q8_0.
- PR shape (128, 8, 4) at p0 = 500 with F16.
- In each, 42–65 % of output elements differ, with max diff 3.0e-8 to 4.5e-8.
- In the model these grow, over 6 attention layers and the layers that follow, into the logit differences of up to 2.88 seen in v1.4.0.

**Crossings that pass:** every p0 = 1020 case (1024 → 1280 cells), and the PR shape at 250, 255 and at 500 with Q8_0. In those cases the `parallel_blocks` search picks the same split for both lengths. Whether the gap shows depends on the head count and the lengths involved.

## v1.5.2: two sequences in one call

**Patch.** [`test-backend-ops-fa-padkv-streams-vs-pr-head.patch`](../../patches/v1.5/test-backend-ops-fa-padkv-streams-vs-pr-head.patch) is a superset of the patch above and applies cleanly to `3f25ea199`. It adds `test_flash_attn_ext_batch_inv_streams`. Per-case results: [`fa-streams-test.csv`](../../results/v1.5/fa-streams-test.csv).

**What the test does.**
- Two sequences go in separate KV streams in one FA call (`Q->ne[3]` = 2), as with llama.cpp's non-unified KV cache and two slots.
- The same queries are also computed with one FA call per sequence (`Q->ne[3]` = 1).
- The K/V length `L` and the causal masks are identical in both, so only the stream count differs.
- This exercises the second half of the v1.5.0 patch, which takes `Q->ne[3]` out of the split sizing (`fattn-common.cuh:1263` on the PR head).

**Cases.** 24 in total:
- KV types: F16 and Q8_0.
- Head shapes (hs, nh_kv, gqa): (256, 2, 4) and (128, 8, 4).
- L ∈ {256, 1024, 4096}.
- nb ∈ {1, 4} tokens per sequence.

| Cases | PR head | PR head + FA patch |
|---|---|---|
| PR's own FA cases (16) | 16 OK | 16 OK |
| padkv (24) | 17 OK, 7 FAIL | 24 OK |
| streams (24) | **16 OK, 8 FAIL** | **24 OK** |
| Total | 49/64 | **64/64** |

**Stream failures on the PR head:**
- Qwen shape at L = 4096, nb ∈ {1, 4}, F16 and Q8_0.
- PR shape with Q8_0 at L ∈ {1024, 4096}, nb ∈ {1, 4}.
- 88–93 % of elements differ, with max diff 1.1e-8 to 2.6e-8.
- Every L = 256 case passes, because the `ntiles_KV` clamp leaves the split unchanged.

**The failures include nb = 1.** In invariant mode on the PR head, one decode token per sequence in a two-sequence batch doesn't match single-sequence decoding at long contexts. That is the study's two-slot setting.

**Reproducibility.** The padkv cases were rerun in this binary. Every case's pass/fail status is identical to the v1.5.1 run. `test-backend-ops` draws fresh random inputs each run, so for the 7 failing cases the differing-element counts vary by under 2 % between the two runs. The max diffs stay between 3.0e-8 and 4.5e-8.

Compile and link took 37 s. Runner: `bash scripts/v1.5/run.sh streams`.

## Limits

- One GPU and the vector FA kernel only.
- The stream test covers two streams with equal K/V length. It does not combine different stream counts with different padded lengths.
- Nothing has been posted upstream.
