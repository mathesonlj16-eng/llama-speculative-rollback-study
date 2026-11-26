# Changelog

## v1.5.1 (2026-11-25)

Adds an operator-level reproduction of the flash-attention gap from v1.4.0 and v1.5.0. No model is needed.

- **Found** why the PR's `test_flash_attn_ext_batch_inv` misses the gap: it never changes the K/V length between the batch and single-token calls.
- **Added** `test_flash_attn_ext_batch_inv_padkv` (`patches/v1.5/test-backend-ops-fa-padkv-vs-pr-head.patch`). It uses llama.cpp's 256-cell padding from each call's position. On the PR head, 7 of its 16 boundary-crossing cases fail, with 42–65 % of elements differing; its 8 controls and the PR's 16 FA cases pass. With the v1.5.0 FA patch, all 40 cases pass.
- **Added** `provenance/v1.5/padkv-test.md`, `results/v1.5/fa-padkv-test.csv`, a `padkv` step in `scripts/v1.5/run.sh`, and v1.5.1 checks in `scripts/verify-results.py`.
- **Changed** README (a v1.5.1 paragraph) and REPRODUCE.

## v1.5.0 (2026-11-25)

Adds a fix for the flash-attention gap that v1.4.0 found in PR #32081's batch-invariant mode, with its correctness and cost.

- **Found** that one change, gated on the flag, makes every tested grouping bit-identical with no KV pin, two-sequence batches included. In invariant mode, it sizes the flash-attention `parallel_blocks` search for one sequence and caps it at one wave instead of the padded KV length. Default mode is bitwise unchanged.
- **Measured** the cost in 20 interleaved runs: 0.4–0.7 % slower prefill, decode within ±0.3 %. Invariant mode itself costs about 3.1× in decode and 21–23 % in prefill, from forced MMQ.
- **Added** `provenance/v1.5/FINDINGS.md` and `run-log.txt`; `results/v1.5/`; `patches/v1.5/fa-kv-split-invariant-vs-pr-head.patch`; `scripts/v1.5/` (runner, timing tool, analysis); and v1.5.0 checks in `scripts/verify-results.py`.
- **Changed** README (a v1.5.0 paragraph), REPRODUCE (a v1.5.0 section) and LICENSING (the v1.5 patch directory).

## v1.4.0 (2026-11-25)

Resolves the README's open question of which CUDA kernels caused the grouping-dependent logits. Each fix was tested on the PR #32081 head, with the flag off and on in the same binary.

- **Found** that the batch-invariant mode removes the effect on the original 11-token, one-sequence fixture (E1): 24/34 plans differ with the flag off, 0/34 with it on. The flag-off output is byte-identical to the v1.0.0 archive.
- **Found** a second, uncovered source on longer sequences. The flash-attention KV split follows the KV length, which is padded to 256 and depends on the batch. With the flag on, single-sequence shapes kept 603–768 of 1440 positions different, with no top-token changes. Pinning `n_kv` to the cache size made them bit-identical.
- **Found** that the residual in two-sequence (`pair4`) runs comes from the per-stream cache size (1024 vs 2048 cells). At matching size it is bit-identical.
- **Added** `provenance/v1.4/FINDINGS.md` and `run-log.txt`; `results/v1.4/`; `patches/v1.4/`, a replay-tool toggle and the diagnostic `n_kv` pin; `scripts/v1.4/`; and v1.4.0 checks in `scripts/verify-results.py`.
- **Changed** the README (a new section, plus the two sentences that called the kernels unresolved or untested) METHODOLOGY (scope and strength of evidence) and LICENSING, which now names the v1.3 and v1.4 patch directories next to v1.2.
- **Fixed** three characters in the README's v1.3.0 paragraph that had been saved in Windows-1252 rather than UTF-8 since v1.3.0 (–, ×, ·).

## v1.3.2 (2026-11-25)

Reports a batch-invariance bug in PR #32081 on the stock sm_75 build. In invariant mode, BF16 MUL_MAT_ID fails at every tested N, and F16 MUL_MAT_ID fails once the batch exceeds MMF's token limit. The MoE fallback passes F16/BF16 src1 to `ggml_cuda_mul_mat`, which sends non-F32 src1 to cuBLAS ahead of the invariant branches.

- **Added** `provenance/v1.3/moe-gap.md`, `results/v1.3/moe-gap.csv`, `patches/v1.3/moe-invariance-candidate-fixes.patch` and the scripts.
- **Found** that candidate fix A (sort src1 to F32 in invariant mode) makes BF16 invariant, with default output unchanged. Fix B (skip MMF) trips `ggml_cuda_mul_mat_id_needs_sync`, so F16 at large N needs a design decision.
- **Added** v1.3.2 checks in `scripts/verify-results.py`.

## v1.3.1 (2026-11-25)

Extends v1.3.0's coverage. With Stream-K off, tile width still doesn't change the bits across 13 quant types, fallback-config M, K values that aren't a multiple of 256, and quantized MUL_MAT_ID with seeded routing at tile widths up to 128. The PR's own batch suite passes on stock sm_75.

- **Added** `provenance/v1.3/coverage.md`, coverage and MoE summaries in `results/v1.3/` and `provenance/v1.3/`, the harness patch `patches/v1.3/test-backend-ops-moe-vs-pr-head.patch`, and the scripts.
- **Noted** that, on the dp4a diagnostic build only, F16 MUL_MAT_ID fails the batch test whether J is pinned or free. This is unrelated to tile width; it may be an invariance gap on GPUs without Turing MMA, and hasn't been traced.
- **Added** v1.3.1 checks in `scripts/verify-results.py`.

## v1.3.0 (2026-11-25)

Tests whether PR #32081's batch-invariant MMQ needs to pin the tile width. On sm_75, for the 5 tested types, it doesn't: with Stream-K off, the outputs are bit-identical at any tile width. On the dp4a path, invariant mode then matches default MMQ at N=512.

- **Added** `provenance/v1.3/FINDINGS.md` and `provenance/v1.3/code-reading.md`.
- **Added** `patches/v1.3/mmq-tile-width-switch.patch`, a host-only diagnostic environment switch.
- **Added** `results/v1.3/`, `scripts/v1.3/` and v1.3.0 checks in `scripts/verify-results.py`.
- **Changed** `provenance/v1.2/cuda13-pascal-hint.md` (after v1.2.0). It now acknowledges upstream PR #15468, which was closed after the maintainer declined dual dp4a/MMA variants for CC 7.5. The rejected runtime-selection suggestion is gone. What remains is the CUDA-version-aware hint and the fix to the missing `-` in the flag.

## v1.2.0 (2026-11-25)

Adds an operator-level supplement on CUDA `MUL_MAT` batch invariance on the GTX 1650. It asks whether the Pascal-style (dp4a) MMQ path changes the cost of the two batch-invariant options: diagnostic chunked MMVQ and the forced-MMQ mode of PR #32081. It does: at N ≥ 16, the PR's forced MMQ goes from 6–47% slower than chunked MMVQ to between 37% faster and 11% slower. Nothing in v1.0.0 or v1.1.0 changes.

- **Added** `provenance/v1.2/FINDINGS.md`, with SASS, kernel-dispatch and accuracy checks, batch-invariance grids, the 70-run performance sweep, measurement quality, limits and cost.
- **Added** `provenance/v1.2/cuda13-pascal-hint.md`: llama.cpp's GTX 16xx hint (`61-virtual`) can't be built with CUDA 13. This note has not been posted upstream.
- **Added** `patches/v1.2/`: the macro-guarded `GGML_CUDA_DIAG_NO_TURING_MMA` switch, plus full patches against master `22b99122e` and PR head `3f25ea199`.
- **Added** `results/v1.2/`, `scripts/v1.2/` and v1.2.0 checks in `scripts/verify-results.py`.
- **Changed** the README (new supplement section) and LICENSING (the two new upstream bases).

## v1.1.0 (2026-11-24)

Investigates whether the CUDA toolkit change (13.1.115 to 13.2.86) explains the concurrent token and restore-count differences reported in v1.0.0. It doesn't.

- **Added** `provenance/v1.1/FINDINGS.md`: the differences came from the paired and flag-off servers' target snapshot capacity (0 in the archived flag-off and interleaved runs, 3 in the rerun). Nonzero capacity changes two-sequence microbatch composition. Every build reproduces the archive at capacity 0.
- **Added** the toolkit controls: clean CUDA 13.1 builds (device code identical to the archived 13.1 library), the original archived binaries rerun, and teacher-forced logits that are bit-identical between 13.1 and 13.2 in five batch shapes.
- **Added** a reverse-stagger admission control. The rerun's single unexplained flag-off output reproduces whenever prompt 1 is admitted at least 15 ms before prompt 0, for paired and flag-off builds alike.
- **Added** like-for-like interleaved series with paired at capacity 0: 13.9% (13.2) and 16.2% (13.1) median two-slot wall reduction for the prototype, consistent with the archived 15.3%.
- **Added** `provenance/v1.1/efficiency-retrospective.md`, `results/v1.1/`, `scripts/v1.1/`, and v1.1.0 checks in `scripts/verify-results.py`.
- **Changed** `scripts/run.sh` (backward compatible): optional `CAP_OVERRIDE` to replace the capacity preload, and a `reverse` experiment.
- **Changed** the README, METHODOLOGY and REPRODUCE statements that described the 34/39 modes, the toolkit hypothesis and the unexplained flag-off run as open.

## v1.0.0 (2026-11-24)

Initial publication. Tag `v1.0.0`, commit `57874bd`.
