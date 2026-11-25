# Changelog

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
