# Changelog

## v1.7.0 (2026-11-26)

Extends the v1.6.0 kernels to Q4_K and Q6_K at 2–8 tokens per batch, adds a sanitizer pass and a matched layout A/B, and collects the batch-invariance results in an evidence pack.

- **Added** Q4_K/Q6_K kernels for N = 2..8 (`mmq-bi-nb-kq.cuh`). They keep v1.6.0's per-output float program, stage weights once per chunk and fold each (row, column) serially.
- **Proved** bit identity with forced MMQ:
  - 0 differences in 46,812,672 floats over 13 draft shapes and 7 input families, including −0.0;
  - `test-backend-ops batch` 112/112;
  - every draft and target fixture plan byte-identical;
  - speculative tokens and full logits equal plain decoding.
- **Found and fixed** a Q4_K slowdown at N = 7–8 (up to 1.64× default). The cause was shared-memory occupancy (38.4 KB per block), not registers. The fix is compact staging at N ≥ 3 and the original float4 layout at N = 2.
- **Measured:**
  - per op: Q4_K 0.73–1.16× default at N = 2 and 0.72–0.98× at N = 8; Q6_K 0.66–1.32×;
  - invariant-mode speculation at k = 1: 67.9–70.0 tok/s, 1.03–1.05× plain invariant decoding, up from 0.79×. This changes v1.6.0's "plain invariant decoding is fastest" for k = 1; larger k still loses at short context.
- **Measured** that the staging layout's effect end to end is within run-to-run noise (≤ 2%), in a matched A/B. Between-session variation is about ±3%.
- **Ran** compute-sanitizer: memcheck, racecheck and synccheck are clean on all fast kernels. Every initcheck report is in upstream's forced-MMQ `need_check` kernel, identically at PR head. Source analysis shows these are intended padding over-reads that are never consumed.
- **Added** `patches/v1.7/` (superseding v1.6's patch), `provenance/v1.7/` (FINDINGS.md, EVIDENCE.md, run-log.txt), `results/v1.7/` and `scripts/v1.7/` (harness, archived drivers, perf summary), plus v1.7.0 checks in `scripts/verify-results.py`.
- **Changed** README, REPRODUCE and LICENSING.

## v1.6.0 (2026-11-26)

Adds fast kernels for PR #32081's batch-invariant mode on sm_75 that keep the PR's invariant bits, and an end-to-end speculative-decoding check.

- **Found** that forced MMQ's per-output order on sm_75 is a serial fold in ascending K with exact integer block dots. A GEMV can reproduce it bit for bit without MMA instructions. The patch adds such kernels for:
  - Q8_0 at N = 1..8;
  - Q4_K and Q6_K at N = 1.

  They are gated to sm_75 and include the v1.5.0 FA fix.
- **Proved** bit identity with forced MMQ:
  - operator sweeps over 6–8 adversarial input families on every model shape, including the 248320-row heads;
  - deliberately wrong variants detected in essentially every cell;
  - `test-backend-ops batch` 112/112;
  - every fixture plan of the 2B target and 0.8B draft byte-identical to PR head + FA fix.

  Default mode is unchanged.
- **Measured** in invariant mode:
  - 2B decode: 70.0–72.9 tok/s, 1.09–1.14× default mode, up from 0.31–0.33×;
  - 0.8B decode: 1.04–1.07×;
  - 2–8-token verify batches: 0.85–0.92× default's time, down from 1.66–2.97×;
  - prefill unchanged.

  Being faster than default reflects MMVQ on these shapes on this GPU.
- **Proved** that greedy speculative decoding in invariant mode reproduces plain decoding's tokens and full logits byte for byte, for k = 1..7 and both rollback methods. Default mode differs from k = 2.
- **Measured** that speculation is still slower than plain decoding at short context in both modes. Plain invariant decoding is the fastest configuration.
- **Added** `patches/v1.6/`, `provenance/v1.6/` (FINDINGS.md, run-log.txt), `results/v1.6/` and `scripts/v1.6/` (harnesses, analysis, parameterized runner), plus v1.6.0 checks in `scripts/verify-results.py`.
- **Changed** README, REPRODUCE and LICENSING.

## v1.5.3 (2026-11-25)

A prior-art correction. No results change.

- **Credited** makoni's 2026-11-03 patch in [issue #25618](https://github.com/ggml-org/llama.cpp/issues/25618). It already makes the flash-attention KV split independent of the padded KV length: no `ntiles_KV` clamp, a search capped at one wave, per-column sizing. v1.4.0–v1.5.2 presented the v1.5.0 change without this reference. The issue also reports a symptom close to this study's.
- **Credited** [PR #16016](https://github.com/ggml-org/llama.cpp/pull/16016) (September 2025, open) as earlier deterministic-attention work.
- **Changed** the README's v1.5.0 paragraph and the v1.4/v1.5 write-ups. They now describe v1.5.0 as the earlier approach applied to PR #32081. What they claim as the study's own:
  - the regression in #32081;
  - removing the `Q->ne[3]` term, which #25618 keeps;
  - the `test-backend-ops` regression cases;
  - the cost measurement.

## v1.5.2 (2026-11-25)

Covers the second half of the v1.5.0 FA patch: the split must not depend on the number of sequences in the call.

- **Added** `test_flash_attn_ext_batch_inv_streams` in `patches/v1.5/test-backend-ops-fa-padkv-streams-vs-pr-head.patch`, a superset of the v1.5.1 test patch. It runs two sequences as separate KV streams in one FA call (`Q->ne[3]` = 2) and compares that with one call per sequence, at equal K/V length.
- **Found** that on the PR head 8 of its 24 cases fail, including nb = 1, i.e. two-slot decoding vs one-slot. They fail at L = 4096 for the Qwen shape and with Q8_0 at L ≥ 1024 for the PR shape, with 88–93 % of elements differing. With the v1.5.0 patch all 64 cases pass.
- **Rerun** of the v1.5.1 cases in the new binary: every pass/fail status is the same. The counts vary slightly because the inputs are random.
- **Added** `results/v1.5/fa-streams-test.csv`, a v1.5.2 section in `provenance/v1.5/padkv-test.md`, a `streams` step in `scripts/v1.5/run.sh`, and v1.5.2 checks in `scripts/verify-results.py`.
- **Changed** README and REPRODUCE.

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
