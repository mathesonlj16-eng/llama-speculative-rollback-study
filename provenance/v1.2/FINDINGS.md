# v1.2.0: does the Pascal-style (dp4a) MMQ path change batch-invariance costs on a GTX 1650?

Run on 24 November 2026, 19:23–23:36 local time. Same host as v1.1.0: GTX 1650 (TU117, CC 7.5, no tensor cores), WSL2, driver 592.00. Toolchain: CUDA 13.2, GCC 15.2. This is a supplementary operator-level study. It makes no end-to-end inference claims and doesn't change any v1.0.0 or v1.1.0 result.

## Context

The README leaves the specific CUDA kernels behind the original batch-shape discrepancy unresolved. A separate, unpublished operator study on this card looked at one candidate mechanism: CUDA `MUL_MAT` output that depends on how many columns are batched together. It also tested upstream [PR #32081](https://github.com/ggml-org/llama.cpp/pull/32081), which adds a batch-invariant mode that forces MMQ. This supplement relies on four points from that study:

- **Default dispatch on Turing** (master `22b99122e`):
  - N ≤ 8 runs on MMVQ. Its warp count changes with N: K-quants change at N = 2–4, the other tested types at N = 5–8.
  - N ≥ 9 runs on MMQ.
  - Either change makes batched output differ bitwise from serial output. It is deterministic, not random.
- **Diagnostic modes:** `GGML_CUDA_MMVQ_INV` is a local debug switch, not a proposal.
  - Mode 1 pins the MMVQ warp count to its N = 1 value.
  - Mode 2 also splits the columns into MMVQ calls of at most 8.
  - Mode 2 was bitwise batch-invariant on every tested shape.
- **PR #32081 at `3f25ea199`:** its invariant mode was also bitwise batch-invariant on the tested grids.
- **Cost on the default sm_75 build:** for N ≥ 16, chunked MMVQ took 0.68–0.94× the time of the PR's forced MMQ.

On GTX 16xx cards, llama.cpp logs that Turing GPUs without tensor cores should use the "Pascal code". This supplement tests whether doing so changes that cost comparison.

## Method

CUDA 13 can't build llama.cpp's suggested `61-virtual` architecture: `nvcc fatal: Unsupported gpu architecture 'compute_61'`. Upstream CMake already omits it for CUDA ≥ 13. Instead, [`diag-no-turing-mma.patch`](../../patches/v1.2/diag-no-turing-mma.patch) adds two hunks to `common.cuh`, guarded by `-DGGML_CUDA_DIAG_NO_TURING_MMA`:

- On the host, `turing_mma_available()` returns false.
- On the device, `TURING_MMA_AVAILABLE` stays undefined.

Without the macro, the code is unchanged. Builds keep the same commits, toolkit and `CMAKE_CUDA_ARCHITECTURES=75`. This isolates the MMQ tile path. It is not the literal upstream recipe, which would also change other checks such as `fast_fp16_available`.

Patches, each against a public commit:

| Patch | Base | Contents |
|---|---|---|
| [`fixed-warp-dp4a-vs-master.patch`](../../patches/v1.2/fixed-warp-dp4a-vs-master.patch) | master `22b99122e` | test-backend-ops batch mode, local `SM75_*` shape/seed/hash/perf options, the `GGML_CUDA_MMVQ_INV` diagnostic, the switch |
| [`pr32081-dp4a-vs-pr-head.patch`](../../patches/v1.2/pr32081-dp4a-vs-pr-head.patch) | PR head `3f25ea199` | local `SM75_*` options, the switch (timed PR builds) |
| [`pr32081-dp4a-pool-vs-pr-head.patch`](../../patches/v1.2/pr32081-dp4a-pool-vs-pr-head.patch) | PR head `3f25ea199` | as above, plus five test-harness context pool enlargements for large-N tests |

Arms:

| Arm | Build | Code path | Dispatch |
|---|---|---|---|
| `s75m0` | sm_75, no switch | Turing MMA | default (mode 0) |
| `s75m2` | sm_75, no switch | Turing MMA | chunked MMVQ (mode 2) |
| `s75pron` | sm_75, no switch | Turing MMA | PR invariant mode |
| `fm0` | switch on, FORCE_MMQ on | dp4a | default (mode 0) |
| `fm2` | switch on, FORCE_MMQ on | dp4a | chunked MMVQ (mode 2) |
| `fpron` | switch on, FORCE_MMQ on | dp4a | PR invariant mode |
| `nm0` | switch on, FORCE_MMQ off | dp4a | default (mode 0) |

The `s75*` arms are the earlier builds, rerun in the same session.

## Checking that the switch worked

| Check | sm_75 build | switched build |
|---|---|---|
| IMMA instructions in the 2,566 MMQ kernels (cuobjdump SASS) | 290,683 | 0 |
| IDP.4A instructions in MMQ kernels | 0 | 650,496 |
| Kernels launched (nsys) | `mul_mat_vec_q` N ≤ 8, `mul_mat_q` N > 8 | same; without FORCE_MMQ, N ≥ 64 goes to cuBLAS `volta_h884gemm_*` |
| Accuracy vs CPU, `test-backend-ops test -o MUL_MAT`, 5 types | 0 failures | 0 failures (default and PR builds) |

Data: [`static-check.csv`](../../results/v1.2/static-check.csv), [`static-check-dp4a.csv`](../../results/v1.2/static-check-dp4a.csv) and [`accuracy-summary.txt`](accuracy-summary.txt). In that summary, the `ok` count includes non-case lines and `fail=0` is the result.

## Batch invariance on the switched build

The grids match the earlier study:

- Types: IQ3_S, IQ4_XS, Q4_K, Q5_K, Q8_0.
- M ∈ {256, 1024} and K ∈ {1024, 4096, 5120}.
- Small grid: N = 1–10. Large grid: N ∈ {9–16, 24, 32, 48, 64, 128, 256, 512}.
- Each grid ran with one seeded and one random initialization.

Results are in [`correctness.csv`](../../results/v1.2/correctness.csv).

- **Chunked MMVQ (`fm2`):** 0 failures on both grids (300 + 300 and 450 + 450 cases).
- **Modes 0 and 1:** fail, as on sm_75.
- **PR invariant mode (`fpron`):** 0 failures on the small grid. On the large grid, the timed binary's harness aborts after 12 passing cases per seed with `ggml_new_object: not enough space in the context's memory pool (needed 177184, available 176816)`. This is a test-harness limit, not a numerical failure.
- **Follow-up, 23:07–23:36:** the pool-enlarged harness was built twice:
  - `pool-d4f-pr`: switch on, FORCE_MMQ on.
  - `pool-s75-pr`: no switch, FORCE_MMQ off.

  On the large grid, each scored 900/900 OK with 0 failures. So the PR's invariant mode was batch-invariant over the tested large-N grid on both the MMA and dp4a paths. These are rebuilt test binaries that differ only in the harness; they aren't the binaries that were timed. Log: [`pr-pool-summary.txt`](pr-pool-summary.txt).

## Performance

Setup: M = K = 4096, `test-backend-ops perf -o MUL_MAT`, 10 measured rounds with rotated arm order, 70/70 runs accepted. A lower ratio means the numerator is faster. Ranges cover the 5 types.

| Ratio | Meaning | N ≤ 8 | N ≥ 16 |
|---|---|---|---|
| fm2 / s75m2 | control: chunked MMVQ has no MMA path | 0.998–1.010 | 1.000–1.002 |
| fpron / s75pron | switch effect on PR forced MMQ | 0.50–0.78 | 0.48–0.83 |
| fm0 / s75m0 | switch effect on default dispatch | 0.99–1.01 (MMVQ) | 0.18–0.55 |
| s75pron / s75m2 | PR MMQ vs chunked, sm_75 | 1.45–5.44 | 1.06–1.47 |
| fpron / fm2 | PR MMQ vs chunked, switched | 0.89–3.64 | 0.63–1.11 |
| fm2 / fm0 | chunked vs default, switched | 0.96–1.08 | 1.66–6.20 |
| nm0 / fm0 | FORCE_MMQ off (cuBLAS at N ≥ 64) vs on | about 1.0 | up to 22.4 |

Selected medians (µs):

| type | N | sm_75 default | sm_75 chunked | sm_75 PR | dp4a default | dp4a chunked | dp4a PR | dp4a no FORCE_MMQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q4_K | 1 | 66.7 | 66.8 | 363.2 | 66.9 | 67.1 | 182.2 | 66.7 |
| q4_K | 16 | 495.9 | 410.5 | 602.2 | 200.2 | 411.0 | 296.6 | 199.4 |
| q4_K | 512 | 14824.9 | 13135.8 | 17093.7 | 3582.4 | 13153.3 | 8288.5 | 51290.1 |
| iq3_s | 512 | 15007.7 | 18744.4 | 19907.2 | 3028.5 | 18763.0 | 16384.3 | 51471.6 |
| q8_0 | 512 | 14895.6 | 14306.6 | 19010.3 | 2870.2 | 14324.4 | 13421.1 | 51196.0 |

Data: [`medians.csv`](../../results/v1.2/medians.csv), [`ratios.csv`](../../results/v1.2/ratios.csv), per-observation [`rounds.csv`](../../results/v1.2/rounds.csv) and per-run clocks [`run-clocks.csv`](../../results/v1.2/run-clocks.csv).

## Findings

1. **The code path changes the trade-off, by a lot.** At N ≥ 16 with default dispatch, dp4a MMQ is 1.8–5.5× faster than Turing MMA MMQ on this card. The PR's forced MMQ gets 17–52% faster at every N.
2. **For N ≥ 16, PR MMQ moves from slower than chunked MMVQ to roughly even.** It was 6–47% slower on sm_75. On dp4a it ranges from 37% faster to 11% slower. At N ≤ 8 the PR arm is still up to 3.6× slower than chunked (q4_K at N = 8 is the exception, at 0.89×).
3. **Chunked MMVQ falls well behind default MMQ at N ≥ 16 on dp4a,** at 1.7–6.2× slower. What it still offers is batch invariance, not speed.
4. **The control holds.** Chunked MMVQ, which has no MMA path, stayed within 1%, so session conditions didn't bias the comparison.
5. **FORCE_MMQ is essential.** Without it, N ≥ 64 goes to cuBLAS FP16 GEMM, which is 14–22× slower than dp4a MMQ here.
6. **The PR's invariant MMQ is much slower than default MMQ on the same dp4a code.** For example, q4_K at N = 512 takes 8288 µs vs 3582 µs. The cause wasn't examined.

Interpretation: on TU117, code that issues Turing `mma.sync` (IMMA/HMMA) is much slower than dp4a code. That is consistent with the MMA path's cost explaining MMQ's weakness on this card, though it doesn't prove it. It says nothing about Turing GPUs with tensor cores, such as the T4 or RTX 20xx.

## Measurement quality

- **Acceptance guard:** every non-idle 1 s sample had to show a 6120 MHz memory clock. A continuous 1 s sampler also checked for compiler activity.
  - 70/70 measured runs were accepted, 4 of them after a first-try rejection.
  - The sampler recorded 0 busy samples.
- **Throttling:** from about 20:10 to 21:00, the GPU dropped to P3 (5000 MHz memory; flags 0x24 = SW thermal slowdown + SW power cap). Only warm-ups ran then. All were rejected and are excluded by design.
- **Clocks:**
  - Median SM clock was 1785 MHz in every arm, and every arm had brief dips to 1380 MHz.
  - Peak temperature was 85 °C.
  - A strict filter (no sample below 1785 MHz) keeps only 12/70 runs, too few for per-cell medians. The near-1.00 control ratio is the stronger evidence against clock bias.
- **Worst within-cell spread**, (max − min) / median: 6.16% (switched PR, iq3_s, N = 1).

## Reproduction

The scripts are in [`scripts/v1.2/`](../../scripts/v1.2/), with paths under `${SM75_ROOT:-$HOME/research/sm75-batch-invariance}`. They expect:

- a bare llama.cpp clone at `upstream.git` with PR #32081 fetched
- worktrees at the commits above

`setup_dp4a.sh` rebuilds the worktrees from the original local diffs. Those aren't published. Applying the published patches to the public commits gives the same trees; [`worktree-provenance.txt`](worktree-provenance.txt) records the original worktree heads and diff hashes. The order is:

1. `build_p61.sh`
2. `static_check.sh` and `dispatch_nsys.sh`
3. `run_corr_p61.sh` and `run_perf_p61.sh` (driven by `run_all.sh` / `launch.sh`)
4. `analyze.py` and `post.sh`
5. `acc.sh`
6. `pr_pool.sh`

## Limits

- One device, one session.
- A diagnostic patch, not a production change and not the literal upstream recipe.
- Synthetic `MUL_MAT` only.
- The accuracy check covers test-backend-ops' standard `MUL_MAT` cases for these 5 types, not every timed shape.
- PR large-N correctness comes from rebuilt binaries with the enlarged harness.
- It hasn't been tested whether this mechanism caused the Qwen3.5 batch-shape discrepancy in the main study.
- It's also untested whether the Pascal-path result changes with a real CUDA 12.x `61-virtual` build.

## Cost

Estimated about 2.5 h; the main pipeline took 3 h 31 min. Two things drove the overrun:

- About 50 min went to guarded warm-up retries during thermal throttling. Warm-ups should have been single and unguarded.
- 30 of the 70 timed runs were same-session reruns of earlier arms.

The PR follow-up took 29 min against an estimate of about 25.

See also [`cuda13-pascal-hint.md`](cuda13-pascal-hint.md), a note on the GTX 16xx build hint. It has not been posted upstream.
