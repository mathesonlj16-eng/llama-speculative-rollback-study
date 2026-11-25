# v1.3.0: PR #32081 invariant mode does not need its tile-width pin on sm_75 (tested types)

Run 2026-11-25, 11:36–12:08 local, on one GTX 1650 (TU117, CC 7.5), WSL2, driver 592.00, CUDA 13.2. Follows [v1.2.0](../v1.2/FINDINGS.md). The prediction and the code reasoning behind it are in [code-reading.md](code-reading.md). Raw logs are archived outside this repository.

## Setup

- **Source:** the `pr-pool` worktree, which is PR head `3f25ea199` plus the local SM75 harness, the macro-guarded dp4a switch and the context-pool enlargement.
- **Change:** a host-only environment switch in `mmq.cu` ([`mmq-tile-width-switch.patch`](../../patches/v1.3/mmq-tile-width-switch.patch), applied on top of [`pr32081-dp4a-pool-vs-pr-head.patch`](../../patches/v1.2/pr32081-dp4a-pool-vs-pr-head.patch)), so only `mmq.cu` recompiled (22–26 s per build). Settings in invariant mode:

  | `SM75_INV_NCOLS` | Behaviour |
  |---|---|
  | unset | PR behaviour: J pinned to the smallest width |
  | `0` | J follows the batch, as in default dispatch |
  | `64` | J optimised for 64 columns |

- **Builds:**
  - `pool-d4f-pr`: dp4a path, FORCE_MMQ on
  - `pool-s75-pr`: Turing MMA path, stock sm_75

Tile widths actually used (logged by `SM75_INV_LOG`, the same on both builds):

| Setting | N=1 | N=9 | N=16 | N=64 | N=512 |
|---|---|---|---|---|---|
| unset (PR) | J8 | J8 | J8 | J8 | J8 |
| `0` (free) | J8 | J16 | J16 | J64 | J128 |
| `64` | J64 | J64 | J64 | J64 | J64 |

## Correctness

The grid was the same as before:
- 5 types: IQ3_S, IQ4_XS, Q4_K, Q5_K, Q8_0
- M ∈ {256, 1024}, K ∈ {1024, 4096, 5120}
- small grid N = 1–10; large grid N ∈ {9–16, 24, 32, 48, 64, 128, 256, 512}

Results ([`correctness.csv`](../../results/v1.3/correctness.csv), [`hash-compare.csv`](../../results/v1.3/hash-compare.csv), [`run-log.txt`](run-log.txt), [`analysis.txt`](analysis.txt)):

- **0 failures in all 16 runs:** 2 builds × 3 settings × 2 grids with seeded data, plus free J with random data on both grids and both builds.
- **Free J and J=64 give the same bits as the PR's J=8**, in both the batched and the serial half, for 750/750 seeded cases on each build.
- **Today's pinned binary reproduces yesterday's pool run**, 450/450 batch hashes equal, so the rebuild didn't change the default behaviour.
- **dp4a and MMA outputs differ from each other** (0/450 equal), as expected. Each path is invariant with respect to itself.

On these shapes, once Stream-K is off, the tile width doesn't affect the result bits on either path, as predicted.

## Timing, dp4a build only ([`perf-medians.csv`](../../results/v1.3/perf-medians.csv))

Setup: M = K = 4096, 3 rotated rounds plus 1 unguarded warm-up, clocks logged.
- 9/9 runs had a 6120 MHz memory clock on every non-idle sample.
- SM clock dipped to 1380 MHz in some runs.
- The worst within-cell spread was 6.4% (iq3_s N=1); the others were ≤ 1.5%.

Ratios, ranges over the 5 types (lower ratio = numerator faster):

| N | PR pinned / default | free J / default | free J / PR pinned |
|---|---|---|---|
| 1 | 2.24–3.72 | 2.22–3.71 | 0.98–1.00 |
| 16 | 1.48–1.89 | 1.05–1.26 | 0.62–0.71 |
| 128 | 2.02–5.34 | 1.04–1.17 | 0.22–0.52 |
| 512 | 2.31–5.41 | 0.99–1.00 | 0.19–0.43 |

Example: q4_K at N=512 takes 3585 µs by default, 8288 µs with the PR pin, and 3574 µs with free J.

## Findings

1. **The PR's tile-width pin (`ncols_opt = 1`) isn't needed for bitwise batch invariance** on sm_75 for these 5 types. Disabling Stream-K (one tile per block) is enough. This holds on both the dp4a and the Turing MMA code paths.
2. **On dp4a, dropping the pin removes almost all of the large-N cost.**
   - At N=512, invariant mode matches default MMQ (0.99–1.00×), down from 2.3–5.4×.
   - What remains at N=16–128 (1.04–1.26×) is the cost of losing Stream-K, or of non-tile-aligned N.
3. **The N=1 decode cost is unchanged** (2.2–3.7×). It comes from using MMQ instead of MMVQ, not from tile width, and J is already 8 at N=1.
4. **This also settles local question 1** (pinned width vs Stream-K overhead) on dp4a: the large-N penalty was almost entirely tile width.

## Limits

- One GPU, one session. Timing is a short 3-round pass.
- Only 5 types and plain MUL_MAT were tested. MUL_MAT_ID (MoE), other quant types, and the Blackwell and AMD configs are not covered; those configs can change I, nthreads or K step with J.
- MMA-path timing wasn't measured, because this card has no tensor cores.
- The bitwise result on the MMA path is expected to hold on a T4, since it is the same sm_75 code with exact integer MMA. A T4 is still needed to measure the free-J cost there.

## Cost

I estimated 45–60 min. The run itself took 32 min: 11:36–12:08, including a 6-minute rerun of the timing pass after an `env` argument-order bug. A failed patch pattern added another minute beforehand.

## Reproduction

The scripts are in [`scripts/v1.3/`](../../scripts/v1.3/). They use the same `SM75_ROOT` layout as v1.2.0 and need the `pr-pool` worktree and the `pool-*` builds.

1. `jfree.sh` applies the switch with `edit_mmq.py`, rebuilds both builds, logs the tile widths and runs the grids. Its first timing pass had a bug in the `env` argument order, so that pass is discarded.
2. `jfree2.sh` logs the tile widths again and runs the timing pass that the results use.
3. `jfree_an.py` writes the hash comparison and the timing medians.
