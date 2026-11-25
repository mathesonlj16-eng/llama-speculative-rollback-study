# v1.5.0: a flash-attention fix for the remaining batch-shape gap in PR #32081 (2026-11-25)

**Background.** v1.4.0 found that [PR #32081](https://github.com/ggml-org/llama.cpp/pull/32081)'s batch-invariant mode leaves one source of batch-shape dependence on the GTX 1650. The flash-attention KV split follows the padded KV length. Pinning that length showed the mechanism, but it is not a usable fix.

**Finding.**
- A small change to the flash-attention launch, active only in invariant mode, makes every tested grouping bit-identical without pinning the KV length.
- That includes two sequences per batch with a different per-stream cache size.
- Default mode is bitwise unchanged.
- In invariant mode the change costs under 1%.

**Where things are:**
- Patch: [`fa-kv-split-invariant-vs-pr-head.patch`](../../patches/v1.5/fa-kv-split-invariant-vs-pr-head.patch)
- Results: [`results/v1.5/`](../../results/v1.5/)
- Runner: [`scripts/v1.5/run.sh`](../../scripts/v1.5/run.sh)
- Environment, hashes and timelines: [`run-log.txt`](run-log.txt)

Nothing has been posted upstream.

## The change

The patch changes `launch_fattn` in `ggml/src/ggml-cuda/fattn-common.cuh` at PR head `3f25ea199`, in the non-Stream-K branch that the PR's forced vector kernel uses. When `ctx.batch_invariant` is set:

1. **Search cap.** The `parallel_blocks` search is capped by one wave of blocks (`nsm × max_blocks_per_sm`), not by `ntiles_KV`. `ntiles_KV` is computed from the padded KV length `K->ne[1]`, which changes from 256 to 512 cells when a batch writes past position 256.
2. **Tile count.** The search is sized for one query tile of one sequence, `ntiles_z_gqa × K->ne[2]`. The PR's version is `ntiles_dst / ntiles_x`, which still includes the number of sequences (`Q->ne[3]`).

When the flag is off, both values are the same as before.

**Why a fixed split count is enough:**
- The vector kernel assigns KV tiles to blocks with a stride: block `y` takes tiles `y`, `y + P`, … where `P` is the split count.
- Masked cells add `exp(-inf) = 0`.
- A block with no tiles leaves its partial maximum at `-FLT_MAX/2` and its sum at 0. The combine kernel weights it by `expf(-FLT_MAX/2 - max) = 0`.
- So once `P` is fixed, the summation order over the unmasked positions is the same for every batch shape and KV length.

## Correctness

- **Build.** The patched build is a fresh worktree at `3f25ea199` plus the patch, in `build/pr-bi-fafix`, configured like v1.4.0.
- **Tools.** The tools are the v1.4.0 replay tools.
- **KV length.** There is no `n_kv` pin.

| Check | Differs | max abs Δ | Top token changes |
|---|---|---|---|
| E1 flag on: 34 plans vs singles | **0/34** | 0 | 0 |
| Teacher-forced fixture, flag on: singles vs whole / c4 / c8 | **0 / 0 / 0** of 1440 positions | 0 | 0 |
| Same, pair4 (1024 cells per stream vs 2048 for singles) | **0** of 1440 | 0 | 0 |
| Default mode, E1: patched vs v1.4.0 A0 logit files | 0 of 34 `.f32` files | — | — |
| Default mode, teacher-forced: unpatched vs patched, singles / whole | 0 / 0 of 1440 | 0 | 0 |

- **Every flag-on context** logged `batch invariance enabled for CUDA0`.
- **v1.4.0 needed more for this.** There, single-sequence shapes needed the `n_kv` pin, and pair4 also needed a matching cache size. Here neither is needed.
- **E1 output files are unchanged.** The E1 flag-off and flag-on CSVs are byte-identical to the v1.4.0 A0 and A1 CSVs.
- **Data.** [E1 off](../../results/v1.5/e1-flag-off.csv) · [E1 on](../../results/v1.5/e1-flag-on.csv) · [Per-sequence comparisons](../../results/v1.5/logit-cmp-by-sequence.csv)

## Cost

**Tool.** `llama-bench` can't set `batch_invariant`, so [`bench-bi.cpp`](../../scripts/v1.5/bench-bi.cpp) times the context API directly. On each run it:
- loads the model with n_ctx 4608 and ubatch 512;
- for each depth d ∈ {256, 1024, 4096}, clears memory, prefills d tokens (pp), then runs 64 one-token decodes at that depth (tg);
- does one untimed warm-up at 256 first.

**Configurations.** Four configurations ran interleaved, with 5 repeats each:
- unpatched / default mode
- unpatched / invariant mode
- patched / invariant mode
- patched / default mode

| Depth | Patched ÷ unpatched, invariant (pp / tg) | Invariant ÷ default, unpatched (pp / tg) | Patched ÷ unpatched, default (pp / tg) |
|---|---|---|---|
| 256 | 0.993 / 0.998 | 0.771 / 0.312 | 1.001 / 1.002 |
| 1024 | 0.995 / 0.997 | 0.791 / 0.316 | 1.000 / 1.002 |
| 4096 | 0.996 / 1.002 | 0.787 / 0.323 | 1.000 / 0.999 |

**The patch costs little.**
- Decode is within ±0.3 %, inside the repeat noise.
- Prefill is 0.4–0.7 % slower, about the size of the noise but slower at all three depths.

**Invariant mode itself is the expensive part.**
- Default-mode decode is about 66 tok/s at depth 256; invariant mode drops it to about 21 tok/s.
- Prefill is 21–23 % slower.
- That fits the cost v1.3.0 measured at the operator level: MMQ at N = 1, in place of MMVQ, is 2.2–3.7× slower. The patch has nothing to do with it.

**Measurement quality.**
- The GPU held 1785 MHz on every timed run, at 42–68 °C.
- The first pre-run sample was 300 MHz at idle; the untimed warm-up covers it.
- Repeats ranged by ≤ 0.7 % of the median in every cell but one, which was 2.1 %.
- Data: [timing rows](../../results/v1.5/bench-times.csv), with medians and ratios from [`bench_an.py`](../../scripts/v1.5/bench_an.py).

## Limits

- **Hardware and model.** One GPU (GTX 1650, sm_75) and one model, Qwen3.5-2B Q8_0. It is hybrid: the KV-cache log shows 6 attention layers.
- **Workload.** Timing covers batch size 1 and depths up to 4096. A model with more attention layers, longer contexts or a GPU with more SMs could weigh the split differently. The flash-attention kernel was not timed on its own.
- **Kernels not covered.** The Stream-K tile and MMA flash-attention kernels, and the sinks and sparse paths. In invariant mode the PR forces the vector kernel when it is eligible, and only that path was exercised.
- **Not tested.** Other head sizes, quantized KV caches, other GPUs, and server-level speculative runs.
- **Remaining gaps.** These change nothing about the gaps v1.3.2 and v1.4.0 noted elsewhere: F16/BF16 MUL_MAT_ID, the gated-delta-net MMA switch on Ampere and newer, and the cuBLAS fallbacks.

## Cost of this work

| Step | Estimate | Actual |
|---|---|---|
| Patched build, full `llama` target, ccache cold (0/223) | ~14 min | 842 s |
| Correctness runs (E1, 5 flag-on shapes, default-mode check) | ~9 min | ~9 min |
| Timing tool, plus 20 interleaved runs | ~10 min | ~5 min to write, 8.6 min to run |

- No idle time and no paid compute.
- The session restarted during the build. The detached job kept running, and only a wait-only keep-alive was reattached.
