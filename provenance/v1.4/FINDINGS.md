# v1.4.0: the study's CUDA shape discrepancy under PR #32081 batch-invariant mode (2026-11-25)

**Question.** In v1.0.0, prefill grouping changed CUDA logits with one sequence and no draft or rollback. The responsible kernels were left unresolved. Does the batch-invariant mode of [PR #32081](https://github.com/ggml-org/llama.cpp/pull/32081) remove the effect?

**Answer for the tested fixtures.**
- Only partly.
- The mode removes the matmul dispatch dependence that v1.2.0 identified. That is enough for the original 11-token fixture.
- On longer sequences a second source remains. The flash-attention KV split follows the KV length, which is padded to 256 and depends on the batch.
- With that length held fixed, every grouping tested is bit-identical.

**Where things are:**
- Results: [`results/v1.4/`](../../results/v1.4/)
- Patches: [`patches/v1.4/`](../../patches/v1.4/)
- Runner: [`scripts/v1.4/run.sh`](../../scripts/v1.4/run.sh)
- Environment, hashes and timelines: [`run-log.txt`](run-log.txt)
- Raw logit dumps (about 21 GB) and per-position CSVs are archived outside this repository.

## Setup

- **Hardware and software.** GTX 1650 (TU117, sm_75), CUDA 13.2.86, WSL2. Target is the study's Qwen3.5-2B Q8_0 (`e8bf47d4…`, checked). Every run uses `n_gpu_layers` 99 and the default `flash_attn` auto.
- **Code.** PR head `3f25ea199a36a9e83480b6dac0941bb82c96c5b9`, on merge-base `40fe9164f4` (2026-11-24), in a clean worktree. Qwen3.5 (`LLM_ARCH_QWEN35`) and every replay-tool API used here are present on that base. The study pin `787a5612` is not.
- **Flag.** The PR adds `llama_context_params.batch_invariant`, and `--batch-invariant` / `LLAMA_ARG_BATCH_INVARIANT` for common-based tools. The replay tools set context params directly. [`replay-tools-batch-invariant-toggle.patch`](../../patches/v1.4/replay-tools-batch-invariant-toggle.patch) adds an env toggle `BI`. It also adds `N_CTX` to `logit-dump`, which is used only in A5 (default 2048, unchanged).
- **Mode check.** Every flag-on context logged `batch invariance enabled for CUDA0`.
- **Fair comparison.** Flag off vs flag on in the same binary. Each grouping is compared with the one-token-per-call `singles` run of the same arm.

## A0/A1: the original fixture (E1)

E1 is [`invariance.cpp`](../../scripts/archive/fixed-token-replay/invariance.cpp): 11 fixed tokens, one sequence, 17 decode plans, at recurrent capacity 0 and 3. That makes 34 contexts, and each run compares last-position logits with `singles`.

| Arm | Plans not bit-identical to singles | max abs Δ | Top token changes |
|---|---|---|---|
| A0, flag off | 24/34 (`whole`, `split1–10`, `whole-repeat`) | 0.277 | split5, split6, split8 |
| A1, flag on | **0/34** | **0** | none |

- **A0 matches the archive exactly.** Its CSV is byte-identical to the archived [`results/e1-cuda.csv`](../../results/e1-cuda.csv) apart from the backend label. The newer base reproduces the v1.0.0 effect exactly.
- **A1 removes it completely.** [Off](../../results/v1.4/e1-flag-off.csv) · [On](../../results/v1.4/e1-flag-on.csv)

## A3: longer teacher-forced sequences

- **Tool.** [`logit-dump`](../../scripts/v1.1/logit-dump.cpp) on [`ngram-token-fixtures.tsv`](../../results/ngram-token-fixtures.tsv).
- **Fixture.** 19 sequences with 6–348 prompt tokens and 48 or 96 generated tokens, for 1440 predicted positions.
- **Shapes.**
  - `whole`: prompt plus generated tokens in one decode.
  - `c4` / `c8`: the prompt, then chunks of 4 or 8.
  - `pair4`: two sequences per batch.

| Flag | vs singles: positions with nonzero Δ (whole / c4 / c8 / pair4) | max abs Δ | Top token changes |
|---|---|---|---|
| off | 1440 / 1440 / 1440 / 1440 | 7.91 | 8 / 16 / 15 / 12 |
| on | **768 / 603 / 607 / 1051** | 2.88 | **0 / 0 / 0 / 2** |

**The flag-on residual follows a 256-token boundary** ([per-sequence table](../../results/v1.4/logit-cmp-by-sequence.csv)):
- Sequences whose whole span stays ≤ 256 tokens are bit-identical in `whole`, `c4` and `c8`. These are all eight short sequences, plus three 139-token prompts at 139 + 96 = 235.
- `whole` differs from index 0 whenever prompt + 95 > 256.
- In `c8`:
  - seq9 (prompt 186) is identical up to generated index 65, where an 8-token chunk covers positions 250–257. From there the difference carries forward through the KV cache.
  - seq12 (prompt 216) stays identical. Its chunk boundary falls exactly on 256, so no batch straddles it.
  - Long prompts (274–348) differ from index 0, because the prompt batch itself goes past 256.

**Mechanism, from the code at the PR head:**
- `llama_kv_cache::get_n_kv` pads the KV length to a multiple of 256, from the furthest cell the ubatch uses (`n_pad = 256`).
- In the flash-attention launch (`fattn-common.cuh`), the PR fixes `ntiles_dst_pb` in invariant mode. But the `parallel_blocks` search still runs up to `ntiles_KV`, which comes from the padded `K->ne[1]`.
- When a batch pushes the padded length from 256 to 512, the KV split changes, and so does the order partial results are combined in.
- This is a gap separate from the MUL_MAT paths the PR covers.

## A4: KV length pinned

- **Patch.** [`kv-n_kv-pinned-vs-pr-head.patch`](../../patches/v1.4/kv-n_kv-pinned-vs-pr-head.patch) makes both `n_kv` paddings in `llama-kv-cache.cpp` return the full cache size. It is diagnostic only.
- **Build.** Only that object was recompiled and libllama relinked. The ggml and ggml-cuda libraries are the same files as in A0–A3.
- **Run.** Flag on, same fixture and shapes.

| Flag on, n_kv pinned: singles vs | Positions with nonzero Δ | max abs Δ | Top token changes |
|---|---|---|---|
| whole | **0** | 0 | 0 |
| c4 | **0** | 0 | 0 |
| c8 | **0** | 0 | 0 |
| pair4 | 209 (seq11/13/15/16, first at position 385) | 0.676 | 0 |
| A3 flag-on singles vs A4 flag-on singles | 1056 | 1.22 | 2 |

- **Single-sequence shapes become fully invariant**, including batches longer than 256 tokens and chunks that straddle position 256.
- **The last row is the patch's own effect.** A different, fixed KV split gives different but consistent numbers.

## A5: pair4 control

- **Why `pair4` differs.** `logit-dump` builds the `pair4` context with `n_seq_max = 2` and a non-unified KV cache. That gives 1024 cells per stream against 2048 for `singles`; the logs show `n_ctx_seq = 1024` vs 2048. So even with the length pinned, the two contexts give flash attention different KV lengths.
- **Control.** `singles` at `N_CTX=1024`, which also has 1024 cells.

| Comparison (flag on, n_kv pinned) | Positions with nonzero Δ | max abs Δ |
|---|---|---|
| singles @1024 cells vs pair4 (1024 cells per stream) | **0** | 0 |
| singles @2048 vs singles @1024 | 209 | 0.676 |

- **Result.** The A4 `pair4` residual is exactly the cache-size effect. Two sequences batched together are invariant once the cache size matches.

## What this resolves and what it doesn't

**Resolved for these fixtures on sm_75.** The v1.0.0 CUDA shape discrepancy comes from two sources that depend on batch shape:
1. The MUL_MAT dispatch. The MMVQ warp count changes for N ≤ 8, and dispatch switches from MMVQ to MMQ at N = 9 (v1.2.0). PR #32081's mode removes this.
2. The flash-attention KV split, which follows the padded KV length. PR #32081's mode does not remove this.

With both handled, every grouping tested is bit-identical.

**Coverage notes from reading the code, not tested:**
- **Recurrent layers on sm_75.** The PR doesn't touch `GATED_DELTA_NET` or `SSM_CONV`. On sm_75 that is not a gap: `select_gdn_mma_path` returns the sequential `ar` kernel unless the GPU is GA10x, Ada, Blackwell or GB10.
- **Recurrent layers on Ampere and newer.** The chunked MMA path turns on at `n_tokens ≥ 64`, a batch-shape switch the mode doesn't cover.
- **Remaining MUL_MAT fallbacks.** The early cuBLAS return, and non-quantized weights where MMVF can't be used, also stay N-dependent. Non-quantized MUL_MAT_ID was already covered in v1.3.2.

**Not tested:**
- Speculative decoding or rollback under the mode.
- Server-level greedy outputs; everything here is teacher-forced logits.
- Multi-sequence runs where the per-stream cache sizes differ from the single-sequence reference by design.
- Other GPUs, quantizations and models.
- Timing: these were correctness-only runs. The flag-on `singles` dump took 186 s against 59 s flag off, from one run each.

**Related prior work (added in v1.5.3).** [Issue #25618](https://github.com/ggml-org/llama.cpp/issues/25618) (opened 2026-07-13) reports a related symptom: greedy speculative output diverging on quantized targets. makoni's 2026-11-03 patch in that issue already makes the FA split independent of `ntiles_KV`. [PR #16016](https://github.com/ggml-org/llama.cpp/pull/16016) proposed deterministic attention in 2025. v1.4.0 was written without these references.

**Possible fixes (not implemented).** In invariant mode, either:
- choose the FA `parallel_blocks` independently of `ntiles_KV`, or
- pin `n_kv` to the cache size, at the cost of attending over masked cells.

Nothing has been posted upstream.

## Cost

| Step | Estimate | Actual |
|---|---|---|
| Free checks (models, fixtures, PR reading) | ~15 min | ~15 min |
| Build, libllama + ggml-cuda (sm_75, ccache cold: 0/223 hits) | 12–14 min | 861 s |
| A0 + A1 | ~2 min | 15 s |
| A3 (10 dumps, comparisons) | ~10 min | ~9 min |
| A4 (relink + 5 dumps) | 8–10 min | ~5.5 min, plus a ~1 min first attempt that failed before building (wrong unity-file assumption) |
| A5 | ~4 min | ~3.2 min |

No idle time. No paid compute.
