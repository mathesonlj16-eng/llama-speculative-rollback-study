# v1.3.2: PR #32081 invariant mode is not batch-invariant for non-quantized MUL_MAT_ID (traced 2026-11-25)

Per-case results: [`moe-gap.csv`](../../results/v1.3/moe-gap.csv). Candidate fixes: [`moe-invariance-candidate-fixes.patch`](../../patches/v1.3/moe-invariance-candidate-fixes.patch), env-guarded diagnostics on top of [the v1.3 harness](../../patches/v1.3/test-backend-ops-moe-vs-pr-head.patch). Scripts: `scripts/v1.3/moeid3.sh` (bug), `moeid4.sh` (fix A), `moeid5.sh` (fix A+B). Raw logs are archived outside this repository.

## Bug

On the **stock sm_75 build** (no diagnostic switch), PR head `3f25ea199`, `test-backend-ops batch -o MUL_MAT_ID`, seeded routing (`SM75_SEED=1234`). Cases: 8 or 4 experts, 2 used, plain and GLU, N ∈ {16, 128, 256}.

| type | result |
|---|---|
| BF16 | **9/9 FAIL**, at every N. Max diff 1e-5 to 2e-3 |
| F16 | 7 OK, **2 FAIL**: m=2048, N=256 (non-GLU, both expert counts). Up to 1,048,552/1,048,576 elements differ, max diff 0.25 |
| F32 | 9/9 OK |
| Q8_0 | 9/9 OK |

The PR's own batch suite misses this. Its MUL_MAT_ID cases are f16/q8_0/q4_K/q6_K with N ≤ 8 tokens, and they have no BF16.

## Cause (`ggml_cuda_mul_mat_id`, ggml-cuda.cu)

1. The PR's invariant branch in `ggml_cuda_mul_mat_id` covers only MMQ, i.e. quantized types. F16 and BF16 fall through to normal dispatch.
2. On NVIDIA, F16/BF16 MoE then uses MMF only if `ggml_cuda_should_use_mmf(..., mul_mat_id=true)` allows it. That requires:
   - BF16: Ampere MMA.
   - F16: Volta or Turing MMA.
   - Token limits: at most 128 tokens if src0 has more than 1024 rows, at most 512 otherwise.
   Otherwise it takes the sorted per-expert fallback.
3. In the fallback, `type_src1_sorted` equals src0's type (F16/BF16) when fast FP16 is available. Each expert's `ggml_cuda_mul_mat` therefore gets a non-F32 src1, and its first check sends that straight to `ggml_cuda_mul_mat_cublas`, **before** the batch-invariant branches. cuBLAS with a per-expert column count that depends on the batch isn't invariant.
4. The BF16 failures on Turing: the fallback is always used, and cuBLAS is used every time.
5. The F16 failures at N=256: the serial half (1 token) uses MMF, while the 256-token batch exceeds MMF's limit and takes the fallback. Different kernels give different bits.
6. The earlier F16 failures on the dp4a diagnostic build follow from point 2. Disabling Turing MMA makes MMF unavailable, so F16 always takes the fallback, as BF16 does on Turing.

Expected on other GPUs, from the code and not tested:
- The F16/BF16 token limits in `should_use_mmf` don't depend on the architecture. So Ampere and later should also fail for F16/BF16 MoE with more than 128 tokens (rows > 1024) or more than 512 tokens.
- Pre-Volta GPUs should fail for F16 MoE at any N whenever fast FP16 is available.

## Candidate fixes tested (diagnostic, env-guarded; `moe-fix.diff`)

- **A. In invariant mode, sort src1 to F32 in the fallback** (`type_src1_sorted = F32 when ctx.batch_invariant`). This is a one-condition change. With it:
  - Each expert's `ggml_cuda_mul_mat` sees F32 src1 and takes the PR's existing MMVF-chunk invariant path.
  - BF16 goes from 9/9 FAIL to **9/9 OK**. F16 is unchanged (still 2 FAIL, see B).
  - With the environment variable unset, output is bit-identical to the unpatched build: 18/18 hashes.
- **B. Additionally skip MMF for MUL_MAT_ID in invariant mode**, so serial and batch both use the fallback. This **aborts**: `GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc))` at ggml-cuda.cu:2725. The sync and CUDA-graph decision predicts MMF, and it takes only `dst` and `cc`, with no access to `ctx.batch_invariant`. Fixing F16 at large N therefore needs a design decision. The options I see, none tested:
  - make `needs_sync` aware of invariant mode
  - let MMF accept any token count in invariant mode
  - pick MMF or the fallback independently of the token count

  Each option trades speed differently.
- With A+B, the PR's own suite aborted at the same assert after 156 OK, 0 FAIL, when it reached an F16 MoE case.

## Limits

- One GPU, sm_75. Only synthetic `test-backend-ops` shapes.
- The Ampere and later behaviour is inferred from code. Not tested.
- The speed cost of fix A wasn't measured: it means MMVF chunks instead of cuBLAS for BF16/F16 MoE when invariant mode is on.
