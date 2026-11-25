# CUDA: the GTX 16xx "use the Pascal code" hint can't be built with CUDA 13, and how much it helps

**Status:** note for maintainers and users. It has not been posted to llama.cpp. Supporting data: [FINDINGS.md](FINDINGS.md).
**Tested at:** master `22b99122e` (2026-11-24)
**Hardware:** GTX 1650 (TU117, CC 7.5, no tensor cores), WSL2, driver 592.00, CUDA 13.2, GCC 15.2

## Summary

On GTX 16xx GPUs, `ggml-cuda.cu` logs:

> Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.

1. **CUDA 13.x can't build this.** CUDA 13 removed compute_61, so nvcc fails with `nvcc fatal: Unsupported gpu architecture 'compute_61'`. I checked this with 13.1 and 13.2. `ggml/src/ggml-cuda/CMakeLists.txt` already leaves out `61-virtual` for `CUDAToolkit_VERSION VERSION_LESS "13"`, but the runtime hint doesn't account for the toolkit version. A user on CUDA 13 has no documented way to get the faster path.
2. **The faster path is worth getting.** I made it reachable on an sm_75 build with a diagnostic, macro-guarded patch (details below). On a GTX 1650, MMQ through the dp4a path is **1.8–5.5× faster** than through the Turing MMA path at N ≥ 16 on plain MUL_MAT.

## How the Pascal path was reached on CUDA 13 (diagnostic only)

The patch is two hunks in `common.cuh`, guarded by `-DGGML_CUDA_DIAG_NO_TURING_MMA`:

- `TURING_MMA_AVAILABLE` stays undefined on the device.
- `turing_mma_available(cc)` returns `false` on the host.

Everything else is unchanged: same commit, same toolkit, `CMAKE_CUDA_ARCHITECTURES=75`, `GGML_CUDA_FORCE_MMQ=ON`.

Checks that the patch did what it claims:

- **SASS** (cuobjdump, 2,566 MMQ kernels):
  - Normal sm_75 build: 290,683 IMMA and 0 IDP.4A.
  - Patched build: 0 IMMA and 650,496 IDP.4A.
- **Kernels launched** (nsys): `mul_mat_vec_q` for N ≤ 8 and `mul_mat_q` above that, as expected.
- **Accuracy:** `test-backend-ops test -o MUL_MAT` for q4_K, q5_K, q8_0, iq3_s and iq4_xs gives 0 failures.

This is **not** the literal recipe from the hint. A real `61-virtual` build would also change other availability checks, for example `fast_fp16_available`.

## Numbers

Setup: `test-backend-ops perf -o MUL_MAT`, M = K = 4096. Each figure is the median of 10 rounds with arm order rotated. Memory clock was checked on every sample, and runs were rejected if it dropped below 6120 MHz.

Times in µs:

| type | N | sm_75 default (MMA) | dp4a switch + FORCE_MMQ | dp4a switch, no FORCE_MMQ |
|---|---|---|---|---|
| q4_K | 16 | 495.9 | 200.2 | 199.4 |
| q4_K | 512 | 14824.9 | 3582.4 | 51290.1 |
| iq3_s | 512 | 15007.7 | 3028.5 | 51471.6 |
| q8_0 | 512 | 14895.6 | 2870.2 | 51196.0 |

- The patched/default ratio at N ≥ 16 was 0.18–0.55 across the 5 types. At N ≤ 8 it was 0.99–1.01, as expected, since MMVQ handles those sizes.
- `FORCE_MMQ` is essential. Without it, N ≥ 64 goes to cuBLAS FP16 GEMM (`volta_h884gemm_*`), which is up to 22× slower than dp4a MMQ on this card.
- Control: MMVQ, which has no MMA path, stayed within 1% between builds in the same session.

## Prior upstream decision

[PR #15468](https://github.com/ggml-org/llama.cpp/pull/15468) proposed an option to disable MMA on Turing. It was closed without merging. The maintainer's reasoning:

- Existing build options already give the same result.
- Compiling both dp4a and MMA variants for CC 7.5 isn't worth the maintenance cost, given how few GTX 16xx GPUs are in use.

The runtime warning that names the affected devices was the agreed alternative. This note doesn't reopen that decision. A runtime dp4a/MMA selection in sm_75 builds would mean the dual variants that were declined, so it isn't proposed here.

## Suggested fix (for maintainers to judge)

PR #15468 assumed that "existing build options" give users the faster path. With CUDA 13 that is no longer true: the suggested `61-virtual` target can't be compiled. The data above quantify what those users miss.

The narrow fix is to make the hint depend on the CUDA version:

- With CUDA ≥ 13, say that the Pascal path needs a CUDA 12.x toolkit, or drop that part of the hint.
- Fix the flag spelling while there. The message prints `DGGML_CUDA_FORCE_MMQ` without its leading `-`.

I haven't run the literal CUDA 12.x `61-virtual` build on this machine. The speedups above come from the source-level switch.

## Limits

- One GPU in one session, measured only with synthetic MUL_MAT. I make no claims about end-to-end inference speed.
- These results don't cover Turing GPUs with tensor cores (T4, RTX 20xx).
- I didn't run the literal CUDA 12.x `61-virtual` build.

Full CSVs, scripts and logs: [results/v1.2](../../results/v1.2/), [scripts/v1.2](../../scripts/v1.2/), [patches/v1.2](../../patches/v1.2/).
