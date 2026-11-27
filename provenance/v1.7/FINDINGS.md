# v1.7.0: batch-invariant K-quant kernels for 2–8 tokens, sanitizer pass, and a matched layout A/B (2026-11-26)

**Background.** v1.6.0 covered Q8_0 at N = 1..8 and Q4_K/Q6_K at N = 1 ([v1.6.0 findings](../v1.6/FINDINGS.md)). In greedy speculative decoding, the draft model's first batch each round has 2+ tokens: the accepted tail plus the bonus token. For the Q4_K_M draft that batch still went through forced MMQ in invariant mode. It was the one invariance-specific cost left: 18.5 ms against 10.4 ms per round in default mode, at a draft length of k = 1.

**Question.** Can the v1.6.0 construction be extended to Q4_K and Q6_K at N = 2..8, with the same bits as forced MMQ and default-like speed? Does that remove the speculative-decoding cost?

**Findings** (GTX 1650, sm_75, PR #32081 head `3f25ea199a36a9e83480b6dac0941bb82c96c5b9`; Qwen3.5-0.8B Q4_K_M draft, Qwen3.5-2B Q8_0 target):
- **Proven (exact byte compares):**
  - The new kernels give **the same bits as forced MMQ for every column of N = 1..8**: 0 differences in 46,812,672 compared floats, on 13 draft shapes and 7 input families.
  - `test-backend-ops batch` passes 112/112.
  - Every draft and target fixture plan is byte-identical.
  - Speculative tokens and full-logit hashes are unchanged from v1.6.0.
- **Sanitizer:** memcheck, racecheck and synccheck report 0 errors and 0 hazards on every fast kernel (Q4_K, Q6_K and Q8_0, N = 1..8). initcheck reports nothing in the fast kernels. Its only reports are in upstream's forced-MMQ kernel, identical at unmodified PR head, and are benign by source analysis (§5).
- **Measured, per op on the real decode shapes, new kernel time / default time:**
  - Q4_K: 0.73–1.16× at N = 2 and 0.72–0.98× at N = 8;
  - Q6_K: 0.66–1.32× at N = 2..8;
  - forced MMQ on the same shapes: 0.9–5.4×.
- **Measured, speculative decoding in invariant mode at k = 1:** 67.9–70.0 tok/s, 1.03–1.05× plain invariant decoding, up from 52.6–52.8 tok/s (0.79×).
- **The first version was slow at N = 7–8** (Q4_K up to 1.64×). The cause was shared-memory occupancy, not registers. A per-width staging layout fixed it (§3).
- **A matched A/B of the layouts** shows their model-level effect is within run-to-run noise (≤ 2%). Earlier between-session differences of up to 6% were noise (§4).

## Prior art

As in v1.6.0: fixed-order invariant kernels, per-token MMVQ for small batches ([#25618](https://github.com/ggml-org/llama.cpp/issues/25618)) and weight reuse across a few tokens are earlier work. Two were checked against source for this release:
- **Few-token MMQ ([#31533](https://github.com/ggml-org/llama.cpp/pull/31533) / [#32065](https://github.com/ggml-org/llama.cpp/pull/32065), in the pinned source).** `ggml_cuda_mmq_few_min_ncols` returns 0 unless `ampere_mma_available(cc)`, so it has no sm_75 path. It uses `cp.async` staging and splits K across four warps with FP32 partials. That is a different float program from ordinary whole-K MMQ.
- **[#30741](https://github.com/ggml-org/llama.cpp/pull/30741)** (merged, OpenCL/Adreno, "cooperative-k dp4a gemm for q4_0, q4_k and q6_k"). It strides K blocks across subgroups and reduces per-subgroup float accumulators through local memory. That is split float accumulation, and it makes no invariance claim.

Both are prior art for cooperative small-batch quantized kernels. Neither reproduces an existing pinned MMQ's bits.

## 1. Kernels (in the patch: `mmq-bi-nb-kq.cuh`, dispatch in `mmq-bi-n1.cu`)

`bi_nb_q4_K_v3<NW=2, RW=4, NC>` and `bi_nb_q6_K_v3<…>`, NC = 2..8, extend v1.6.0's K-quant `v3` kernels the way v1.6.0 extended Q8_0:
- each chunk of 4 superblocks is staged in shared memory once per row, and its 4-bit or 6-bit values are unpacked once and dotted against all NC activation columns;
- lane `l < 4·NC` then folds (row `l/NC`, column `l%NC`) serially.

**The per-output float program is v1.6.0's, unchanged:**
- Q4_K: `acc = fma(fmul(dmA.x, dsB.x), float(C), acc)` then `acc = fma(dmA.y, dsB.y, acc)`, per 32-value group in order;
- Q6_K: `tmp = fma(Cf, dB, 0)`, three more FFMA, then `acc = fma(tmp, dA, acc)` per 16-value pair.

**Activations** come from the real `quantize_mmq_q8_1_cuda` with `ne1 = N`: block (kb, col) at `kb·N + col`.

**Gate:** the v1.6.0 N = 1 gate for each type (sm_75; K % 512 for the K-quants) plus `ne11 ≤ 8`.

**Switches:**
- `GGML_CUDA_BI_FAST_NB_KQ=0` keeps Q4_K/Q6_K at N ≥ 2 on forced MMQ;
- `GGML_CUDA_BI_NB_Q4K_LAYOUT=f4|compact` forces one Q4_K staging layout, for research A/B only. Unset gives the hybrid of §3.

The Q8_0 path is unchanged from v1.6.0.

## 2. Bit identity, controls and sanitizer

**Operator sweep** ([`scripts/v1.7/bi_nb_kq_lib.cu`](../../scripts/v1.7/bi_nb_kq_lib.cu); [`results/v1.7/kq-nb-sweep-intree-hybrid.csv`](../../results/v1.7/kq-nb-sweep-intree-hybrid.csv)):
- **What it compares:** every column of N = 1..8 through the real `ggml_mul_mat` in invariant mode, bitwise against N = 16, which stays on forced MMQ in every build.
- **Shapes:** 13, namely Q4_K 1024 → {16, 512, 2048, 3584, 4096, 6144}, 2048 → 1024, 3584 → 1024; Q6_K 1024 → {512, 6144, 248320}, 3584 → 1024, 4096 → 4096.
- **Input families:** 7, namely normal, subnormal/FTZ range, 25% zero blocks, overflow to ±Inf, alternating cancellation, single nonzero, and −0.0 activations.
- **Result: 0 bad in 46,812,672 compared floats**, both for the first float4 build and for the final hybrid build ([`…-float4.csv`](../../results/v1.7/kq-nb-sweep-intree-float4.csv)).
- **Coverage of special values:** the references include 12,980 outputs equal to −0.0 and 1,805,678 non-finite outputs.
- **Control:** the same products in default mode (MMVQ) differ from the reference in 315 of 450 rows.
  - Every row of normal, zeros, cancel and single fails, so the test detects a different float program.
  - In 45 of the 64 rows of each of subnormal, large and −0.0, the outputs collapse to ±0 or ±Inf and the control does not separate them. Those families cover special values but are weak separators.

**Model level** ([`results/v1.7/model-checks-hybrid.txt`](../../results/v1.7/model-checks-hybrid.txt)):
- `test-backend-ops batch -o MUL_MAT` passes 112/112, with the new kernels and with `NB_KQ=0`.
- **Draft model, BI on:** 34/34 E1 files byte-identical to `NB_KQ=0`, with 0/34 plans not invariant. The 19-sequence plans singles, whole, c4, c8 and pair4 are byte-identical to `NB_KQ=0` and to singles; c4, c8 and pair4 run 4- and 8-token batches through the new kernels.
- **2B target, BI on:** 34/34 E1 files byte-identical to v1.5.0's `pr-bi-fafix`.

**SASS** ([`results/v1.7/sass-ops-nb-kq.txt`](../../results/v1.7/sass-ops-nb-kq.txt)):
- `IDP.4A.S8.S8`, `I2F`, `FMUL.FTZ` / `FFMA.FTZ` (plus `HMUL2` / `I2F.F16` for Q4_K's half scales), and no IMMA.
- dp4a count scales with NC: Q4_K 32·NC, Q6_K 16·NC.

**Sanitizer** ([`scripts/v1.7/run-sanitizer.sh`](../../scripts/v1.7/run-sanitizer.sh), [`results/v1.7/sanitizer/`](../../results/v1.7/sanitizer/)):
- **Configurations:** Q4_K and Q6_K at K = 1024 and 3584 (3584 has a 2-superblock tail chunk), and Q8_0 at K = 2048, each with M = 100 (not a multiple of the 8 rows per block) and every N = 1..8 plus the N = 16 reference. That exercises every NC instantiation and the row bounds.
- **Results:** memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors, in all five.
- **initcheck:** for initcheck see §5.
- **Setup:** under WSL2 the sanitizer needs the WDDM debugger interface: registry value `HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger\EnableInterface = 1`, set by an administrator.

## 3. The N = 7–8 slowdown: cause and fix

**Symptom.** The first build staged one `float4` (t, float(C), dmA.y, dsB.y) per (row, column, group). Its Q4_K kernel ran at 1.10–1.64× default at N = 8, slower than forced MMQ on some shapes ([`perf-nbkq-v1-float4.csv`](../../results/v1.7/perf-nbkq-v1-float4.csv)).

**Cause** (`ptxas -v` with the build's own flags; [`results/v1.7/ptxas-nb-kq.txt`](../../results/v1.7/ptxas-nb-kq.txt)):
- **Not registers:** no spills at any NC, and 87–163 registers. With 64-thread blocks, registers do not limit occupancy.
- **Shared memory:** the float4 staging grows from 13,056 B at NC = 2 to 34,176 B at NC = 7 and 38,400 B at NC = 8. At NC = 7–8 that leaves one 2-warp block per SM.
- Q6_K stages one float per term and stays at 17,504 B or less, which is why it did not slow down.

**Fix: two layouts, chosen per width.** Both use the same float operations on the same operands, in the same order:
- **compact staging:** `int C` per (row, column, group), `float2 dmA` per (row, group) and `float2 dsB` per (column, group). The fold forms `__fmul_rn(dmA.x, dsB.x)` and `float(C)` itself. NC = 8 drops to 19,392 B, with no spills.
- **Compact alone was slower at N = 2:** 0.84–1.36× default, against 0.74–1.21× for float4 ([`perf-nbkq-v2-compact.csv`](../../results/v1.7/perf-nbkq-v2-compact.csv)). With only 8 lanes folding, the extra FMUL and the two extra shared loads sit on the serial chain.
- **The hybrid** therefore uses float4 at NC = 2 and compact at NC ≥ 3. NC = 2 keeps exactly float4's resources, 87 registers and 13,056 B.

**Per op, Q4_K, new / default on the real shapes** (m ≥ 512; medians over 3–10 samples per cell; [`perf-nbkq-v3-hybrid.csv`](../../results/v1.7/perf-nbkq-v3-hybrid.csv)):

| build | N = 2 | N = 3..6 | N = 8 | forced MMQ, N = 2..8 |
|---|---|---|---|---|
| float4 | 0.74–1.21 | 0.67–1.19 | 1.10–1.64 | 1.09–5.33 |
| compact | 0.84–1.36 | 0.64–1.18 | 0.72–0.98 | 1.09–5.27 |
| **hybrid (final)** | **0.73–1.16** | (= compact) | **0.72–0.98** | 1.09–5.42 |

- The remaining cells above 1.05× are:
  - the K = 3584 shape (1024 × 3584; it has a 2-superblock tail chunk) at N = 2–3: 1.16–1.18×;
  - in the compact measurements, a few large-K shapes at N = 3 and 5: 1.06–1.10×.
- Q6_K is unchanged by the fix: 0.66–1.32× at N = 2..8 ([`perf-nbkq-v1-float4.csv`](../../results/v1.7/perf-nbkq-v1-float4.csv)).
- The rebuilt build was gated again with the full sweep, the batch test and all model checks (identical results).

## 4. Speculative decoding and the matched layout A/B

**Setup:** greedy speculation, rs rollback, 19 prompts × 64 tokens, invariant mode. Each run interleaves the new kernels with `NB_KQ=0`, or, in the A/B, the three layouts, over 3 rounds. Medians, tok/s ([`results/v1.7/spec/`](../../results/v1.7/spec/)):

| run | arm | plain | k = 1 | k = 3 | k = 7 |
|---|---|---|---|---|---|
| float4 build | new kernels | 66.7 | 70.0 | 61.7 | 55.5 |
| | `NB_KQ=0` | 66.7 | 52.8 | 55.1 | 53.6 |
| hybrid build | new kernels | 66.2 | 68.9 | 58.1 | 53.8 |
| | `NB_KQ=0` | 66.3 | 52.6 | 54.6 | 53.3 |
| layout A/B (hybrid build) | hybrid | 66.2 | 67.9 | 59.0 | 54.5 |
| | float4 | 66.3 | 69.1 | 60.1 | 55.6 |
| | compact | 66.3 | 68.5 | 59.8 | 54.5 |

**Correctness:** in every run, tokens and full-logit hashes match plain decoding for every k, match each other across arms, and match v1.6.0's [`r1-bi-rs.tok`](../../results/v1.6/spec/r1-bi-rs.tok) (`compare.txt` in each folder).

**The draft's 2+-token batch no longer costs extra in invariant mode:** 9.8–10.3 ms per round at k = 1, against 10.4 in default mode (v1.6.0) and 18.4–18.5 with `NB_KQ=0`.

**A/B: the layout effect at model level is within noise.**
- At k = 1 the catch-up batch is always N = 2, so the hybrid and float4 arms run the identical kernel. They still differ by 1.7%.
- The between-session spread at k = 3 (58.1–61.7) is not reproduced by any layout in one run, so it was session-to-session variation, about ±3%. Compare speeds only within one interleaved run.
- WSL had at least 10.2 GB available in every A/B run. Windows memory pressure during the hybrid-build run therefore does not explain the earlier spread.

**Conclusion:** invariant-mode speculation at k = 1 is now 1.03–1.05× plain invariant decoding on this pair. Larger k still loses at short context, for the reason v1.6.0 gave: the draft's single-token steps cost about 80% of a target step, which has nothing to do with invariance.

## 5. initcheck in upstream forced MMQ (source analysis)

**What initcheck reports.** 224 "uninitialized __global__ memory read" reports per configuration, all in `mul_mat_q<type, J=8, need_check=true>` (types Q8_0, Q4_K, Q6_K). The count is identical at unmodified PR head (`build/pr-bi`, clean worktree): [`results/v1.7/sanitizer/prhead-q4K-initcheck.digest.txt`](../../results/v1.7/sanitizer/prhead-q4K-initcheck.digest.txt).

**Cause (from the source; not runtime-verified).** The y-tile copy loop in `mmq.cuh` (`mul_mat_q_process_tile`, :997–1001) is not bounds-checked:
- with J = 8 and `MMQ_TILE_Y_K` = 36 a tile needs 288 ints, but 256 threads × 2 iterations copy 512;
- in the second iteration, warps 1–7 read 224 ints, which matches the report count;
- only the CTA handling the last column tile at its last k-block runs past the quantized data, into the padding that `mmq.cu:269–273` adds on purpose. The quantize kernel never writes that padding.

**Why it is benign.**
- **The read address:** the first reported address is exactly the end of the quantized src1 data for all three types (for example 0x4800 = 16 · 8 · 144 B for Q4_K at K = 1024).
- **The value is never used:** it is stored into the shared-memory pad `tile_y[288..511]`, which no `vec_dot` reads (indices ≤ 287) and which `tile_x` does not overlap (it starts at 512).
- **The read stays in bounds:** memcheck is clean, and the padding formula covers the worst case.
- **Current master:** `mmq.cuh` is unchanged there.

So this is not a bug, and nothing needs reporting. Zeroing the padding would silence initcheck. A NaN-poison run to confirm the value is unused was not done.

## 6. Limits

- **Scope:** one GPU, one revision, dense `MUL_MAT`, N ≤ 8, the gated K multiples. Larger N, `MUL_MAT_ID` and other architectures keep the PR's forced MMQ.
- **Evidence for exactness:** it rests on identical per-output operation sequences, checked in source and SASS, plus exact compares over adversarial inputs. It is not a formal proof.
- **Timing noise:** about ±3% between sessions at model level on this laptop. µs-scale perf cells have spreads up to 45%. Shapes that also appear in upstream's perf list pool 10 (or 6) samples instead of 5 (or 3).
- **Speculative results:** greedy only, one model pair, short context. The depth ~1000 runs are v1.6.0's.
- **Drivers:** the archived drivers in `scripts/v1.7/` are the scripts that ran, with only their paths changed. They have not been rerun end to end from this repository.
- **Earlier builds:** only the final patch is published. The float4 and compact builds measured in §3 differ from it only in the Q4_K staging, which the patch's `GGML_CUDA_BI_NB_Q4K_LAYOUT=f4|compact` switch reproduces (§4, gated bit-identical in [`layout-ab-gate.csv`](../../results/v1.7/layout-ab-gate.csv)).
