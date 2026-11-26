# v1.6.0: fast batch-invariant matmul kernels for PR #32081 on sm_75, and speculative decoding end to end (2026-11-25/26)

**Background.** In [PR #32081](https://github.com/ggml-org/llama.cpp/pull/32081)'s batch-invariant mode, every quantized `MUL_MAT` goes through forced MMQ, at every batch width N including 1. With the v1.5.0 flash-attention fix the study's fixtures were bit-invariant, but decode ran at about 0.32× of default mode (v1.5.0).

**Question.** Can an N = 1 kernel reproduce forced MMQ's per-output summation order bit for bit, at the speed of the default GEMV (MMVQ)? If so, does that make greedy speculative decoding exact and fast in invariant mode?

**Findings** (GTX 1650, sm_75, Qwen3.5-2B Q8_0 target and Qwen3.5-0.8B Q4_K_M draft):
- **Proven (exact byte compares):**
  - The new kernels give **the same bits as forced MMQ**: Q8_0 at N = 1..8, Q4_K and Q6_K at N = 1. This holds on adversarial inputs, in `test-backend-ops batch`, and for every fixture plan of both models.
  - **Deliberately wrong variants are detected** on almost every input cell. The two that pass are provably equivalent.
  - **Greedy speculative decoding in invariant mode gives plain decoding's exact tokens and byte-identical full logits** for every draft length k = 1..7. **Default mode does not**, from k = 2.
- **Measured:**
  - 2B target decode in invariant mode: **70.0–72.9 tok/s, 1.09–1.14× default mode** (PR head invariant mode: 20.6–20.8).
  - Draft decode: 1.04–1.07× default.
  - Verify batches of 2–8 tokens: 0.85–0.92× default's time.
  - Prefill is unchanged (forced MMQ above N = 8).
  - On this GPU, speculative decoding is still slower than plain decoding at short context in both modes. **Plain invariant-mode decoding is the fastest configuration measured.**

Being faster than default reflects MMVQ's speed on these shapes on this GPU. It is not a general property of invariant mode.

**Where things are:**
- Patch (on the PR head, includes the v1.5.0 FA fix): [`invariant-gemv-fafix-vs-pr-head.patch`](../../patches/v1.6/invariant-gemv-fafix-vs-pr-head.patch)
- Results: [`results/v1.6/`](../../results/v1.6/)
- Harnesses and runner: [`scripts/v1.6/`](../../scripts/v1.6/)
- Environment, model hashes and timelines: [`run-log.txt`](run-log.txt)

Nothing has been posted upstream.

## Prior art

- **Fixed reduction order and no split-K** for batch invariance: Thinking Machines' `batch_invariant_ops`, [PR #16016](https://github.com/ggml-org/llama.cpp/pull/16016), and #32081 itself.
- **Per-token MMVQ for small batches:** makoni's 2026-11-03 patch in [issue #25618](https://github.com/ggml-org/llama.cpp/issues/25618) makes N ≤ 4 match decode by running MMVQ per token.
- **Few-column MMQ:** `mul_mat_q_few` (upstream). It is off on sm_75 and uses its own reduction, so it does not match MMQ either.

None of these is claimed as new here.

**What this release adds** (searches found no earlier instance; that is not proof of absence):
- a GEMV that reproduces **forced MMQ's** order, rather than MMVQ's, so the invariant bits stay identical to the PR head's;
- its extension to N = 2..8;
- the controls;
- the end-to-end speculative-decoding check.

**Comparison with #25618:**
- Per-token MMVQ keeps MMVQ's bits. It cannot cover large N without slowing prefill, and its bits differ from the PR's invariant bits.
- This design keeps the PR's MMQ reference at every N and makes only N ≤ 8 cheap.

The two give different invariant bits, so they are alternatives.

## 1. What has to be reproduced (code reading and SASS, PR head `3f25ea199`)

- **Routing.** In invariant mode every MMQ-supported quantized `MUL_MAT` goes to `ggml_cuda_mul_mat_q` (`ggml-cuda.cu:2555-2559`). The tile width is pinned and there is one tile per block, so Stream-K never runs.
- **Per-output order.** Each output is **one serial left fold over K, in ascending order, in one thread's register** (`mmq.cuh:972-1007`). The write-back is ×1.0. There is no cross-thread float reduction.
- **The integer block dots are exact.** IMMA on sm_75; dp4a gives the same integers. **So the MMA instruction is not needed.**
- **What must be copied exactly, per 32-value block,** confirmed in the PR build's SASS (FTZ from `-use_fast_math`):
  - **Q8_0** (`mmq-vec-dot.cuh:201-276`): `t = FMUL(float(C), dA)`, then `sum = FFMA(t, dB, sum)`. Activations use the D4 layout (fp32 `d`).
  - **Q4_K** (`:366-436`, `mmq-load-tiles.cuh:815-819`): `dmA = half2(d, −dmin) × half2(sc, m)` is computed **in fp16**. Then `FMUL(dmA.x, dsB.x)`, `FFMA(·, float(C), sum)`, `FFMA(dmA.y, dsB.y, sum)`. Activations use DS4: half2 `d` and `s = d·Σq`.
  - **Q6_K** (`:1084-1162`): per 128 values, `tmp = FFMA(float(C0·sc0 + C1·sc1), dB, +0)` plus 3 more FFMAs, then `sum = FFMA(tmp, dA, sum)`. A fixed two-level tree.
- **The activation quantizer** must be MMQ's (`quantize.cu:456-570`): `d_inv = 127/amax`, `q = roundf(x·d_inv)`, `d = 1/d_inv`. The kernels call the real `quantize_mmq_q8_1_cuda`.
- **Why MMVQ cannot be reused:** it differs at every level. Its quantizer, its 8-value partials, its association and its cross-warp tree reduction (`mmvq.cu:826-907`) are all different.
- **The dp4a MMQ path also differs:** it uses `(dA·dB)·Cf`, a different association from the MMA path.

## 2. Kernels (in the patch: `mmq-bi-n1.cu`, `mmq-bi-n1-kq.cuh`, `mmq-bi-n1-kq-v2.cuh`)

**The one constraint:** one output's fold is serial and cannot be split. **What runs in parallel** is the exact integer dots and the `t` products.

**v3 layout, used for all types:**
- A warp holds 4 rows.
- For each chunk of 32 blocks, lane `b` computes block b's term for each row from coalesced 16-B weight loads (staged in shared memory and realigned with funnel shifts) and stores it to shared memory, padded to 33.
- Then lane `r` folds row r's 32 terms serially.

**N = 2..8 (Q8_0):**
- Each weight row is staged once and dotted against all N activation columns.
- Lane `l < 4N` then folds (row `l/N`, column `l%N`).
- The per-output program is unchanged. That is enough, because forced MMQ gives every column of an N-column product the N = 1 result.

**Gate:** NVIDIA sm_75 only (`turing_mma_available && !ampere_mma_available`), F32 activations, contiguous, 16-B aligned, no batch dims.
- Q8_0: `K % 1024 == 0`, N ≤ 8.
- Q4_K / Q6_K: `K % 512 == 0`, N = 1.

**Off switches for A/B tests:**
- `GGML_CUDA_BI_FAST_N1=0`: all types;
- `GGML_CUDA_BI_FAST_N1_KQ=0`: K-quants;
- `GGML_CUDA_BI_FAST_N1_Q8V3=0`: earlier Q8_0 N = 1 kernel;
- `GGML_CUDA_BI_FAST_NB=0`: Q8_0 N ≥ 2.

**The test file** adds perf cases for the model shapes. `LLAMA_TEST_BATCH_INVARIANT_PERF=1` turns the invariant flag on in perf mode.

## 3. Operator-level bit identity and controls

**Harnesses:** [`bi_n1_ctrl.cu`](../../scripts/v1.6/bi_n1_ctrl.cu) (Q8_0) and [`bi_n1_kq.cu`](../../scripts/v1.6/bi_n1_kq.cu) (K-quants). The reference is ggml `MUL_MAT` on CUDA with the invariant flag on.

**Two links:**
- **Against the PR build** (`pr-bi`): every N is forced MMQ, which tests the harness copies of the kernels.
- **Against the patched build:** the reference at N = 1..8 runs the in-tree kernels. Each column of each N is compared byte for byte with N = 16, which is still forced MMQ.

**Input families** (weights quantized with `ggml_quantize_chunk`):
- `normal`;
- `subnormal`: half-subnormal scales, and activations at 1e-44–1e-34;
- `zeros`: zero blocks, so `amax = 0` and `d_inv = ∞`;
- `large`: overflow to ±Inf;
- `cancel`: alternating partial sums;
- `single`: one nonzero per block;
- for K-quants also `large16` and `mid` (|x| ≤ 1500, large finite magnitudes).

**Shapes:** every model shape, including the 248320-row output heads, plus 4096². Seeds: 8–50 per cell (1–2 for the heads).

| result | file |
|---|---|
| Q8_0 production copy vs forced MMQ: **0 of 110.0 M floats differ, 0 of 1,240 seed-cells** | [`q8-sweep-harness-vs-pr-bi.csv`](../../results/v1.6/q8-sweep-harness-vs-pr-bi.csv) |
| Q8_0 in-tree, every column of N = 1..8 vs N = 16: **0 differences**; production copy 0 of 85.3 M floats | [`q8-sweep-intree-n1-8.csv`](../../results/v1.6/q8-sweep-intree-n1-8.csv) |
| Q4_K / Q6_K production copies vs forced MMQ (7 families): **0 of 68.6 M floats differ** | [`kq-sweep-harness-vs-pr-bi.csv`](../../results/v1.6/kq-sweep-harness-vs-pr-bi.csv) |
| Q4_K / Q6_K in-tree N = 1 vs N = 2..16 (8 families): **0 differences** in every cell; production copy 0 of 76.0 M floats | [`kq-sweep-intree-n1.csv`](../../results/v1.6/kq-sweep-intree-n1.csv) |
| 12 Q8_0 and 22 K-quant kernel variants (layouts, rows per warp; each K-quant variant on both types): **0 mismatches each**, with CUDA-event timing | [`variants-q8.csv`](../../results/v1.6/variants-q8.csv), [`variants-kq.csv`](../../results/v1.6/variants-kq.csv) |

The archived Q8_0 harness-vs-PR sweep used N ∈ {1, 2, 3, 5, 8, 16}; the in-tree sweep used N ∈ {1..8, 16}.

**Controls** (each changes one thing in the production kernel; Q8_0 counts from the harness-vs-PR sweep, K-quant counts from the in-tree sweep, whose informative families are normal, zeros, cancel, single and mid):

| control | detected |
|---|---|
| Q8_0 `(dA·dB)·Cf` (the dp4a association) | 1,236 / 1,240 seed-cells; the 4 passes are `large` cells on 16 rows where nearly every output is ±Inf |
| Q8_0 no FMA; 2-way split-K; MMVQ-style quantizer; reversed fold | 1,240 / 1,240 each |
| Q4_K split-K; fp32 `dmA`; m-term first; MMVQ quantizer | every informative cell (480 / 480) |
| Q4_K association swap | 239 / 480: it differs only when `C·dsB.x` rounds |
| Q6_K split-K; flat fold; reversed order; MMVQ quantizer | 306 / 306 each |
| Q6_K `tmp` started with FMUL instead of FFMA(·, ·, +0) | 2 / 306, the sign of zero in `subnormal` |
| Q4_K m-term unfused; Q6_K scales in fp32 | 0: **equivalent by exactness** (fp16×fp16 is exact in fp32; the integer sums are < 2²⁴) |

For Q4_K, `large` and `large16` give non-finite rows. MMQ keeps `s = d·Σq` in fp16, so any 32-value block above about 2047 overflows the whole row. The kernel reproduces that byte for byte, which is why `mid` was added.

**fp64 check** on the quantized operands: 0 rows out of tolerance (normalized error ≤ 6.7e-7). The exceptions are FTZ underflow in `subnormal`, where MMQ does the same, and legal intermediate overflow in Q6_K `large`.

**`test-backend-ops batch -o MUL_MAT`:** 112/112 with the kernels on and with them off ([`model-checks.txt`](../../results/v1.6/model-checks.txt)).

## 4. Model-level bit identity

These use the study's v1.4 replay tools and fixtures ([`model-checks.txt`](../../results/v1.6/model-checks.txt), [`e1-flag-on.csv`](../../results/v1.6/e1-flag-on.csv), [`logit-cmp/`](../../results/v1.6/logit-cmp/)).

| check, invariant mode on | result |
|---|---|
| 2B, 11-token fixture, final build vs PR head + FA fix (`pr-bi-fafix`) | 34/34 byte-identical; 0/34 plans differ from singles |
| 2B, 19-sequence fixture, all 5 plans (singles, whole, c4, c8, pair4) | sha256 of every dump = `ec330c4e…`, the `pr-bi-fafix` digest ([`dumps-q8v3-SHA256SUMS.txt`](../../results/v1.6/dumps-q8v3-SHA256SUMS.txt)). c4, c8 and pair4 run 4- and 8-token batches through the N = 2..8 kernel |
| 2B, singles vs whole / c4 / c8 / pair4 | 0/1440 differing positions each |
| 0.8B draft, 11-token and 19-sequence fixtures, K-quant kernels on vs off | byte-identical; 0/1440 vs singles |
| Invariant mode off, final build vs PR head default | 34/34 byte-identical: default mode is unchanged |

The logit dumps (64.4 GB) were deleted after comparison. Their digests are archived.

## 5. Speed

**Protocol.** Each table comes from one binary, with arms chosen by environment variables only, 5 rounds in shuffled or interleaved order, and the GPU clock recorded before each run ([`clocks/`](../../results/v1.6/clocks/); SM at 1785 MHz apart from idle-start samples). Per-op times include activation quantization.

**Per op vs default (MMVQ), median time ratio:**

| type, shapes | new kernel | forced MMQ (PR head) | file |
|---|---|---|---|
| Q8_0 N = 1, ≥ 512 rows | 0.77–0.96× | 2.6–5.1× | [`perf-q8-n1.csv`](../../results/v1.6/perf-q8-n1.csv) |
| Q8_0 N = 2..8, ≥ 512 rows | 0.69–1.01× | 1.4–5.1× | [`perf-q8-nb.csv`](../../results/v1.6/perf-q8-nb.csv) |
| Q4_K N = 1, draft and upstream shapes | 0.77–1.08× | 3.2–5.8× | [`perf-kq-n1.csv`](../../results/v1.6/perf-kq-n1.csv) |
| Q6_K N = 1, draft and upstream shapes | 0.90–1.23× | 2.5–4.2× | same |

The 16-row tensors (`ssm_alpha` / `ssm_beta`) are 0.99–1.35×, a few µs where launch overhead dominates. In-tree, Q8_0 N = 1 uses the earlier one-row-per-warp kernel below 512 rows.

**Decode** (`bench-bi`, 64 tokens; [`bench-2b-decode.csv`](../../results/v1.6/bench-2b-decode.csv), [`bench-08b-decode.csv`](../../results/v1.6/bench-08b-decode.csv)):

| model, depth 256–4096 | default | invariant + fast kernels | invariant, PR head + FA fix |
|---|---|---|---|
| 2B Q8_0 | 63.5–66.6 tok/s | **70.0–72.9 (1.09–1.14×)** | 20.6–20.8 (0.31–0.33×) |
| 0.8B Q4_K_M | 132–146 | **141–152 (1.04–1.07×)** | 49.5–51.0 (0.35–0.37×) |

Prefill is unchanged at 0.77–0.88× of default.

**Verify-sized batches** (2B target, n tokens with logits for every token, depth 1024, ms per batch; [`verify-batches-2b.csv`](../../results/v1.6/verify-batches-2b.csv)):

| n | default | invariant, new | invariant, forced MMQ |
|---|---|---|---|
| 1 | 15.31 | 13.88 | 13.88 |
| 2 | 16.72 | **15.46 (0.92×)** | 49.63 (2.97×) |
| 4 | 22.03 | **18.84 (0.86×)** | 50.58 (2.30×) |
| 8 | 31.66 | **27.28 (0.86×)** | 52.66 (1.66×) |
| 12–16 | 69.7–70.8 | 83.3–85.2 (1.20×) | same |

**Observation:** forced MMQ costs the same for N = 2..8. That is why small batches, not decode, were the remaining invariant-mode cost.

## 6. Speculative decoding end to end ([`scripts/v1.6/spec-bi.cpp`](../../scripts/v1.6/spec-bi.cpp), [`results/v1.6/spec/`](../../results/v1.6/spec/))

**Setup.** Greedy speculation: the 0.8B draft proposes k tokens, and the 2B target verifies them in one batch. The workload is the 19 fixture prompts, 64 tokens each, k ∈ {1, 2, 3, 4, 5, 7}. The baseline is plain one-token decoding in the same context.

**What is recorded:** every generated token and an FNV-1a hash of the **full 248320-float target logits row** it was chosen from.

**Target rollback** (Qwen3.5 has recurrent layers):
- `rs`: the experimental recurrent snapshots, `n_rs_seq = 8`;
- `ckpt`: save and restore the recurrent state, then re-decode the accepted tokens. The re-decoded row is also compared with the verify row.

**The draft** always restores its own state and feeds the accepted tokens with its next batch.

**Deep variant:** 1000 filler tokens prepended.

**Correctness (proven):**

| arm | k = 1 | k = 2..7 | deep, k = 3 and 7 |
|---|---|---|---|
| invariant, `rs` | identical | **identical: 0/19 prompts differ in tokens or logits** | identical |
| invariant, `ckpt` | identical | identical; re-decode check 0 mismatches | — |
| default, `rs` | identical | **logits differ in 19/19 prompts; tokens in 4/19** | logits 19/19, tokens 0/19 |
| default, `ckpt` | identical | logits 19/19, tokens 4–7/19; re-decode mismatches 31–87 | — |

- **Repeat rounds are byte-identical** to round 1 in every arm ([`tok-SHA256SUMS.txt`](../../results/v1.6/spec/tok-SHA256SUMS.txt)).
- **Snapshots don't change the arithmetic:** plain decoding is byte-identical with `n_rs_seq` = 8 and 0, in both modes.
- **Hypothesis, not checked:** default mode changes a kernel choice between 2- and 3-column batches.

**Speed** (tok/s = generated tokens / decode time, prefill excluded; median of 3 rounds; × = vs plain in the same arm):

| k | invariant, `rs` | invariant, `ckpt` | default, `rs` | acceptance |
|---|---|---|---|---|
| plain | **66.5** | 68.8 | 61.1 | — |
| 1 | 52.6 (×0.79) | 50.9 | 64.8 (×1.06) | 0.91 |
| 3 | 55.0 (×0.83) | 52.3 | 57.5 (×0.94) | 0.85 |
| 4 | 56.3 (×0.85) | 53.9 | 57.5 (×0.94) | 0.81 |
| 7 | 53.0 (×0.80) | 51.4 | 53.3 (×0.87) | 0.70 |
| deep, k = 7 | 72.3 (×1.11) | — | 72.4 (×1.20) | 0.95 |

**Why speculation loses here** (measured per-round breakdown; causes are hypotheses):
- **A draft step costs about 11–12 ms,** about 80 % of a target step. `bench-bi` measures the draft at about 7 ms per token. This harness alternates two contexts with varying batch sizes; CUDA-graph reuse is a plausible cause, not verified.
- **The draft's first batch each round has 2+ tokens.** In invariant mode that batch is forced MMQ for Q4_K / Q6_K: 18.5 vs 10.4 ms at k = 1. **A K-quant N = 2..8 kernel would remove that** (not built).

This matches the study's original server-level conclusion: for this model pair on this GPU, plain decoding is fastest on the ordinary workload.

## 7. Limits

- **One GPU** (GTX 1650, sm_75), **two models,** and the study's fixtures. The gate disables the kernels on every other architecture, where MMQ's association or tiling differs.
- **The bar moves:** any change to MMQ's tile code or vec_dot association would break bit identity silently. The kernels need their own test against forced MMQ, not just a batch-invariance test.
- **Not covered:** N > 8, which stays on forced MMQ; K-quants at N ≥ 2; `MUL_MAT_ID` / MoE; other quant types.
- **Speculative decoding** was tested with a library-level harness, not `llama-server`.
- **The runner** in `scripts/v1.6/run.sh` is a parameterized copy of the drivers that ran. It has not been rerun end to end.
