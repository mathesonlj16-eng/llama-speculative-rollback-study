# Evidence pack: exact, fast small-batch kernels for llama.cpp's batch-invariant quantized matmul on Turing

This page collects the study's batch-invariance results from v1.4.0 to v1.7.0 into one place: claims, setup, evidence and limits. Every figure links to the release findings it comes from; where they differ, the findings win.

On a GTX 1650 (sm_75), the study's kernels reproduce the output bits of [PR #32081](https://github.com/ggml-org/llama.cpp/pull/32081)'s batch-invariant forced MMQ exactly, for Q8_0, Q4_K and Q6_K at 1–8 tokens per batch. They run at about the default kernel's speed instead of 1.1–5.8× slower. This makes invariant decoding faster than default mode on this GPU, and greedy speculative decoding bit-exact against plain decoding.

## Claims

All claims are scoped to PR #32081 head `3f25ea199a36a9e83480b6dac0941bb82c96c5b9`, one GTX 1650 (TU117, CC 7.5), dense `MUL_MAT`, and the shapes and types tested.

| # | Claim | Kind | Evidence |
|---|---|---|---|
| C1 | Kernels for Q8_0, Q4_K and Q6_K at N = 1..8 whose every output is **bit-identical to PR #32081's forced MMQ** | proven: exact compares | [v1.6.0](../v1.6/FINDINGS.md) §3–4, [v1.7.0](FINDINGS.md) §2 |
| C2 | Their per-op time is **0.64–1.32× the default kernel (MMVQ)** on the real decode shapes, mostly ≤ 1.05×. Forced MMQ is 0.9–5.8× | measured | v1.6.0 §5, v1.7.0 §3 |
| C3 | With them plus the v1.5.0 FA fix, the 2B model is **fully batch-invariant** on every fixture plan and decodes at **1.09–1.14× default mode** | proven / measured | [v1.5.0](../v1.5/FINDINGS.md), v1.6.0 §4–5 |
| C4 | Greedy speculative decoding in invariant mode is **bit-exact against plain decoding**, in tokens and full logits. At draft length k = 1 it is **1.03–1.05× plain invariant decoding**, up from 0.79× | proven / measured | v1.6.0 §6, v1.7.0 §4 |
| C5 | Regressions specific to #32081: the FA split follows the padded KV length (the fix credits prior art), and the tile-width pin is unnecessary on sm_75 for the tested types | measured | [v1.5.0](../v1.5/FINDINGS.md), [v1.3.0](../v1.3/FINDINGS.md) |

**Not claimed:**
- batch invariance in general, fixed-order reductions, adaptive invariant tiling, arithmetic-equivalence arguments, or the FA split technique, all of which are prior art;
- "first": the prior-art search was targeted, not exhaustive.

## Setup

- **Hardware and software:** GTX 1650 laptop GPU (4 GB), Ryzen 7 5800HS, Windows 11 + WSL2, driver 592.00, CUDA 13.2. Builds use upstream's `-use_fast_math` (FTZ). Full environment: each release's `run-log.txt`.
- **The reference bits:** PR #32081 in invariant mode on the **stock sm_75 build**: forced MMQ, pinned J = 8, no Stream-K, the Turing integer-MMA path. This is what the PR executes on this GPU. The study's diagnostic dp4a build is a different arithmetic path, differing from it on 450/450 compared hashes ([v1.3.0](../v1.3/FINDINGS.md)). It is not the reference.
- **Why exact reproduction is possible.** In invariant mode every output column already gets the N = 1 per-output program, a serial chain over 32-value blocks:
  - integer block dot products, which are exact;
  - then a fixed sequence of `FMUL.FTZ` / `FFMA.FTZ` in ascending K.

  The kernels compute the integer dots in any lane layout. They fold the float terms serially in that order, with the operand roles copied from the reference kernel's SASS. The float chain is never split or reassociated.
- **Gates** (otherwise the PR's forced MMQ runs unchanged):
  - sm_75 only;
  - `MUL_MAT`, 2-D, contiguous, F32 activations and output;
  - N ≤ 8;
  - Q8_0 with K % 1024 = 0, and Q4_K/Q6_K with K % 512 = 0.
- **Models and fixtures:** Qwen3.5-2B Q8_0 (target) and Qwen3.5-0.8B Q4_K_M (draft). The fixtures are the 11-token E1 chunking fixture (34 plans) and the 19-sequence fixture (plans singles, whole, c4, c8, pair4).
- **Patch:** [`patches/v1.7/invariant-gemv-nbkq-fafix-vs-pr-head.patch`](../../patches/v1.7/invariant-gemv-nbkq-fafix-vs-pr-head.patch), against PR head. It includes the FA fix and research-only test cases and switches.

## Correctness (exact byte compares)

| check | Q8_0 | Q4_K / Q6_K |
|---|---|---|
| **Operator sweep:** every column of N = 1..8 through the real library path vs N = 16 forced MMQ | 0 differences (v1.6.0) | 0 differences in 46.8 M floats on 13 draft shapes incl. the 248,320-row head (v1.7.0) |
| **Input families** | normal, subnormal/FTZ range, zero blocks, overflow to ±Inf, cancellation, single nonzero | the same, plus −0.0 activations |
| **Negative controls** | 5 deliberately wrong variants fail on essentially every cell (v1.6.0) | wrong variants are detected, except the ones that are provably equivalent (v1.6.0); default-mode MMVQ fails every row of the informative families (v1.7.0) |
| `test-backend-ops batch -o MUL_MAT` | 112/112 | 112/112 |
| **Sanitizer** (memcheck, racecheck, synccheck) | 0 errors | 0 errors (v1.7.0) |
| **Model fixtures, BI on** | 2B: every plan byte-identical to PR head + FA fix | 0.8B: every plan byte-identical with the kernels on and off |

The degenerate families (outputs ±0 or ±Inf) cover special values but separate float programs poorly. The informative families are normal, zeros, cancellation and single nonzero.

**Speculative decoding** (v1.6.0 §6, v1.7.0 §4): 0.8B draft and 2B target, 19 prompts × 64 tokens. Each generated token is recorded with a hash of its full 248,320-float logits row.

| mode | k = 1 | k = 2..7 | depth ~1000 |
|---|---|---|---|
| **invariant** (both rollback methods; every build and layout) | identical to plain | **identical: 0/19 prompts differ in tokens or logits** | identical |
| default | identical | logits differ in 19/19; tokens in 4–7/19 | logits differ 19/19 |

## Performance

**Protocol.**
- Matched `test-backend-ops perf`: all arms in one binary, 3–5 shuffled rounds, medians, GPU clock logged (1785 MHz apart from idle-start samples).
- Model timings: interleaved arms.
- Between sessions, model-level figures vary by about ±3% on this laptop (v1.7.0 §4), so compare only within a run.

**Per op, new kernel / default (MMVQ), real decode shapes:**

| type | N = 1 | N = 2..8 | forced MMQ |
|---|---|---|---|
| Q8_0 (≥ 512 rows) | 0.77–0.96 | 0.69–1.01 | 2.6–5.1 (N = 1), 1.4–5.1 (N = 2..8) |
| Q4_K | 0.77–1.08 | 0.64–1.18 (N = 2: 0.73–1.16; N = 8: 0.72–0.98) | 3.2–5.8 (N = 1), 1.1–5.4 (N = 2..8) |
| Q6_K | 0.90–1.23 | 0.66–1.32 | 2.5–4.2 (N = 1), 0.9–4.1 (N = 2..8) |

The 16-row tensors are 0.96–1.35×, a few µs where launch overhead dominates. The worst real cells are the K = 3584 shapes, which have a tail chunk (up to 1.32×). Sources: v1.6.0 §5, v1.7.0 §3.

**Model level** (v1.6.0 §5; 2B target unless noted):

| metric | default | invariant, PR head + FA fix | invariant + these kernels |
|---|---|---|---|
| decode, depth 256–4096 | 63.5–66.6 tok/s | 20.6–20.8 (0.31–0.33×) | **70.0–72.9 (1.09–1.14×)** |
| 0.8B draft decode | 132–146 tok/s | 49.5–51.0 (0.35–0.37×) | **141–152 (1.04–1.07×)** |
| verify batch, 2–8 tokens | 1.00× | 1.66–2.97× time | **0.85–0.92× time** |
| prefill | 1.00× | 0.77–0.88× | 0.77–0.88× (N > 8 stays on forced MMQ) |

"Faster than default" means MMVQ is slower per op than these kernels on these shapes on this GPU. It is not a property of invariant mode in general.

**Speculative decoding, invariant mode, short context, tok/s** (v1.6.0 §6, v1.7.0 §4; ranges over runs):

| k | draft batches on forced MMQ | all kernels |
|---|---|---|
| plain | 66.1–66.7 | 66.2–66.7 |
| 1 | 52.6–52.8 (×0.79) | **67.9–70.0 (×1.03–1.05)** |
| 3 | 54.6–55.1 (×0.83) | 58.1–61.7 (×0.88–0.93) |
| 7 | 53.0–53.6 (×0.80) | 53.8–55.6 (×0.81–0.84) |

Larger k still loses at short context, because the draft's single-token steps cost about 80% of a target step, which is unrelated to invariance. At depth ~1000 on predictable text, k = 7 reached 1.11× (v1.6.0, before the v1.7.0 kernels).

## Negative results and corrections

- **The Q4_K slowdown at N = 7–8 was shared-memory occupancy, not registers.** Compact staging everywhere slowed N = 2, hence the per-width hybrid (v1.7.0 §3).
- **A 58.1–61.7 tok/s spread at k = 3 across sessions was noise, not the code.** A matched A/B settled it (v1.7.0 §4).
- **Default mode is invariant at k = 1 but not from k = 2.** The cause, a kernel choice between 2 and 3 columns, is an unchecked hypothesis (v1.6.0).
- **An earlier statement that the draft decodes one token at a time was wrong;** the catch-up batch has 2+ tokens (v1.6.0).
- **Sanitizer initcheck reports in upstream's forced MMQ** are intended padding over-reads whose values are never used. The count is identical at unmodified PR head; the analysis is from source (v1.7.0 §5).

## Prior art

**Checked against source or the merged change:**

| work | relation |
|---|---|
| [#31533](https://github.com/ggml-org/llama.cpp/pull/31533) / [#32065](https://github.com/ggml-org/llama.cpp/pull/32065), few-token MMQ | Ampere–Blackwell only, `cp.async` staging, K split across warps with FP32 partials. Different arithmetic, no sm_75 path. Prior art for weight reuse across a few tokens |
| [#30741](https://github.com/ggml-org/llama.cpp/pull/30741), OpenCL cooperative-K dp4a GEMM | small-batch Q4_0/Q4_K/Q6_K with split float accumulation; no invariance claim. Prior art for cooperative small-batch dp4a |
| [#25618](https://github.com/ggml-org/llama.cpp/issues/25618) (makoni's patch) | the KV-length-independent FA split, and per-token MMVQ for small batches. Credited in v1.5.3 |
| [#16016](https://github.com/ggml-org/llama.cpp/pull/16016) | earlier deterministic CUDA inference mode (open proposal) |

**Leads from a separate literature and code audit, not independently verified here:**
- Thinking Machines' "Defeating Nondeterminism in LLM Inference" and `batch_invariant_ops`;
- vLLM's architecture-aware batch-invariant tiling work;
- recent papers on bitwise reproducibility of tensor-core and quantized kernels;
- the Axon inference engine, which defines its own canonical quantized contract rather than reproducing MMQ.

**What remains distinct** (absence in a non-exhaustive search, not a priority claim): we found no work that reproduces **an existing, pinned quantized-matmul reference's bits** with faster small-batch kernels on a GPU the reference already supports. The closest are:
- engines that define their own invariant contract;
- few-token kernels with different arithmetic;
- adaptive tiling that keeps a reduction fixed rather than reproducing a quantized program.

## Limitations

- **Scope:** one GPU, one revision, three weight types, dense `MUL_MAT`, N ≤ 8, two models. Bit identity holds per build: different compiler flags could change the reference itself.
- **Evidence for exactness:** identical per-output operation sequences, checked in source and SASS, plus exact compares over adversarial inputs. It is not a formal proof.
- **Timing:** a laptop GPU with about ±3% between sessions; µs-scale shapes are noisy. All ratios are within-run.
- **Speculative results:** greedy only, one model pair, short context.
- **Model-level invariance** also needs the v1.5.0 FA fix. `MUL_MAT_ID`, the F16/BF16 MoE gaps (v1.3.x) and other ops are out of scope.

## Reproduction

See [REPRODUCE.md](../../REPRODUCE.md):
- v1.6.0 has a parameterized runner;
- v1.7.0 ships the archived drivers that ran, with only their paths changed;
- `scripts/verify-results.py` rechecks the archived aggregates.

## AI-use disclosure

The kernels, harnesses, experiment scripts, measurements and this write-up were produced with AI assistance under the author's direction and review:
- Claude ran the local implementation and experiments;
- ChatGPT and Codex ran separate prior-art and source audits.

All figures come from logged runs on the author's hardware. Nothing here has been submitted upstream. Any contribution to llama.cpp will follow that project's AI policy.
