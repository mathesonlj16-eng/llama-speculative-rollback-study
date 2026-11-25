# v1.3.1: coverage for the tile-width result (2026-11-25)

Builds are the same as in [FINDINGS.md](FINDINGS.md):
- `pool-d4f-pr`: the dp4a diagnostic build
- `pool-s75-pr`: stock sm_75, Turing MMA path

Every run was done twice, once with the tile width free (`SM75_INV_NCOLS=0`) and once pinned to the PR's narrowest width (unset). `SM75_SEED=1234`. Summaries: [`coverage-compare.csv`](../../results/v1.3/coverage-compare.csv), [`coverage-analysis.txt`](coverage-analysis.txt), [`coverage-run-log.txt`](coverage-run-log.txt). Raw logs are archived outside this repository.

## Results

| Test | Cases per run | Free J | Pinned J | Free vs pinned bits |
|---|---|---|---|---|
| 13 quant types: q2_K, q3_K, q6_K, q4_0, q4_1, q5_0, iq2_xxs, iq2_xs, iq3_xxs, iq4_nl, iq3_s, q4_K, q8_0. M ∈ {256, 1000}, K ∈ {1024, 4352}, N ∈ {9, 16, 24, 64, 128, 512} | 312 | 0 fail, both builds | 0 fail, both builds | identical, 312/312 on both builds |
| K not a multiple of 256: q4_0, q8_0, q5_0, iq4_nl. M ∈ {1000, 4096}, K ∈ {1056, 2080}, N ∈ {1, 9, 16, 64, 512} | 80 | 0 fail, both builds | 0 fail, both builds | identical, 80/80 on both builds |
| PR's own batch suite: MUL_MAT incl. F32/F16/BF16/Q4_0/Q6_K, MUL_MAT_GLU, MUL_MAT_ID, FLASH_ATTN_EXT, MUL_MAT_ATTN | 224 (sm_75) | 0 fail on sm_75; 16 fail on dp4a | 0 fail on sm_75; 18 fail on dp4a | identical except MUL_MAT_ID (see below) |

M = 1000 exercises the `fallback` configs, whose tile widths are limited to 8, 16, 32, 64 and 128.

## Notes

1. **MUL_MAT_ID (MoE) needs care in reading.**
   - Hashes differ between the free and pinned runs, but they also differ between two identical pinned runs: 0 of 36 equal. The test's expert routing isn't covered by the local `SM75_SEED`, so comparing bits across runs doesn't work for MoE.
   - Within each run, batch equals serial (OK) on the stock build in both settings.
   - The PR's MoE cases use n ≤ 8 tokens with 128 experts. The per-expert column count is then 1, so J = 8 even when it's free. **The suite therefore doesn't exercise wider tile widths for MoE.** That is still open.
2. **All dp4a-build failures are F16 MUL_MAT_ID: 30/30 FAIL lines across 4 runs.** They happen with J pinned and free alike, and the number changes from run to run (7 vs 9 in repeated pinned runs) as the routing changes. MMQ doesn't handle F16, so this isn't related to tile width. It appears only when the diagnostic switch disables `TURING_MMA_AVAILABLE`/`turing_mma_available()`; the stock sm_75 build has 0. This could mean the PR's invariant mode misses an F16 MoE path on GPUs without Turing MMA, such as Pascal. That's a hypothesis: the switch isn't a real pre-Turing build, and the path taken wasn't traced.
## Follow-up: MoE at larger tile widths (13:19–13:28)

**Harness changes** ([`test-backend-ops-moe-vs-pr-head.patch`](../../patches/v1.3/test-backend-ops-moe-vs-pr-head.patch), the complete `tests/test-backend-ops.cpp` diff against the PR head; it supersedes the test-file part of the v1.2 pool patch):
- `SM75_SEED` now also seeds the MoE routing (`init_mul_mat_id_ids`).
- A new `SM75_MOE_TYPES`/`SM75_MOE_N` list adds MUL_MAT_ID cases with few experts, so the number of tokens per expert grows with N.

**Grid:**
- Types: q4_K, q8_0, q6_K, iq4_xs.
- 8 experts with 2 used, and 4 experts with 2 used.
- Variants: plain, GLU, and GLU + shared expert.
- N ∈ {16, 64, 128, 256}, capped per variant. The serial half adds graph nodes for every token, and the first attempt hit the test's 8192-node limit (aborted in `ggml_build_forward_expand`).
- 52 cases per run.

**Tile widths used:**

| Setting | J |
|---|---|
| free | 8, 16, 32, 64, 128 |
| pinned | 8 |

**Results:**

| Build | Free J: 0 fail, batch == serial | Pinned run A == B (seed check) | Free bits == pinned bits |
|---|---|---|---|
| dp4a | 52/52 | 52/52 | 52/52 |
| stock sm_75 | 52/52 | 52/52 | 52/52 |

The tile-width result therefore also holds for quantized MUL_MAT_ID at tile widths up to 128, with the routing seeded.

## Notes, continued

3. The quant-type and odd-K results extend the MUL_MAT conclusion from 5 to 13 types, and to fallback-config M and to K values that aren't a multiple of 256. With Stream-K off, tile width doesn't change the bits.

MoE summaries: [`moe-compare.csv`](../../results/v1.3/moe-compare.csv), [`moe-analysis.txt`](moe-analysis.txt) and [`moe-run-log.txt`](moe-run-log.txt).

The scripts are in [`scripts/v1.3/`](../../scripts/v1.3/):
- `cov.sh`, analyzed by `cov_an.py` and `cov_an2.py`
- `cov3.sh`, the unseeded-routing repeat
- `moe.sh`, which applies `edit_moe.py` and `edit_moe2.py` and is analyzed by `moe_an.py`

The `ok=`/`fail=` counts in the run logs come from a grep that misses colour-coded lines. Use the analysis files, which strip the colour codes before parsing.
