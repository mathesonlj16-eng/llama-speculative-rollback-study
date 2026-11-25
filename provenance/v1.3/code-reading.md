# Does the MMQ reduction order depend on tile width J? (code reading, no GPU)

Source: PR #32081 head `3f25ea199`, `ggml/src/ggml-cuda/{mmq.cu, mmq.cuh, mmq-vec-dot.cuh, mmq-config-ampere.cuh}`. Read 2026-11-25.

## What changes with J on sm_75

- **Config table:** sm_75 builds, including the dp4a-switched ones, use `ggml_cuda_mmq_get_config_ampere`, because the highest compiled architecture is ≥ Volta. For IQ3_S, IQ4_XS, Q4_K, Q5_K and Q8_0, every J entry has the same `nthreads=256`, `I=128` and `K_vram=MMQ_ITER_K=256`. So the K loop in `mul_mat_q_process_tile` (`kb0 += blocks_per_iter`) and the warp count are independent of J.
- **J-dependent:** the number of columns per tile, which thread holds which output, and on MMA `rows_per_warp` (16, or 32 when J ≥ 48 and J % 16 == 0), i.e. `ntx`.

## Per-output arithmetic

- **dp4a vec_dots** (`q8_0_q8_1_dp4a`, `q4_K_q8_1_dp4a`, `q8_1_q8_1_dp4a`): for each `k01` in ascending order, each output does `sum[(i,j)] += impl(row i, col j, k0)`. The term depends only on (i, j, k0), and the update order is K-ascending whatever J is.
- **Turing MMA vec_dots** (`q8_0_q8_1_mma`, `q8_1_q8_1_mma`, non-AMD branch): for each `k01`, an int8 `mma` computes an exact integer partial over one 32-value block. Then `sum += C*dA*dB` (or `dmA.x*dsB.x*C` + `dmA.y*dsB.y`), again K-ascending. Integer accumulation is exact, so how the tiles are grouped can't change the bits.
- **src1 quantization** (`quantize_mmq_q8_1_cuda`) works per column and is independent of N.

## What actually changes the order: Stream-K

- With Stream-K, a tile's K range is split across blocks. The partial sums are merged by `mul_mat_q_stream_k_fixup`, which changes the order.
- Default NVIDIA MMQ already avoids the split in two cases: when tile efficiency is at least 90% (`one_tile_per_block`), or when `force_tiling` holds (at least 2 waves of tiles, or few K chunks).
- The PR's `ctx.batch_invariant` forces one tile per block.
- This explains the earlier cross-arm result: invariant N=1 (J=8, whole tiles) differed from default MMQ at N=9 (max abs 3e-5). At N=9 the tile efficiency is 9/16, and with M=256 there are too few tiles, so Stream-K split K.

## Prediction to test

For these 5 types on sm_75 (dp4a and MMA paths), with one tile per block:

1. Any fixed J gives bitwise batch invariance.
2. **The PR's pin `ncols_opt = 1` (mmq.cu:262) should not be needed.** Letting J follow N, as default dispatch does, should still be invariant and match J=8 bit for bit.

Then the invariant mode's large-N cost would only be the loss of Stream-K. Its N=1 cost (MMQ instead of MMVQ) is unchanged, because J_best(1) is already 8.

## Residual risks the test must cover

- **Compiler effects:** different J template instances could be compiled differently (FMA contraction, instruction selection). The arithmetic is the same expression, but this hasn't been checked.
- **Out of scope:** other types (Q4_0/Q6_K/Q2_K/Q3_K/IQ2* use other vec_dots), MUL_MAT_ID, and the Blackwell and AMD configs, where I, nthreads and K_vram may vary with J.

## Cheapest test

The change is host-only, in `mmq.cu`: an env switch that skips `ncols_opt = 1`, and optionally pins J. Only `mmq.cu` recompiles, not the template instance files, so this takes minutes per build.

1. Run the batch grid with J free on both paths: the `pool-d4f-pr` and `pool-s75-pr` trees.
2. Cross-check output hashes with J free against J=8 at the same N.
3. Short dp4a timing at N = 1, 16 and 512.
