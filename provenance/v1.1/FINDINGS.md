# v1.1.0: What explained the concurrent CUDA differences

Investigation date: 24 November 2026. Same host, GPU (GTX 1650, SM 75, driver 592.00), models and harness as v1.0.0.

## Question

The v1.0.0 independent rerun (CUDA 13.2.86) matched the archive (CUDA 13.1.115) on every serial two-slot run but not on concurrent ones. The archive's six concurrent runs shared one token-array set with 34 checkpoint restores (`437590e4…`, called **C-archive** below). The rerun produced a different set with 39 restores (`9170e155…`, **C-rerun**). One rerun flag-off run produced a third set (`21efb482…`, **C-rerun-alt**). The 22 forced-admission runs didn't reproduce that third set. v1.0.0 also reported 34- and 39-restore "modes" of the paired baseline with an unknown cause. The proposed hypothesis was the CUDA toolkit change.

## Answer

**The CUDA toolkit does not explain any of these differences.** One test-harness setting explains the archive-vs-rerun difference and the 34/39 modes: the target's recurrent snapshot capacity (`n_rs_seq`). Request admission order explains the alternative token set.

All 198 concurrent v1.1 runs, classified by target capacity and by which prompt was admitted first:

| Target capacity | Admission | Builds | Runs | Token-array set | Target restores |
|---|---|---|---:|---|---|
| 0 | prompt 0 first or within 15 ms | original 13.1 binaries, fresh 13.1, 13.2 | 23 | C-archive (all) | 34 |
| 0 | prompt 1 first by ≥15 ms | 13.2 | 25 | `6e86f030…` (all) | 35–36 |
| 3 | prompt 0 first or within 15 ms | original 13.1 binaries, fresh 13.1, 13.2 | 98 | C-rerun (all) | 35–39 |
| 3 | prompt 1 first by ≥15 ms | fresh 13.1, 13.2 | 52 | C-rerun-alt (all) | 37–38 |

No run on the original 13.1 binaries happened to admit prompt 1 first, and the reverse-stagger control used only the fresh 13.1 and 13.2 builds.

The v1.0.0 rerun's 28 classified concurrent runs follow the same rule: 27 C-rerun and its single C-rerun-alt run, in which prompt 1 led by 21 ms. The archive's v1.0.0 flag-off control ran at capacity 0 with near-simultaneous admission and gave C-archive with 34 restores. Paired and flag-off builds produced identical token arrays within every condition. Serial runs matched the archive (`b654ea17…`, 33 restores) at both capacities and on every build.

## Evidence

1. **Builds and device code.** Clean CUDA 13.1.115 builds of stock, paired and target-prototype were made at pin `787a5612` with the release CMake options and patches. GGML options matched the retained 13.2 builds exactly. After removing addresses and per-file namespace hashes, the fresh 13.1 `libggml-cuda` SASS is identical to the archived 13.1 library's (all 7,236 kernels). Between 13.1 and 13.2, 603 of 7,210 common kernels differ, mostly `mul_mat_q` (178), `flash_attn_tile` (40), `mul_mat_vec_q` (33) and `mul_mat_q_stream_k_fixup` (32). All builds load the same CUDA 13.2 runtime libraries (cuBLAS 13.4.1.3), because `/usr/local/cuda` has pointed to 13.2 since 21 November, before the v1.0.0 experiments. The archived 13.1 runs therefore also used them. [Build and environment record](build-and-environment.txt)
2. **Original binaries rerun.** The archived 13.1 binaries (hashes match the v1.0.0 record) ran in today's harness and produced C-rerun with 39 restores, the same as 13.2. The same bytes behave differently today, so the build can't be the cause.
3. **Logits.** Six prompts were teacher-forced on the archived concurrent token arrays (288 positions per comparison). Logits are bit-identical between the original 13.1, fresh 13.1 and 13.2 libraries for every batch shape tested: one token per decode, whole sequence, chunks of 4 and 8, and two sequences decoded together in chunks of 4. Within one build, batch shape changes logits by up to 1.27 and flips the top token at margins as small as 0.003. [Toolkit and shape comparison](../../results/v1.1/logit-toolkit-and-shape.csv)
4. **The capacity setting.** Every archived v1.0.0 flag-off control log and every archived interleaved *paired* log reports `target n_rs_seq=0`. The prototype sets its own capacity and reports 3. The archive's evaluation and profiling runs report the shim's value, which equals the draft maximum (3 for the compared runs). Across the 14 archived two-slot, draft-maximum-3 paired-style runs on the ordinary workload (flag-off control, interleaved paired, evaluation), capacity 0 gave 34 restores and capacity 3 gave 39, with no exceptions. Other archived experiments used different workloads or draft maxima and aren't comparable. [Archive audit](../../results/v1.1/archive-capacity-audit.csv) The release `run.sh` loads the capacity preload correctly, so the rerun and every default-shim v1.1 run report `n_rs_seq=3`. The archived preload binary (`f3d2d519…`) also loads correctly today. The dynamic-linker trace shows `llama_init_from_model` interposed for both shims. The exact invocation that left the archive's flag-off and interleaved runs without the shim was not recorded.
5. **Decisive control.** A preload that intercepts nothing (`scripts/v1.1/nocap.cpp`) leaves the target at capacity 0. With it, the original 13.1 binaries, fresh 13.1 and 13.2 each reproduce the archive: C-archive with 34 restores in 18 of 18 concurrent runs, and the archive's serial set in 18 of 18 serial runs.
6. **Mechanism.** At the pin, `n_rs_seq > 0` makes the hybrid memory's ubatch splitter keep each sequence's trailing `n_rs_seq + 1` tokens together (`split_equal(..., n_rs_seq + 1)`, `src/llama-memory-hybrid.cpp`). With one sequence the split is unchanged. When two slots share a decode step with unequal token counts, the ubatch composition changes, and so do the matrix shapes. In teacher-forced logits, capacity 0 and 3 are bit-identical for a single sequence (288/288). For two sequences decoded together, 96 of 288 positions differ (max 0.378), with 3 top-token flips at margins down to 0.0008. Both toolkits give identical results at either capacity. [Capacity comparison](../../results/v1.1/logit-capacity.csv)
7. **The alternative set is admission order, not the flag.** Each request is pinned to the slot matching its prompt index. In both earlier C-rerun-alt runs, prompt 1 was admitted first and prompt 0 joined about 22 ms later. The v1.0.0 admission control only ever delayed prompt 1. A reverse-stagger control (`scripts/v1.1/reverse-admission.py`) delays prompt 0 by 0, 10, 21, 30 or 50 ms. It produced C-rerun-alt in all 48 runs with a delay of at least 10 ms, on both toolkits and for both paired and flag-off. At capacity 0, the same order produced a separate, equally stable set (`6e86f030…`) in all 24 delayed runs. Across 226 classified concurrent runs from v1.0.0's rerun and v1.1, prompt 1 leading by at least 15 ms gave the alternative set in 78 of 78 runs. Every other admission pattern gave the default set for its capacity in 148 of 148. [Admission timelines](admission-timeline.txt) · [Reverse-stagger results](../../results/v1.1/reverse-admission-matrix.csv)

## Consequences for v1.0.0 statements

- "Concurrent two-slot CUDA runs produced token sequences and restore counts that differed from the archive … Whether the toolkit change explains this difference was not tested." The toolkit is now ruled out. The difference came from target capacity 0 (archive) vs 3 (rerun) in the paired and flag-off servers.
- "The paired server exhibited two restore-count modes … The mechanism behind the two modes remains open." The modes are capacity 0 (34) and capacity 3 (39).
- Interleaved speed comparison: the archived 15.3% compared paired at capacity 0 with the prototype, which always runs at capacity 3. The rerun's 20.4% gave the paired baseline an unused capacity of 3. That baseline doesn't need snapshots, and capacity 3 costs it about 0.4–0.6 s per run. Like-for-like reruns today measured 13.9% (13.2) and 16.2% (13.1) at paired capacity 0, versus 18.3–20.6% at capacity 3. Paired at its default capacity is the configuration a deployment would run, so the ~14–16% figure is the fair one. [Interleaved results](../../results/v1.1/interleaved-matrix.csv)
- "One concurrent flag-off run differed … its cause is unresolved." It is a reproducible consequence of admission order, occurs equally with the paired build, and does not indicate a flag-off defect.
- The flag-off equivalence claim is strengthened: across all 198 concurrent v1.1 runs, the token arrays depended only on capacity and admission order, never on whether the build was paired or flag-off.

## Limitations

- The archived flag-off and interleaved invocation was not recorded, so *why* the shim was absent there is inferred, not observed. The archive logs, the shim's correct behavior today and the decisive no-capacity reproduction establish *that* it was absent, and that this alone accounts for the difference.
- Logit equality across toolkits covers the six study prompts and the tested shapes. Other shapes may exercise the 603 differing kernels. The 13.1/13.2 difference in device code was not shown to be numerically inert in general.
- One host and GPU. The finding is about this harness and pin, not upstream llama.cpp at large. That concurrent greedy output in llama.cpp depends on batch composition is expected behavior, not a new defect.

## Superseded hypotheses

| Hypothesis | Status | Deciding evidence |
|---|---|---|
| CUDA 13.1 vs 13.2 compiler | Rejected | Identical logits on all tested shapes. Original 13.1 binaries reproduce the rerun. Every build reproduces the archive at capacity 0. |
| cuBLAS/runtime version | Not a factor | Both archive and rerun resolved the same 13.2 runtime libraries |
| Rare flag-off-specific divergence | Rejected | Deterministic under reversed admission, also seen with the paired build |
| Target capacity setting | Confirmed | Archive audit, no-capacity reproduction, capacity logit test |
| Admission order (prompt 1 first) | Confirmed for the alternative set | Reverse-stagger control, 78/78 |
