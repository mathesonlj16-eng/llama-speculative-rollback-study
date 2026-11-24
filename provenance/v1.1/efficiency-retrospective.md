# v1.1.0 computational efficiency retrospective

Host: AMD Ryzen 7 5800HS (16 threads), WSL2 with 7.6 GiB RAM, GTX 1650 4 GiB (SM 75). Timestamps are local (EST) on 24 November 2026, taken from build, phase and server logs in the evidence archive.

## Wall-clock budget

| Interval | Start–end | Minutes | What ran |
|---|---|---:|---|
| Inventory and design | 09:05–09:11 | ~6 | Read-only inspection of builds, toolkits, archive |
| CUDA 13.1 builds | 09:11:20–10:06:45 | 55.4 | stock 18.3, paired 18.7, prototype 18.5 (`-j6`) |
| Device-code analysis (overlapped with builds) | ~09:31–~10:00 | ~30 CPU | `cuobjdump -sass` of three 85 MB libraries, per-kernel comparison |
| Phase 1: flag-off, admission, interleaved on 3 builds | 10:07:16–10:58:05 | 50.8 | 120 server runs |
| Idle: queue deadlock | 10:58:05–11:19:47 | 21.7 | Nothing (see redundancy 3) |
| Numerical: 17 logit dumps + comparisons | 11:19:47–11:25:23 | 5.6 | Teacher-forced CUDA logits |
| Phase 2: archived and no-capacity shims, linker trace | 11:25:32–11:58:17 | 32.8 | 72 server runs + 2 traced starts |
| Phase 3: capacity logit dumps | 12:00:29–12:02:38 | 2.2 | 8 logit dumps |
| Phase 4: reverse-stagger control | 12:02:39–12:39:25 | 36.8 | 90 server runs |
| Phase 5: interleaved at paired capacity 0 | 12:39:26–12:47:34 | 8.1 | 20 server runs |
| Collection, archive, documentation | 12:47–~13:20 | ~33 | CPU and I/O only |

**Total: about 4 h 15 min (09:05 to ~13:20).** Compilation took 55 min. GPU experiments occupied 160 min of wall time, of which 139 min were active. The remaining ~60 min was inventory, analysis and writing.

## Work performed

- **Builds:** 3 clean CUDA 13.1 builds (stock, paired, prototype). The retained CUDA 13.2 builds and the original 13.1 binaries were reused without rebuilding. A few seconds of small compiles ran per experiment (preload shims, logit fixtures).
- **Experiment steps:** 20, all exit status 0.
- **Server runs:** 302 (198 concurrent two-slot and 54 serial in the flag-off/admission/reverse controls, plus 50 interleaved). There were also 2 linker-trace starts, 25 logit dumps and 2 fixture smoke tests.
- **Data written:** 19 GB in `/var/tmp/llama-v11`, of which 11 GB is full-text SASS dumps, 6.9 GB raw logit dumps and 1.5 GB the 13.1 build tree. The durable archive is 181 MB.

## Where the time went inside server runs

Across the 302 runs (server log timestamps):

| Component | Median per run | Total |
|---|---:|---:|
| Target model load (2.0 GB GGUF from `/mnt/c`, 9P) | 12.2 s | 64.1 min |
| Server start to listening (includes the above) | 12.8 s | 67.5 min |
| Measured request wall (the experiment's actual payload) | 8.45 s | 42.9 min |
| Server lifetime | 21.6 s | 111.3 min |

The draft model (0.54 GB on the WSL ext4 disk) loaded in about 0.24 s (~2.2 GB/s), while the target loaded at about 165 MB/s over the 9P mount. **Loading the target model took more time than the measured experiments.**

## Bottlenecks

- **Filesystem (measured):** The 9P model load was the largest single cost, at 58% of server lifetime, 40% of GPU-phase wall time and 46% of its active time.
- **GPU (measured, not a bottleneck):** Every server used 2,801 of 4,096 MiB. Generation occupied 42.9 of 139 active minutes (~31%). The GPU sat idle during model loads, harness setup and the deadlock.
- **CPU (partially measured):** Builds took ~18.5 min per variant at `-j6` on 16 threads. The job count was kept at the v1.0.0 value for parity and to stay within WSL's 7.6 GiB. Per-process memory wasn't recorded, so whether a higher `-j` would fit is unverified. The SASS parsing ran concurrently with the builds and probably slowed them somewhat (not measured).
- **Memory:** No out-of-memory events. 6.6 GiB was available at the end, with most of the used memory being page cache.

## Redundant or avoidable computation

1. **Identical CUDA code compiled three times (~37 min).** The patches touch only `tools/server` and `common`, so `libggml-cuda` and `libllama` are source-identical across variants. One CUDA build plus incremental server/common rebuilds per variant (~2–3 min each) would have produced equivalent binaries. Clean builds were kept to mirror `scripts/build.sh` exactly.
2. **Model loading over 9P (~58 min avoidable).** Copying the target model once to ext4 (about 15 s) would cut each load to roughly 1 s.
3. **Queue deadlock (21.7 min).** The keep-alive watcher's own command line contained the phase script names, so the next phase's `pgrep -f` saw a "running" phase. Fixed by using a self-excluding pattern.
4. **Ordering (~40–50 min of GPU time, in hindsight).** The cause was found by a zero-compute audit of archived server logs (`target n_rs_seq`) and confirmed by one no-capacity arm. Running the original-binary rerun and the log audit first would have made the admission arms on the 13.1 builds (~14 min) unnecessary. It would also have reduced the archived-shim arms (~16 min) to one arm. Those arms remain useful as replication, but weren't decisive.
5. **Bulk intermediates (~18 GB written).** Full SASS text and raw logit dumps were written to disk and compared afterwards. Streaming hashes or in-process comparison would avoid most of this. The files are retained (hashes in `archive/2026-11-24-v1.1/v11-large-regenerable-files.sha256`) because they're regenerable evidence.
6. **Not redundant:** A fresh server per run is deliberate isolation for the admission experiments, and is worth keeping.

## Cloud hardware versus local workflow changes

- **GPU experiments:** These results are specific to SM 75 kernels, the SM count (MMQ stream-k decomposition depends on it) and this driver. A different cloud GPU would change kernel selection and token outputs, and so wouldn't answer this study's question. Even an SM 75 instance (T4) differs in SM count and clocks. Cloud GPUs offer little benefit and add reproducibility risk.
- **Compilation:** A 32–64 vCPU cloud machine might cut the three builds from 55 min to ~10–15 min. Local changes get most of that for free: one CUDA build plus incremental variants (~25 min), ccache (not installed) for repeat builds, and possibly a higher `-j` if memory allows.
- **Estimate:** Local changes alone (model on ext4, incremental variant builds, a working queue, decisive-first ordering) would have shortened compute from ~215 min (55 build + 160 GPU phase) to roughly 80–95 min, with the same evidence. A cloud build machine would save perhaps a further 10–15 min, at a cost and with toolchain drift risk. **Recommendation: optimize the local workflow. Don't move this study to the cloud.**
