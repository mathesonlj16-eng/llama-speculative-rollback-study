# Reproduction

## Inputs

Use Ubuntu or WSL2 with Git, Bash, CMake, a C++20 compiler and Python 3. CUDA performance requires a compatible NVIDIA device and CUDA toolkit. The archived build used Release, shared libraries, native CPU tuning, GCC 15.2.0, NVIDIA CUDA 13.1.115, a GTX 1650 with 4 GiB (SM 75) and Windows driver 592.00. CMake 4.2.3, Python 3.14.4, Git 2.53.0, glibc 2.43 and the 6.18.58.1 WSL2 kernel were observed in recovered Ubuntu 26.04.1. Figure regeneration was checked with Matplotlib 3.10.0.

| Input | Identifier | SHA-256 |
|---|---|---|
| Target | Ollama `qwen3.5:2b` model layer, Q8_0, 1.88B parameters | `e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7` |
| Draft | `Qwen3.5-0.8B-Q4_K_M.gguf` | `b4e71cb32694733ba40c69e595a0524dfbc2cabe3c28fb7fef99eed4cf8730bf` |

Obtain the model layers separately. Model tags and matching filenames may point to different bytes. The runner checks both digests before starting. The target is available as a content-addressed Ollama model layer. The pinned draft repository's LFS metadata matches the locally verified digest. [Target blob metadata](https://ollama.com/library/qwen3.5:2b-q8_0/blobs/e8bf47d40c5b) · [Pinned draft metadata](https://huggingface.co/axteam/Qwen3.5-0.8B-GGUF/blob/09ad5aaffcab6b7575c030f347beb2635f26cc7d/Qwen3.5-0.8B-Q4_K_M.gguf)

```bash
curl -fL 'https://registry.ollama.com/v2/library/qwen3.5/blobs/sha256:e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7' -o TARGET.gguf
curl -fL 'https://huggingface.co/axteam/Qwen3.5-0.8B-GGUF/resolve/09ad5aaffcab6b7575c030f347beb2635f26cc7d/Qwen3.5-0.8B-Q4_K_M.gguf' -o DRAFT.gguf
sha256sum --check provenance/models.sha256
```

## Build and run

Run from the root of this study on Linux. Use fresh paths for builds and each run. Build scripts disable embedded chat UI assets for the primary experiments. The shell scripts can be invoked with Bash without executable bits.

```bash
export TARGET=/absolute/path/to/target.gguf
export DRAFT=/absolute/path/to/draft.gguf
export STUDY_RUNS=$(mktemp -d)
bash scripts/build.sh "$STUDY_RUNS/cpu" cpu
bash scripts/run.sh "$STUDY_RUNS/cpu" "$STUDY_RUNS/oracle" "$TARGET" "$DRAFT" oracle
bash scripts/run.sh "$STUDY_RUNS/cpu" "$STUDY_RUNS/followups" "$TARGET" "$DRAFT" followups
bash scripts/run.sh "$STUDY_RUNS/cpu" "$STUDY_RUNS/mutations" "$TARGET" "$DRAFT" mutations
bash scripts/run.sh "$STUDY_RUNS/cpu" "$STUDY_RUNS/numerical" "$TARGET" "$DRAFT" numerical
```

```bash
bash scripts/build.sh "$STUDY_RUNS/cuda" cuda
for experiment in profile performance interleaved flagoff admission ngram rollback; do
    bash scripts/run.sh "$STUDY_RUNS/cuda" "$STUDY_RUNS/$experiment" "$TARGET" "$DRAFT" "$experiment"
done
```

The admission control repeats the flag-off comparison while delaying the second request of the first pair. Set `ADMISSION_DELAYS_MS` (default `0,21,100`) and `ADMISSION_REPEATS` (default 3) to change the arms. The server admits requests at its own step boundaries, so the record keeps the measured gap from the server log. The 2026-11-24 rerun used CUDA 13.2.86. Its serial runs matched the archive, while concurrent two-slot token sequences and restore modes differed from the CUDA 13.1 archive. Compare concurrent CUDA outputs against controls from the same run. Keep timed experiments away from concurrent compilation and host sleep. v1.1.0 traced that difference to the target snapshot capacity (`target n_rs_seq=` in each server log), not to the toolkit.

### v1.1.0 toolkit and capacity controls

```bash
# Clean builds with an explicit toolkit (13.1 with GCC 15 needs your own patched crt/math_functions.h, see the script header)
CUDA_HOME=/usr/local/cuda-13.1 CUDA_HEADER_OVERRIDE=/path/to/patched/math_functions.h \
    bash scripts/v1.1/build-cuda131.sh "$STUDY_RUNS/cuda131"
# Same harness on each build root: default capacity shim (target n_rs_seq=3)
for experiment in flagoff admission interleaved reverse; do
    bash scripts/run.sh "$STUDY_RUNS/cuda131" "$STUDY_RUNS/131-$experiment" "$TARGET" "$DRAFT" "$experiment"
done
# No-op shim: the target keeps capacity 0, as in the archived flag-off and interleaved runs
g++ -std=c++20 -O2 -shared -fPIC scripts/v1.1/nocap.cpp -o "$STUDY_RUNS/nocap.so"
CAP_OVERRIDE="$STUDY_RUNS/nocap.so" bash scripts/run.sh "$STUDY_RUNS/cuda" "$STUDY_RUNS/132-flagoff-nocap" "$TARGET" "$DRAFT" flagoff
```

`reverse` delays prompt 0 (slot 0) of the first pair by `ADMISSION_DELAYS_MS` (default `0,10,21,30,50`), so prompt 1 starts alone. `scripts/v1.1/logit-dump.cpp` writes teacher-forced logits for a token TSV in one of five batch shapes (`singles`, `whole`, `c4`, `c8`, `pair4`; set `N_RS` for target capacity). `scripts/v1.1/logit-cmp.cpp` compares two dumps position by position. Build both against the stock build's headers and libraries, then select the library under test with `LD_LIBRARY_PATH`. `scripts/v1.1/admission-timeline.py` classifies concurrent runs by which slot launched first and the gap before the second. The compact results are in `results/v1.1/`. Raw run directories, logs, 13.1 binaries and the analysis scripts are archived outside this repository, with checksums.

### v1.2.0 operator supplement (dp4a MMQ)

The supplement is independent of the server harness. Apply `patches/v1.2/fixed-warp-dp4a-vs-master.patch` to llama.cpp master `22b99122e1dfe1d8a5175bfa2eaa8389699c0b92`. Apply `patches/v1.2/pr32081-dp4a-vs-pr-head.patch`, or the `-pool-` variant for large-N correctness, to the PR #32081 head `3f25ea199a36a9e83480b6dac0941bb82c96c5b9`. Fetch the PR with `git fetch origin pull/32081/head`. Build `test-backend-ops` with `CMAKE_CUDA_ARCHITECTURES=75`. Add `-DGGML_CUDA_DIAG_NO_TURING_MMA` to both CUDA and C++ flags for the dp4a arms, and set `GGML_CUDA_FORCE_MMQ` as listed in [the findings](provenance/v1.2/FINDINGS.md). The scripts in `scripts/v1.2/` expect `SM75_ROOT` (default `$HOME/research/sm75-batch-invariance`) to hold a bare clone at `upstream.git`, worktrees and builds. The findings list their order. Compact results are in `results/v1.2/`. Raw logs, nsys traces and builds are archived outside this repository.

### v1.4.0 batch-invariant replay (PR #32081)

This also runs independently of the server harness. It uses the same `SM75_ROOT` layout as v1.2.0: a bare clone at `upstream.git` with `pull/32081/head` fetched. Steps:
- `TARGET=/abs/target.gguf bash scripts/v1.4/run.sh build`
  - Checks the target digest.
  - Builds libllama at the PR head `3f25ea199a36a9e83480b6dac0941bb82c96c5b9` (sm_75, CUDA 13.2, ccache) in a fresh worktree.
  - Applies `patches/v1.4/replay-tools-batch-invariant-toggle.patch` to copies of the E1 and `logit-dump` tools.
  - Builds a second libllama with `patches/v1.4/kv-n_kv-pinned-vs-pr-head.patch`, recompiling only `llama-kv-cache.cpp`.
- `bash scripts/v1.4/run.sh e1` runs A0/A1, the 11-token fixture with `BI` unset and then set.
- `bash scripts/v1.4/run.sh a3` runs the five teacher-forced shapes on `results/ngram-token-fixtures.tsv`, with the flag off and on.
- `bash scripts/v1.4/run.sh a4` runs the same shapes with the pinned KV length.
- `bash scripts/v1.4/run.sh a5` runs the pair4 control: `singles` at `N_CTX=1024`.

The steps take about 15 min for the build (ccache cold) and about 20 min for all runs. The logit dumps need about 21 GB. Concatenate the `logit-cmp` CSVs with a leading phase column, then run `scripts/v1.4/aggregate.py` to regenerate `results/v1.4/logit-cmp-by-sequence.csv`. The runner is a parameterized copy of the drivers that ran on 2026-11-25. It was checked with `bash -n`, and its patch step was checked against the sources that ran; it has not been rerun end to end.

### v1.5.0 flash-attention fix

Run the v1.4.0 `build` and `e1` steps first. Then:
- `TARGET=/abs/target.gguf bash scripts/v1.5/run.sh build`. This builds `pr-bi-fafix` (PR head plus `patches/v1.5/fa-kv-split-invariant-vs-pr-head.patch`, about 14 min with ccache cold) and the timing tool for both builds.
- `bash scripts/v1.5/run.sh check` runs:
  - E1 flag off and on;
  - the teacher-forced shapes with the flag on, without the KV pin;
  - the default-mode comparison against the unpatched build.
  It takes about 9 min and writes about 12 GB of dumps.
- `bash scripts/v1.5/run.sh bench` runs 20 interleaved timing runs (about 9 min) and prints medians.

`bash scripts/v1.5/run.sh padkv` (v1.5.1) applies `patches/v1.5/test-backend-ops-fa-padkv-vs-pr-head.patch` to a copy of the PR head's `tests/test-backend-ops.cpp`. It compiles that copy once, links it against both builds, and runs `test-backend-ops batch -o FLASH_ATTN_EXT -b CUDA0`. The run takes about a minute and needs no model. Expected: 33/40 on the PR head (exit 1) and 40/40 with the patch.

Aggregate the `logit-cmp` CSVs with `scripts/v1.4/aggregate.py`, using phase `fa`. The runner is a parameterized copy of the drivers that ran; it has not been rerun end to end.

Only one runner per host should use the archived fixed localhost ports at a time. The runner copies the harness to its output directory. Each arm launches a fresh server and records request timing, token IDs and server logs. Look for all requests completing and compare actual token arrays in oracle variants. The mutation runs deliberately include HTTP errors and process aborts. Follow-up output includes the historical odd-length cache attempt and the corrected two-slot run.

The stock build is upstream `787a5612fceef602503e5f0ee6133cdc9b8a85c1`. Apply each of the four patches independently to the pin. The target patch is cumulative with paired checkpoints, and keep-draft is cumulative with both. The exact-activation patch is a separate CPU experiment. The archived prototype used CPU-compiled server/common libraries over a shared CUDA runtime. The build script makes clean variant builds. The original CUDA 13.1/GCC 15 combination required a local, two-declaration CUDA-header exception-specification compatibility adjustment for rsqrt and rsqrtf. Use a supported toolchain pairing or supply an equivalent local toolchain workaround according to your CUDA installation. The upstream llama.cpp changes and NVIDIA headers have separate licensing.

The runner enables the target arm with LLAMA_SPEC_TARGET_CKPT_FREE=1 and the keep arm by adding LLAMA_SPEC_DRAFT_KEEP=1. These switches test environment variable presence. Unset a switch to disable it. The experiments used draft-simple with a snapshot-capable target and a full-only draft.

The CPU reference shim forces the backend reference mode, and reference runners hide CUDA. Confirm the reference marker and absence of CUDA buffers in the logs. The numerical runner exports one-token, 64-token and whole-prompt comparisons and the two perturbation sweeps. For the sweep, the archived single-token perturbations stayed below the GPU offload shape. The release runner additionally hides CUDA. Read [methodology and limits](METHODOLOGY.md) before interpreting throughput and correctness.

## Verify the archive

```bash
sha256sum --check SHA256SUMS
python3 scripts/verify-results.py
python3 -m pip install matplotlib==3.10.0
python3 scripts/figures.py
```

The verifier derives headline aggregates from compact runs and compares archived token arrays. Figure regeneration on another renderer or font installation may change file checksums. Release-manifest checks should run before regeneration. Historical `418fe97b5f01b11eeb89f30d8a0459433edbf1d2` server timing records use a different request and denominator from the prototype figures. Historical tooling lives under [scripts/historical](scripts/historical).


## Historical warmed benchmark

Build the historical revision in a separate checkout. Use the same CUDA architecture and compatible toolchain. Run each server in turn and invoke the matching client command from a separate terminal. Stop the plain server before starting the draft server.

```bash
git clone https://github.com/ggml-org/llama.cpp.git historical-src
git -C historical-src checkout --detach 418fe97b5f01b11eeb89f30d8a0459433edbf1d2
cmake -S historical-src -B historical-build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_BUILD_UI=OFF
cmake --build historical-build --target llama-server --parallel 6
historical-build/bin/llama-server -m "$TARGET" -ngl 99 -c 1024 -np 1 --jinja --host 127.0.0.1 --port 8091
# Separate terminal
python3 scripts/historical/bench_32776.py baseline
# Draft server after stopping the plain server
historical-build/bin/llama-server -m "$TARGET" -ngl 99 -c 1024 -np 1 --jinja --host 127.0.0.1 --port 8091 -md "$DRAFT" --spec-type draft-simple --spec-draft-n-max 3 -ngld 99 -td 4
# Separate terminal
python3 scripts/historical/bench_32776.py spec
```

The historical SQLite reducer accepts baseline, speculative and NVTX-labeled speculative Nsight exports in that order. The compact historical checkpoint table is supplied for checking the report without the exports.

To re-collect the historical checkpoint trace, set NSYS to the Nsight Systems 2025.5.2 executable and CUDA_HOME to the CUDA toolkit path. Run from the study root after building the historical server. Keep generated trace databases outside source control.

```bash
g++ -O2 -shared -fPIC -Ihistorical-src/include -Ihistorical-src/ggml/include -I"$CUDA_HOME/include" scripts/historical/trace32776.cpp -o "$STUDY_RUNS/trace32776.so" -ldl
python3 scripts/historical/profile32776.py "$STUDY_RUNS/nsys-pair" historical-build/bin/llama-server "$TARGET" "$DRAFT" "$NSYS"
python3 scripts/historical/profile32776.py "$STUDY_RUNS/nsys-labeled" historical-build/bin/llama-server "$TARGET" "$DRAFT" "$NSYS" "$STUDY_RUNS/trace32776.so"
python3 scripts/historical/summarize32776.py "$STUDY_RUNS/nsys-pair/baseline.sqlite" "$STUDY_RUNS/nsys-pair/spec.sqlite" "$STUDY_RUNS/nsys-labeled/spec.sqlite"
```
