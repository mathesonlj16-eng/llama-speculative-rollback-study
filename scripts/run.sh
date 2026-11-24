#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root=$(realpath "$1")
out=$(realpath -m "$2")
export STUDY_TARGET=$(realpath "$3")
export STUDY_DRAFT=$(realpath "$4")
experiment=$5
unset LLAMA_SPEC_TARGET_CKPT_FREE LLAMA_SPEC_DRAFT_KEEP LLAMA32776_TARGET_ROLLBACK Q35_TARGET_RS
printf '%s  %s\n' e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7 "$STUDY_TARGET" \
    b4e71cb32694733ba40c69e595a0524dfbc2cabe3c28fb7fef99eed4cf8730bf "$STUDY_DRAFT" | sha256sum --check
mkdir "$out"
cp -R "$repo/scripts/archive/." "$out/"
export STUDY_BASE="$out/base" STUDY_PROTO="$out/proto" STUDY_FLAGOFF="$out/flagoff"
mkdir "$STUDY_BASE" "$STUDY_PROTO" "$STUDY_FLAGOFF" "$out/events"
for pair in "base/bin stock" "base/fixed paired" "proto/bin prototype" "proto/bin-keep keep" \
    "flagoff/bin stock" "flagoff/fixed prototype"; do
    read -r alias variant <<< "$pair"
    ln -s "$root/build-$variant/bin" "$out/$alias"
done
python3 - "$out/evidence" <<'PY'
import json, os, sys
from pathlib import Path
for path in Path(sys.argv[1]).glob('*-command.json'):
    argv = json.loads(path.read_text())
    argv[argv.index('-m') + 1] = os.environ['STUDY_TARGET']
    if '-md' in argv:
        argv[argv.index('-md') + 1] = os.environ['STUDY_DRAFT']
    path.write_text(json.dumps(argv))
PY
src="$root/src-stock"
lib="$root/build-stock/bin"
compile=(g++ -std=c++20 -O2 -I"$src/include" -I"$src/ggml/include" -I"$src/src" -L"$lib" -Wl,-rpath-link,"$lib")
"${compile[@]}" -shared -fPIC "$repo/scripts/fixtures/cap.cpp" -o "$out/cap.so" -ldl
"${compile[@]}" -shared -fPIC "$out/gap-trace/useref.cpp" -o "$out/useref.so" -lggml-cpu -lggml-base -ldl
"${compile[@]}" -shared -fPIC "$out/slowdown/prof.cpp" -o "$out/prof.so" -ldl -pthread
case "$experiment" in
    oracle)
        python3 "$out/exact-oracle/oracle-eval.py" "$out/useref.so" "$out/cap.so"
        ;;
    followups)
        python3 "$out/exact-oracle/oracle-eval.py" "$out/useref.so" "$out/cap.so"
        python3 "$out/exact-oracle/keep-checks.py" "$out/useref.so" "$out/cap.so"
        python3 "$out/exact-oracle/keep-checks-b2.py" "$out/useref.so" "$out/cap.so"
        ;;
    mutations)
        python3 "$out/exact-oracle/oracle-eval.py" "$out/useref.so" "$out/cap.so"
        "${compile[@]}" -shared -fPIC "$out/exact-oracle/break-rollback.cpp" -o "$out/break-trim.so" -ldl
        "${compile[@]}" -shared -fPIC "$out/exact-oracle/break-restore.cpp" -o "$out/break-restore.so" -ldl
        python3 "$out/exact-oracle/mutation.py" "$out/useref.so" "$out/cap.so" "$out/break-trim.so"
        python3 "$out/exact-oracle/mutation-restore.py" "$out/useref.so" "$out/cap.so" "$out/break-restore.so"
        ;;
    profile) python3 "$out/slowdown/profile.py" "$out/prof.so" "$out/events";;
    performance) python3 "$out/slowdown/keep-eval.py" "$out/prof.so" "$out/events";;
    interleaved) python3 "$out/target-ckpt-free/interleaved-throughput.py" "$out/cap.so";;
    flagoff) python3 "$out/target-ckpt-free/flag-off-control.py" "$out/cap.so";;
    admission) python3 "$repo/scripts/fixtures/admission-control.py" "$out/target-ckpt-free" "$out/cap.so";;
    ngram) python3 "$out/ngram/ngram-eval.py" "$out/cap.so";;
    rollback)
        for fixture in rollback-anchored rollback-controls; do
            "${compile[@]}" "$out/fixed-token-replay/$fixture.cpp" -o "$out/$fixture" -lllama -lggml -lggml-base
        done
        run_rollback() {
            local fixture=$1 label=$2 model=$3 layers=$4
            LD_LIBRARY_PATH="$lib" "$out/$fixture" "$model" "$layers" "$label" \
                > "$out/$fixture-$label.csv" 2> "$out/$fixture-$label.log"
        }
        run_rollback rollback-anchored target-cuda "$STUDY_TARGET" 99
        run_rollback rollback-anchored target-cpu "$STUDY_TARGET" 0
        run_rollback rollback-anchored draft-cuda "$STUDY_DRAFT" 99
        run_rollback rollback-controls draft-cuda "$STUDY_DRAFT" 99
        run_rollback rollback-controls draft-cpu "$STUDY_DRAFT" 0
        run_rollback rollback-controls target-cuda "$STUDY_TARGET" 99
        ;;
    numerical)
        for name in sweep-g7 gap-g11; do
            "${compile[@]}" "$repo/scripts/fixtures/$name.cpp" -o "$out/$name" -lllama -lggml -lggml-base
        done
        fixture="$repo/results/ngram-token-fixtures.tsv"
        for variant in stock exact; do
            runtime="$root/build-$variant/bin"
            env CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH="$runtime" LD_PRELOAD="$out/useref.so" \
                "$out/sweep-g7" "$STUDY_TARGET" "$fixture" > "$out/sweep-$variant.csv" 2> "$out/sweep-$variant.log"
            python3 "$out/gap-trace/summarize-sweep.py" "$out/sweep-$variant.csv" > "$out/sweep-$variant-summary.csv"
            env CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH="$runtime" LD_PRELOAD="$out/useref.so" \
                "$out/gap-g11" "$STUDY_TARGET" 0 8 f16 auto "$variant" "$fixture" > "$out/gap-$variant.csv" 2> "$out/gap-$variant.log"
        done
        env CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH="$lib" \
            "$out/gap-g11" "$STUDY_TARGET" 0 8 f16 auto default "$fixture" > "$out/gap-default.csv" 2> "$out/gap-default.log"
        ;;
    *) echo "Unknown experiment $experiment" >&2; exit 2;;
esac
