#!/usr/bin/env bash
# Post-sweep: analysis, corrected dp4a recount (SASS mnemonic is IDP.4A), compile-flag check.
E=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}/evidence-p61; S=$(dirname $0)
cp $S/analyze.py $S/post.sh $E/scripts/
cat $E/STATUS; tail -3 $E/pipeline.log
python3 -I $E/scripts/analyze.py $E/perf/main $E/analysis
CO=/usr/local/cuda-13.2/bin/cuobjdump
echo "build,mmq_kernels,mmq_IMMA,mmq_IDP.4A,all_IDP.4A" > $E/static-check-dp4a.csv
for b in fixed-warp d4f-fw d4f-pr; do L=$(find $D/build/$b ${SM75_ROOT:-$HOME/research/sm75-batch-invariance}/build/$b -name 'libggml-cuda.so*' -type f 2>/dev/null | head -1)
  $CO -sass $L | awk -v b=$b '/Function :/ {mq=($0 ~ /mul_mat_q/); if(mq) nk++} /IMMA/ {if(mq) m1++} /IDP\.4A/ {a++; if(mq) m2++} END{printf "%s,%d,%d,%d,%d\n",b,nk,m1,m2,a}' >> $E/static-check-dp4a.csv; done
cat $E/static-check-dp4a.csv
for b in d4f-fw d4f-pr d4-fw fixed-warp; do C=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}/build/$b/compile_commands.json
  echo "$b: mmq.cu entries with FORCE_MMQ=$(python3 -I -c 'import json,sys; c=[e for e in json.load(open(sys.argv[1])) if e["file"].endswith("ggml-cuda/mmq.cu")]; x=c[0]["command"]; print("-DGGML_CUDA_FORCE_MMQ" in x, "DIAG_NO_TURING_MMA" in x, [t for t in x.split() if "arch" in t or "generate-code" in t][:3])' $C)"; done
