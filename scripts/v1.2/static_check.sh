#!/usr/bin/env bash
# Static evidence: tensor-core MMA (IMMA/HMMA) vs dp4a (IDP4A) instructions in SASS, library-wide and inside MMQ kernels only.
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; E=$D/evidence-p61
CO=/usr/local/cuda-13.2/bin/cuobjdump
echo "build,elf_archs,all_IMMA,all_HMMA,all_IDP4A,mmq_kernels,mmq_IMMA,mmq_HMMA,mmq_IDP4A" > $E/static-check.csv
for b in fixed-warp pr d4f-fw d4f-pr d4-fw; do
  L=$(find $D/build/$b -name 'libggml-cuda.so*' -type f | head -1); [ -z "$L" ] && { echo "$b,no lib" >> $E/static-check.csv; continue; }
  elf=$($CO --list-elf $L 2>/dev/null | grep -oE 'sm_[0-9]+' | sort | uniq -c | tr -s ' ' | tr '\n' ' ')
  $CO -sass $L 2>/dev/null | awk -v b=$b -v elf="$elf" '
    /Function :/ { mq = ($0 ~ /mul_mat_q/); if (mq) nk++ }
    /IMMA/ { a1++; if (mq) m1++ } /HMMA/ { a2++; if (mq) m2++ } /IDP4A/ { a3++; if (mq) m3++ }
    END { printf "%s,%s,%d,%d,%d,%d,%d,%d,%d\n", b, elf, a1, a2, a3, nk, m1, m2, m3 }' >> $E/static-check.csv
done
