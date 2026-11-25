# Arm definitions shared by dispatch, correctness and perf scripts. Usage: exe ARM MODE (MODE = perf|batch)
#   s75*  : audited sm_75 builds (Turing MMA MMQ path)          f*/n* : dp4a-switch builds (same toolkit, same sm_75 target)
#   m0 default dispatch, m1 fixed warp, m2 fixed warp + chunked MMVQ (GGML_CUDA_MMVQ_INV); pron = PR batch-invariant (forced MMQ)
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; B=$D/build
exe() { local a=$1 m=$2 inv=
  [ "$m" = perf ] && inv=1
  case $a in
    s75m0)   GGML_CUDA_MMVQ_INV=0 $B/fixed-warp/bin/test-backend-ops $m -b CUDA0 -o MUL_MAT;;
    s75m2)   GGML_CUDA_MMVQ_INV=2 $B/fixed-warp/bin/test-backend-ops $m -b CUDA0 -o MUL_MAT;;
    s75pron) env ${inv:+SM75_PERF_INVARIANT=1} $B/pr/bin/test-backend-ops $m -b CUDA0 -o MUL_MAT;;
    fm0)     GGML_CUDA_MMVQ_INV=0 $B/d4f-fw/bin/test-backend-ops $m -b CUDA0 -o MUL_MAT;;
    fm1)     GGML_CUDA_MMVQ_INV=1 $B/d4f-fw/bin/test-backend-ops $m -b CUDA0 -o MUL_MAT;;
    fm2)     GGML_CUDA_MMVQ_INV=2 $B/d4f-fw/bin/test-backend-ops $m -b CUDA0 -o MUL_MAT;;
    fpron)   env ${inv:+SM75_PERF_INVARIANT=1} $B/d4f-pr/bin/test-backend-ops $m -b CUDA0 -o MUL_MAT;;
    nm0)     GGML_CUDA_MMVQ_INV=0 $B/d4-fw/bin/test-backend-ops $m -b CUDA0 -o MUL_MAT;;
    *) echo "unknown arm $a"; return 99;;
  esac; }
busy() { pgrep -f "c[c]1plus|cmTC_|nv[c]c |[c]icc|[p]txas|llvm-12134[4]|make -j14 llvm-mc[a]" > /dev/null; }
