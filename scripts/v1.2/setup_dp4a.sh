#!/usr/bin/env bash
# Create NEW worktrees dp4a-fw / dp4a-pr at the audited commits, re-apply the audited uncommitted diffs,
# then add a macro-guarded diagnostic switch (GGML_CUDA_DIAG_NO_TURING_MMA) that disables the Turing MMA path
# on both host (turing_mma_available) and device (TURING_MMA_AVAILABLE). Without the macro the code is unchanged.
set -e
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; E=$D/evidence-p61; G=$D/upstream.git
[ -e $E ] && { echo "evidence-p61 exists, refusing"; exit 1; }
mkdir -p $E/scripts; cp $(dirname $0)/*.sh $E/scripts/; sed -i 's/\r$//' $E/scripts/*.sh
for p in fw:fixed-warp pr:pr; do n=dp4a-${p%%:*}; src=${p#*:}; W=$D/worktrees/$n
  [ -e $W ] && { echo "$W exists, refusing"; exit 1; }
  git -C $G worktree add --detach $W $(git -C $D/worktrees/$src rev-parse HEAD) > /dev/null
  git -C $D/worktrees/$src diff > $E/base-$src.diff
  [ -s $E/base-$src.diff ] && git -C $W apply $E/base-$src.diff
  c=$W/ggml/src/ggml-cuda/common.cuh
  perl -0pi -e 's/(#if !defined\(GGML_USE_HIP\) && !defined\(GGML_USE_MUSA\) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING)\n#define TURING_MMA_AVAILABLE/$1 && !defined(GGML_CUDA_DIAG_NO_TURING_MMA)\n#define TURING_MMA_AVAILABLE/' $c
  perl -0pi -e 's/(static bool turing_mma_available\(const int cc\) \{\n)/$1#ifdef GGML_CUDA_DIAG_NO_TURING_MMA\n    GGML_UNUSED(cc);\n    return false; \/\/ diagnostic: force the dp4a (Pascal-style) code paths on Turing\n#endif \/\/ GGML_CUDA_DIAG_NO_TURING_MMA\n/' $c
  [ $(grep -c GGML_CUDA_DIAG_NO_TURING_MMA $c) = 3 ] || { echo "patch did not apply cleanly to $c"; exit 1; }
  git -C $W diff > $E/worktree-$n.full.diff
  # switch-only diff = full diff minus the re-applied audited diff
  diff <(git -C $D/worktrees/$src diff -- ggml/src/ggml-cuda/common.cuh) <(git -C $W diff -- ggml/src/ggml-cuda/common.cuh) > $E/switch-$n.delta || true
  echo "$n head=$(git -C $W rev-parse HEAD) base_diff_sha=$(sha256sum < $E/base-$src.diff | cut -c1-16) full_diff_sha=$(sha256sum < $E/worktree-$n.full.diff | cut -c1-16)" >> $E/provenance.txt
done
cat $E/provenance.txt; cat $E/switch-dp4a-fw.delta
