import sys
p = sys.argv[1]
s = open(p).read()
if "SM75_MOE_NOMMF" in s:
    sys.exit(0)
old = ("        if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {\n"
       "            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);\n")
assert s.count(old) == 1, s.count(old)
new = ("        // SM75 diagnostic: SM75_MOE_NOMMF=1 skips MMF in batch-invariant mode, because whether MMF is allowed depends on the token count\n"
       "        static const bool sm75_nommf = getenv(\"SM75_MOE_NOMMF\") != nullptr;\n"
       "        if (!(sm75_nommf && ctx.batch_invariant) &&\n"
       "                ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {\n"
       "            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);\n")
s = s.replace(old, new)
open(p, "w").write(s)
