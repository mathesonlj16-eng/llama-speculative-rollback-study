import sys
p = sys.argv[1]
s = open(p).read()
if "SM75_MOE_F32SORT" in s:
    sys.exit(0)
old = ("    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))\n"
       "        || ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F8_E4M3 ? GGML_TYPE_F32 : src0->type;\n")
assert s.count(old) == 1, s.count(old)
new = ("    // SM75 diagnostic: SM75_MOE_F32SORT=1 sorts src1 as F32 in batch-invariant mode, so each expert's\n"
       "    // ggml_cuda_mul_mat sees F32 src1 and can take its batch-invariant path instead of cuBLAS\n"
       "    static const bool sm75_f32sort = getenv(\"SM75_MOE_F32SORT\") != nullptr;\n"
       "    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))\n"
       "        || (sm75_f32sort && ctx.batch_invariant)\n"
       "        || ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F8_E4M3 ? GGML_TYPE_F32 : src0->type;\n")
s = s.replace(old, new)
open(p, "w").write(s)
