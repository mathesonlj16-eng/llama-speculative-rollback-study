import sys
p = sys.argv[1]
s = open(p).read()
if "SM75_INV_NCOLS" in s:
    sys.exit(0)
old1 = "            ncols_opt = 1;\n"
assert s.count(old1) == 1, s.count(old1)
new1 = (
    "            // SM75 diagnostic: SM75_INV_NCOLS unset -> 1 (PR behaviour), 0 -> J follows the batch, n -> optimise J for n columns\n"
    "            static const char * sm75_inv_ncols = getenv(\"SM75_INV_NCOLS\");\n"
    "            const int64_t sm75_n = sm75_inv_ncols ? atoll(sm75_inv_ncols) : 1;\n"
    "            if (sm75_n != 0) {\n"
    "                ncols_opt = sm75_n;\n"
    "            }\n")
s = s.replace(old1, new1)
old2 = "        J_best = ggml_cuda_mmq_get_J_best(src0->type, ncols_opt, fallback, cc, smpbo, prec_src1, &nthreads_best);\n"
assert s.count(old2) == 1, s.count(old2)
new2 = old2 + (
    "        if (getenv(\"SM75_INV_LOG\")) {\n"
    "            fprintf(stderr, \"SM75 mmq ne11=%lld inv=%d J_best=%d\\n\", (long long) ne11, (int) ctx.batch_invariant, J_best);\n"
    "        }\n")
s = s.replace(old2, new2)
open(p, "w").write(s)
