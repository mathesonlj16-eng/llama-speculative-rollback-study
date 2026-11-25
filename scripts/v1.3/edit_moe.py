import sys
p = sys.argv[1]
s = open(p).read()
if "SM75_MOE_TYPES" in s:
    sys.exit(0)
old = ("static void init_mul_mat_id_ids(ggml_context * ctx, int n_mats, bool skip = false) {\n\n"
       "    std::random_device rd;\n    std::default_random_engine rng(rd());\n")
assert s.count(old) == 1
new = ("static void init_mul_mat_id_ids(ggml_context * ctx, int n_mats, bool skip = false) {\n\n"
       "    std::random_device rd;\n"
       "    // SM75 local patch: SM75_SEED also makes the MoE routing reproducible\n"
       "    static std::atomic<unsigned> sm75_ids_ctr{0};\n"
       "    const char * sm75_seed_ids = getenv(\"SM75_SEED\");\n"
       "    std::default_random_engine rng(sm75_seed_ids ? (unsigned) (atoi(sm75_seed_ids) + 7919u*sm75_ids_ctr++) : rd());\n")
s = s.replace(old, new)
old2 = "    // SM75 local patch: SM75_BATCH_TYPES/M/K/N select an explicit MUL_MAT matrix instead of the upstream list\n"
assert s.count(old2) == 1
new2 = ("    // SM75 local patch: SM75_MOE_TYPES/SM75_MOE_N select MUL_MAT_ID cases with 8 experts, 2 used, so tokens per expert grow with N\n"
        "    if (const char * ty = getenv(\"SM75_MOE_TYPES\")) {\n"
        "        for (ggml_type t : sm75_types(ty))\n"
        "        for (int64_t n : sm75_ints(getenv(\"SM75_MOE_N\"))) {\n"
        "            test_cases.emplace_back(new test_mul_mat_id_batch_inv(t, 8, 2, 2048, 768, n, false, false));\n"
        "            if (n <= 256) {\n"
        "                test_cases.emplace_back(new test_mul_mat_id_batch_inv(t, 8, 2, 768, 2048, n, true, false));\n"
        "                test_cases.emplace_back(new test_mul_mat_id_batch_inv(t, 8, 2, 768, 2048, n, true, true));\n"
        "            }\n"
        "        }\n"
        "        return test_cases;\n"
        "    }\n\n") + old2
s = s.replace(old2, new2)
open(p, "w").write(s)
