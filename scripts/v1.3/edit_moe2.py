import sys
p = sys.argv[1]
s = open(p).read()
old = ("            test_cases.emplace_back(new test_mul_mat_id_batch_inv(t, 8, 2, 2048, 768, n, false, false));\n"
       "            if (n <= 256) {\n"
       "                test_cases.emplace_back(new test_mul_mat_id_batch_inv(t, 8, 2, 768, 2048, n, true, false));\n"
       "                test_cases.emplace_back(new test_mul_mat_id_batch_inv(t, 8, 2, 768, 2048, n, true, true));\n"
       "            }\n")
if old not in s:
    assert "SM75 MoE v2" in s; sys.exit(0)
new = ("            // SM75 MoE v2: serial half adds nodes per token, keep graphs under 8192 nodes\n"
       "            if (n <= 256) test_cases.emplace_back(new test_mul_mat_id_batch_inv(t, 8, 2, 2048, 768, n, false, false));\n"
       "            if (n <= 256) test_cases.emplace_back(new test_mul_mat_id_batch_inv(t, 4, 2, 2048, 768, n, false, false));\n"
       "            if (n <= 128) test_cases.emplace_back(new test_mul_mat_id_batch_inv(t, 8, 2, 768, 2048, n, true, false));\n"
       "            if (n <= 64)  test_cases.emplace_back(new test_mul_mat_id_batch_inv(t, 8, 2, 768, 2048, n, true, true));\n")
s = s.replace(old, new)
open(p, "w").write(s)
