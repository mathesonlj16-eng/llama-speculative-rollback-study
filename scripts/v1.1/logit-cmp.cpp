// v1.1.0: compares two logit dumps from logit-dump (same TSV). Prints one CSV row per predicted position.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
static std::vector<int> ints(const std::string & s) { std::istringstream in(s); std::vector<int> v; int x; while (in >> x) v.push_back(x); return v; }
static void top2(const std::vector<float> & z, int & t, float & margin) {
    t = std::max_element(z.begin(), z.end()) - z.begin(); float s = -INFINITY;
    for (size_t k = 0; k < z.size(); k++) if ((int) k != t) s = std::max(s, z[k]); margin = z[t] - s; }
int main(int argc, char ** argv) {
    if (argc != 5) { fprintf(stderr, "usage: tsv a.bin b.bin label\n"); return 2; }
    std::ifstream in(argv[1]); FILE * fa = fopen(argv[2], "rb"), * fb = fopen(argv[3], "rb"); std::string line;
    printf("label,key,index,gen_token,top_a,top_b,margin_a,margin_b,gen_gap_a,gen_gap_b,max_abs_diff,rms_diff\n");
    while (std::getline(in, line)) {
        auto a = line.find('\t'), b = line.find('\t', a + 1); std::string key = line.substr(0, a); auto gen = ints(line.substr(b + 1));
        int32_t ha[2], hb[2]; if (fread(ha, sizeof ha, 1, fa) != 1 || fread(hb, sizeof hb, 1, fb) != 1 || ha[0] != hb[0] || ha[1] != hb[1]) return 3;
        std::vector<float> za(ha[1]), zb(ha[1]);
        for (int i = 0; i < ha[0]; i++) {
            if (fread(za.data(), 4, ha[1], fa) != (size_t) ha[1] || fread(zb.data(), 4, ha[1], fb) != (size_t) ha[1]) return 4;
            double mx = 0, sq = 0; for (int k = 0; k < ha[1]; k++) { double d = za[k] - zb[k]; mx = std::max(mx, std::fabs(d)); sq += d * d; }
            int ta, tb; float ma, mb; top2(za, ta, ma); top2(zb, tb, mb);
            printf("%s,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6g,%.6g\n", argv[4], key.c_str(), i, gen[i], ta, tb, ma, mb,
                   za[ta] - za[gen[i]], zb[tb] - zb[gen[i]], mx, std::sqrt(sq / ha[1]));
        }
    }
}
