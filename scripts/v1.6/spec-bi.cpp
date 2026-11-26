// Greedy speculative decoding: draft model proposes k tokens, target verifies them in one batch (logits for every
// row). Baseline (k = 0): plain target decoding, one token per decode, in the same context.
// For every generated token, records the token and an FNV-1a hash of the full target logits row it was chosen from.
// usage: target draft label kset(e.g. 0,1,2,3,4,5,7) tsv
// env: BI=1 batch-invariant (both models); TRB=rs|ckpt target rollback; G = tokens per prompt (default 64)
//   rs:   target ctx n_rs_seq = 8, rejected tail removed with llama_memory_seq_rm (recurrent snapshots)
//   ckpt: target ctx n_rs_seq = 0, recurrent state saved before verify; on rejection restored, KV tail removed and
//         the accepted tokens re-decoded (their last logits row is checked against the verify row: "redecode_bad")
// draft: always ckpt-style (restore + feed the accepted tokens with the next draft batch), n_rs_seq = 0.
// stdout CSV: label,bi,trb,k,seq,n_gen,ms,rounds,accepted,drafted,redecode_bad ; stderr: log
// tokens file: <label>.tok  lines "k seq tok:hash ..."
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<int> ints(const std::string & s, char sep = ' ') { std::vector<int> v; std::string t; std::istringstream in(s);
    while (std::getline(in, t, sep)) if (!t.empty()) v.push_back(atoi(t.c_str())); return v; }
static uint64_t fnv(const float * z, int n) { uint64_t h = 1469598103934665603ull; const uint8_t * p = (const uint8_t *) z;
    for (size_t i = 0; i < (size_t) n*4; ++i) { h ^= p[i]; h *= 1099511628211ull; } return h; }
static int argmax(const float * z, int n) { int b = 0; for (int i = 1; i < n; ++i) if (z[i] > z[b]) b = i; return b; }
static void die(const char * m) { fprintf(stderr, "FATAL %s\n", m); exit(3); }

struct dec_out { std::vector<int> idx; };
// decode tokens at pos0.. ; out[i] = want logits for row i; returns batch indices of output rows
static std::vector<int> decode(llama_context * c, const std::vector<int> & toks, int pos0, const std::vector<bool> & out) {
    auto * b = llama_batch_ext_init(c); std::vector<int> idx(toks.size(), -1);
    for (size_t j = 0; j < toks.size(); j++) { const int p = pos0 + (int) j;
        int i = llama_batch_ext_add_token(b, 0, toks[j]);
        if (i < 0 || !llama_batch_ext_set_pos(b, i, &p) || !llama_batch_ext_set_output_logits(b, i, out[j])) die("batch");
        idx[j] = i; }
    if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) die("decode");
    llama_batch_ext_free(b);
    return idx;
}
struct ckpt { std::vector<uint8_t> buf; };
static const llama_state_seq_flags CF = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
static void save(llama_context * c, ckpt & k) { const size_t n = llama_state_seq_get_size_ext(c, 0, CF); k.buf.resize(n);
    if (llama_state_seq_get_data_ext(c, k.buf.data(), n, 0, CF) != n) die("save"); }
static void load(llama_context * c, ckpt & k) { if (llama_state_seq_set_data_ext(c, k.buf.data(), k.buf.size(), 0, CF) == 0) die("load"); }

int main(int argc, char ** argv) {
    if (argc != 6) { fprintf(stderr, "usage: target draft label kset tsv\n"); return 2; }
    const bool bi = getenv("BI") != nullptr;
    const std::string trb = getenv("TRB") ? getenv("TRB") : "rs";
    const int G = getenv("G") ? atoi(getenv("G")) : 64;
    const std::vector<int> kset = ints(argv[4], ',');
    std::vector<std::vector<int>> prompts; std::vector<int> filler; { std::ifstream in(argv[5]); std::string line;
        while (std::getline(in, line)) { auto a = line.find('\t'), b = line.find('\t', a + 1);
            prompts.push_back(ints(line.substr(a + 1, b - a - 1)));
            auto f1 = ints(line.substr(a + 1, b - a - 1)), f2 = ints(line.substr(b + 1));
            filler.insert(filler.end(), f1.begin(), f1.end()); filler.insert(filler.end(), f2.begin(), f2.end()); } }
    // PREFIX=n: prepend n filler tokens (fixture prompt+gen tokens, cycled) to every prompt, for a deeper context
    const int PREFIX = getenv("PREFIX") ? atoi(getenv("PREFIX")) : 0;
    if (PREFIX > 0) for (auto & pr : prompts) { std::vector<int> v; for (int i = 0; i < PREFIX; ++i) v.push_back(filler[i % filler.size()]);
        v.insert(v.end(), pr.begin(), pr.end()); pr = v; }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto * mt = llama_model_load_from_file(argv[1], mp); auto * md = llama_model_load_from_file(argv[2], mp);
    if (!mt || !md) return 1;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(mt));
    if (nv != llama_vocab_n_tokens(llama_model_get_vocab(md))) die("vocab");
    auto cp = llama_context_default_params();
    // 4 GB card holding both models: at most 8 logit rows per batch (k + 1 <= 8), 256-token prefill ubatches
    cp.n_ctx = PREFIX > 0 ? 2048 : 512; cp.n_batch = 512; cp.n_ubatch = 256; cp.n_outputs_max = 8; cp.n_seq_max = 1; cp.n_threads = cp.n_threads_batch = 8; cp.batch_invariant = bi;
    auto cpt = cp; cpt.n_rs_seq = trb == "rs" ? 8 : 0;
    auto cpd = cp; cpd.n_rs_seq = 0;
    auto * ct = llama_init_from_model(mt, cpt); auto * cd = llama_init_from_model(md, cpd);
    if (!ct || !cd) die("ctx");
    FILE * tf = fopen((std::string(argv[3]) + ".tok").c_str(), "w");
    printf("label,bi,trb,prefix,k,seq,n_gen,ms,rounds,accepted,drafted,redecode_bad,ms_draft,ms_verify,ms_ckpt,ms_rollback\n");
    ckpt kt, kd;
    // negative k = warm-up pass with |k| on prompt 0, not reported
    for (int kraw : kset) for (size_t s = 0; s < prompts.size(); ++s) {
        const bool warm = kraw < 0; const int k = warm ? -kraw : kraw;
        if (warm && s > 0) continue;
        double t_dr = 0, t_vf = 0, t_ck = 0, t_rb = 0, tm;
        const auto & pr = prompts[s]; const int n = (int) pr.size();
        llama_memory_clear(llama_get_memory(ct), true); llama_memory_clear(llama_get_memory(cd), true);
        for (int p0 = 0; p0 < n - 1; p0 += 512) {   // prefill prompt[0..n-2] in chunks of 512
            std::vector<int> pp(pr.begin() + p0, pr.begin() + std::min(n - 1, p0 + 512));
            decode(ct, pp, p0, std::vector<bool>(pp.size(), false)); if (k > 0) decode(cd, pp, p0, std::vector<bool>(pp.size(), false)); }
        llama_synchronize(ct); llama_synchronize(cd);
        std::vector<int> toks; std::vector<uint64_t> hs; double thash = 0;
        long rounds = 0, acc = 0, drafted = 0, rbad = 0;
        auto rec = [&](const float * z) { const double t0 = now_ms(); hs.push_back(fnv(z, nv)); thash += now_ms() - t0; };
        int last = pr[n - 1], p = n - 1;
        const double t0 = now_ms();
        if (k == 0) {
            while ((int) toks.size() < G) {
                auto idx = decode(ct, {last}, p, {true});
                const float * z = llama_get_logits_ith(ct, idx[0]); const int g = argmax(z, nv); rec(z);
                toks.push_back(g); last = g; ++p;
            }
        } else {
            std::vector<int> P = {last}; int q = p;   // draft: positions < q filled, P fed at q.., P.back() == last at p
            while ((int) toks.size() < G) {
                ++rounds;
                tm = now_ms(); save(cd, kd); t_ck += now_ms() - tm;
                std::vector<int> d;
                tm = now_ms();
                { std::vector<bool> o(P.size(), false); o.back() = true;
                  auto idx = decode(cd, P, q, o); d.push_back(argmax(llama_get_logits_ith(cd, idx.back()), nv)); }
                for (int i = 1; i < k; ++i) { auto idx = decode(cd, {d.back()}, p + i, {true}); d.push_back(argmax(llama_get_logits_ith(cd, idx[0]), nv)); }
                t_dr += now_ms() - tm;
                drafted += k;
                if (trb == "ckpt") { tm = now_ms(); save(ct, kt); t_ck += now_ms() - tm; }
                tm = now_ms();
                std::vector<int> vb = {last}; vb.insert(vb.end(), d.begin(), d.end());
                auto idx = decode(ct, vb, p, std::vector<bool>(vb.size(), true));
                int a = 0; std::vector<int> g(vb.size());
                for (size_t i = 0; i < vb.size(); ++i) g[i] = argmax(llama_get_logits_ith(ct, idx[i]), nv);
                t_vf += now_ms() - tm;
                while (a < k && g[a] == d[a]) ++a;
                for (int i = 0; i <= a; ++i) { rec(llama_get_logits_ith(ct, idx[i])); toks.push_back(g[i]); }
                acc += a;
                const int bonus = g[a];
                tm = now_ms();
                if (a < k) {
                    if (trb == "rs") { if (!llama_memory_seq_rm(llama_get_memory(ct), 0, p + a + 1, -1)) die("seq_rm rs"); }
                    else {
                        const uint64_t want = hs.back();
                        load(ct, kt); if (!llama_memory_seq_rm(llama_get_memory(ct), 0, p, -1)) die("seq_rm tgt");
                        std::vector<int> rb(vb.begin(), vb.begin() + a + 1); std::vector<bool> o(rb.size(), false); o.back() = true;
                        auto ri = decode(ct, rb, p, o);
                        const double th = now_ms(); if (fnv(llama_get_logits_ith(ct, ri.back()), nv) != want) ++rbad; thash += now_ms() - th;
                    }
                    load(cd, kd); if (!llama_memory_seq_rm(llama_get_memory(cd), 0, q, -1)) die("seq_rm dft");
                    P.insert(P.end(), d.begin(), d.begin() + a); P.push_back(bonus);      // q unchanged
                } else {
                    P = {d.back(), bonus}; q = p + k;
                }
                t_rb += now_ms() - tm;
                last = bonus; p += a + 1;
            }
        }
        llama_synchronize(ct); llama_synchronize(cd);
        const double ms = now_ms() - t0 - thash;
        if (warm) continue;
        printf("%s,%d,%s,%d,%d,%zu,%d,%.3f,%ld,%ld,%ld,%ld,%.3f,%.3f,%.3f,%.3f\n", argv[3], bi, trb.c_str(), PREFIX, k, s, G, ms, rounds, acc, drafted, rbad,
               t_dr, t_vf, t_ck, t_rb); fflush(stdout);
        fprintf(tf, "%d %zu", k, s); for (int i = 0; i < G; ++i) fprintf(tf, " %d:%016llx", toks[i], (unsigned long long) hs[i]); fprintf(tf, "\n");
    }
    fclose(tf);
    llama_free(ct); llama_free(cd); llama_model_free(mt); llama_model_free(md); llama_backend_free();
}
