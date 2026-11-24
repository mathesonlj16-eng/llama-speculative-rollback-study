# Methodology and limitations

## Performance measurements

The historical local reproduction used one slot, context 1024, one 160-token chat prompt, five warmed measurements per arm, and server-reported generation rates. Its Nsight Systems 2025.5.2 comparison cropped at the first target vocabulary-sized logits copy. The speculative window can include draft prefill and trailing draft work. The NVTX shim identifies checkpoint operations and CUDA correlation IDs associate their copies. The historical run differs from the issue report in GPU, model sizes, and target quantization.

The primary ordinary CUDA workload uses six prompts, 48 generated tokens each, seed 7, greedy sampling, fixed token limits, and disabled prompt cache. At two slots the runner submits prompts in pairs. The wall interval starts after tokenization and ends after receiving the last completion. It includes request overhead and prompt evaluation. Summed host API intervals count decode, sync and logits access and separately state get/set. They may overlap and have their own instrumentation overhead. Each series uses its own median. The four-way figure shows the two-repeat profiling series for plain, paired and prototype, and the later three-repeat keep-draft row separately. The README table uses the later shared series wherever available.

The two-slot paired baseline exhibited 34- and 39-restore modes. The five-repeat interleaving series used the 34-restore baseline and supersedes the original 20.8% comparison. The mechanism behind the two modes remains open. The untested cheaper-draft hypothesis depends on both draft cost and acceptance. The source of fixed per-decode cost was not isolated. Endpoint wall time is the outcome measure.

The n-gram series used two repeats per configuration on a paired build. Ordinary runs used 288 generated tokens total. Repetitive records, code, copied text, CSV, YAML and logs used 576. No-draft lookup results are reported as such. The repetitive fixture triggered reference disagreement of up to a 1.78-logit gap for a speculation-only output that matched the other reference. The early 0.5 threshold cannot certify the longer workload.

## Correctness hierarchy

1. Reproduce the abort with context labels and rollback state. The historical stack alone could not tell which context rejected the trim.
2. Replay fixed tokens, control grouping, and use an accepted anchor in the same final microbatch. Include repeated rejection, an out-of-range control and altered rejected content. The draft remains checkpoint-based. A content-independent draft snapshot-index problem has not been excluded for all shapes.
3. Run exact greedy comparison on a path where grouping was verified to be invariant for the tested sequences. Keep plain slot-count and repeat controls. Compare token IDs. Deliberate state mutations verify that corrupt executions can either diverge or fail loudly. The original CPU reference oracle contains 16 mode/slot cells including the two plain cells, plus a plain repeat.
4. Exercise keep-draft cache and seeded sampling separately. The one-slot and corrected two-slot cache cases have different prompt coverage. Sampling compares flag on/off for a single seed and temperature/top-k/top-p combination. Equality with plain under that seed is an observation and supplies no distributional test.

## Numerical analysis and superseded evidence

E2 used different prefixes and was superseded by filtered E3. E3 located a CPU difference at Qcur-15 and E5 row substitution reproduced the other grouping’s output. That site was not re-traced on the later reference-vs-default path. E6 GPU instrumentation changed the margin and is diagnostic only. E7's unanchored refusal cannot be used as evidence of equality. E8 contains the anchored result. E9 varied rejected content.

G1, G2 and G6–G9 large-batch “CPU” comparisons are invalid as CPU-only evidence. Zero GPU layers leaves operation offload available. G10 records CUDA placement for the large operation. G11 hides CUDA, disables operation offload and forces CPU reference kernels. Its 384 comparisons per setting count whole and chunk64 against single-token output for two 96-position sequences. G12 uses default CPU kernels under corrected placement controls. The G7 perturbation figure comes from single-token calls and retains the same Q8_0 weights in both arms. Eight deterministic seeds per nonzero point were tested. The plotted perturbation size is the configured scale and the response is the median RMS across the vocabulary. G11's initial layer multiply also compared rounded and exact activation variants against a double-precision calculation. Rounded max relative error was about 8.25e-4 and exact about 5.67e-8 for that check.

## Scope

The verified results are limited to these fixtures and the staged implementation at the pin. The CPU oracle checks the tested state-management behavior. Production backends can choose shape-dependent kernels and produce different tokens. GPU kernel-level root cause, other quantizations and hardware, longer sessions, context shift, parent/child tasks, multimodal inputs, forced production rollback failure and current upstream remain unverified.

## Strength of evidence

Confirmed findings include the draft rollback refusal, the bounded target replay comparisons, the CPU token equality tests and the measured timings. Draft execution cost as the remaining performance explanation and activation rounding as a source of amplification are supported by the profiling comparison and perturbation intervention, respectively. The cause of the two paired restore counts and the corresponding GPU kernel choices remain hypotheses. Untested configurations are listed above.
