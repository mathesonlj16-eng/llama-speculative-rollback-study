# Changelog

## v1.1.0 (2026-11-24)

Investigates whether the CUDA toolkit change (13.1.115 to 13.2.86) explains the concurrent token and restore-count differences reported in v1.0.0. It doesn't.

- **Added** `provenance/v1.1/FINDINGS.md`: the differences came from the paired and flag-off servers' target snapshot capacity (0 in the archived flag-off and interleaved runs, 3 in the rerun). Nonzero capacity changes two-sequence microbatch composition. Every build reproduces the archive at capacity 0.
- **Added** the toolkit controls: clean CUDA 13.1 builds (device code identical to the archived 13.1 library), the original archived binaries rerun, and teacher-forced logits that are bit-identical between 13.1 and 13.2 in five batch shapes.
- **Added** a reverse-stagger admission control. The rerun's single unexplained flag-off output reproduces whenever prompt 1 is admitted at least 15 ms before prompt 0, for paired and flag-off builds alike.
- **Added** like-for-like interleaved series with paired at capacity 0: 13.9% (13.2) and 16.2% (13.1) median two-slot wall reduction for the prototype, consistent with the archived 15.3%.
- **Added** `provenance/v1.1/efficiency-retrospective.md`, `results/v1.1/`, `scripts/v1.1/`, and v1.1.0 checks in `scripts/verify-results.py`.
- **Changed** `scripts/run.sh` (backward compatible): optional `CAP_OVERRIDE` to replace the capacity preload, and a `reverse` experiment.
- **Changed** the README, METHODOLOGY and REPRODUCE statements that described the 34/39 modes, the toolkit hypothesis and the unexplained flag-off run as open.

## v1.0.0 (2026-11-24)

Initial publication. Tag `v1.0.0`, commit `57874bd`.
