Date/time: 2026-08-26 15:32 +03:00
Tool: Codex
Model, if known: GPT-5.6
Operation ID: memsoftmcc-kais-strengthening-20260826

# Prospective protocol: memory-matched MemSoftMCC strengthening

## Authorization and status

The author approved the complete recommended scientific strengthening package
on 2026-08-26. This document was written before the new canonical computation.
It supplements, without rewriting, the reconstructed 2026-06-23 protocol.

**Technical amendment before the final canonical rerun (2026-08-26):** two
pre-verification executions exposed implementation validity failures. The first
converted mathematically undefined one-class MCC windows into out-of-range
values through an epsilon denominator; the second showed probability saturation
under scikit-learn's default SGD `optimal` schedule. Both output sets were
quarantined before manuscript use. The nondegenerate-margin handling,
source-hash binding, constant SGD schedule, and valid-anchor rule recorded below
were fixed in response to those diagnostics, not selected from the final effect
direction. The quarantined artifacts and rationales remain inside the run
folder under `superseded_numeric_guard_20260826/` and
`superseded_sgd_schedule_20260826/`.

## Questions

- **RQ4 (calibration):** When window and fading memories are matched by both
  mean age and asymptotic Kish ESS, what differences remain attributable to
  kernel shape?
- **RQ5 (finite-time tracking):** Do observed step-drift traces respect the
  exact old-regime mass laws and the stated nondegenerate-margin SoftMCC bound?
- **RQ6 (robustness):** Are the matched-memory conclusions qualitatively stable
  over more synthetic replications, documented ordered streams, and two
  probability-producing online learners?

## Locked estimator grid

Primary memory sizes are `w in {20, 40, 100, 200, 500}`. Each window is paired
with `lambda=(w-1)/(w+1)`, giving approximately 0.904762, 0.951220, 0.980198,
0.990050, and 0.996008. Cumulative SoftMCC remains the full-history reference.
The older unmatched grid is retained as historical sensitivity evidence and is
not relabeled as memory matched.

## Theory target

Write SoftMCC as

`F(a,b,c)=(c-ab)/sqrt(a(1-a)b(1-b))`,

where `a=E[p]`, `b=E[y]`, and `c=E[py]`. On the region
`a,b in [delta,1-delta]`, use the conservative Lipschitz constant
`C_delta=3/(2 delta(1-delta))` for the L1 norm. After `k` post-change
observations, the normalized old-regime masses are

- filled window: `q_w(k)=max(1-k/w,0)`;
- fading after a finite prefix of length `tau`:
  `q_lambda(k)=lambda^k(1-lambda^tau)/(1-lambda^(tau+k))`, approaching
  `lambda^k` for a long prefix.

For independent bounded pairs, normalized deterministic weights, and a path
remaining in the margin region, the target high-probability statement is

`|F(hat_theta_k)-F(theta_1)| <= C_delta [q_k ||theta_0-theta_1||_1 + 3 sqrt(log(6/alpha)/(2 ESS_t))]`.

The proof and numerical identity checks must precede manuscript use. The bound
is not extended to arbitrary adaptive dependent streams; those are an empirical
robustness layer.

## Workloads and units

1. **Synthetic mechanism layer:** abrupt, gradual, recurring, and stationary;
   30 locked seeds `1000..1029`; 6,000 observations per stream, with the same
   fixed deployed-model mechanism as the original study. Abrupt is primary for
   the contamination law; gradual/recurring are robustness strata.
2. **Documented ordered-stream layer:**
   - Electricity (45,312 chronological market observations; binary UP/DOWN),
   - Ozone level (2,536 chronologically collected daily observations; binary),
   - INSECTS abrupt balanced (52,848 real sensor observations, controlled order,
     published change points),
   - INSECTS incremental-gradual balanced (24,150 observations, published change
     point).
   For the six-class INSECTS streams, the smallest published class code present
   (`2`) is locked as the positive one-vs-rest label. No biological name is
   assigned without an authoritative codebook.
3. **Learner layer:** online logistic SGD and Gaussian naive Bayes. A 500-item
   warm-up fits a scaler and initializes each learner; all reported probabilities
   thereafter are predict-then-update. Logistic SGD uses a constant step size
   `eta0=0.001`, averaging, and ten passes restricted to the excluded warm-up;
   Gaussian NB receives one warm-up pass. Learners are robustness strata, not
   independent replications and not objects of a superiority claim.

## Endpoints

- Synthetic primary: post-change integrated absolute tracking error, tail gap,
  and tail trace SD against the locked post-regime SoftMCC target.
- Matched-kernel contrast: within identical stream/seed/memory, window minus
  fading endpoint differences.
- Mechanism check: numerical old-mass identity and empirical error/bound ratio
  in the theorem validation simulation.
- Ordered-stream endpoint: at fixed 500-observation anchors, absolute difference
  between the current monitor and SoftMCC on the next 500 emitted prequential
  predictions/labels. This is a retrospective evaluation of prospective monitor
  relevance, not a drift detector score. Dataset-learner strata remain the
  reporting unit; anchors are not treated as independent replicates.
  A dataset--learner--memory contrast enters the cross-stratum sign summary only
  when at least two and at least half of its declared anchors have nondegenerate
  SoftMCC margins. Undefined positions and valid-anchor counts remain reported.
- Metric-family sensitivity: SoftMCC, hard MCC@0.5, and weighted AUPRC at
  `k in {w/2,w,2w}` after abrupt drift for every matched memory pair.

## Analysis and multiplicity

Synthetic matched contrasts use seed-paired medians and BCa 95% intervals.
One primary family contains the five window-fading contrasts for abrupt-drift
integrated error; Holm correction is applied within that family. Stationary SD
ratios, other regimes, ordered streams, and metric-family probes are explicitly
secondary/descriptive. Multiple checkpoints from one stream are aggregated
before cross-stratum summaries.

## Kill criteria

1. Prior art duplicates the combined bound and matched validation: remove the
   novelty claim and reframe as benchmark/replication.
2. Exact identities fail beyond `1e-12`: stop; do not integrate theory.
3. Empirical theorem checks violate the bound under its recorded assumptions:
   diagnose or withdraw the bound.
4. More than 10% of evaluated theorem states leave the declared margin region:
   report the bound only as a restricted proposition and exclude empirical
   coverage language.
5. Matched results show no stable kernel-shape distinction: retain the validity
   correction but do not claim one kernel is generally better.
6. Ordered-stream conclusions change sign across both learners and datasets:
   report heterogeneity; do not claim external generality.
7. Matched hard MCC/AUPRC fully erase any SoftMCC-specific interpretation:
   position the work as a memory-calibrated metric benchmark.

## Data governance

Raw public datasets stay in the device-local cache
`<controlled-workspace>` and are excluded
from submission/replication archives. The run records source URLs, sizes, and
SHA-256 hashes. Generated probabilities, summaries, code, and manifests may be
shared subject to the final license review.
