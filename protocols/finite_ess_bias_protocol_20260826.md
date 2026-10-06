# Finite-ESS bias of MemSoftMCC — prospectively locked protocol

Operation ID: `softmcc-drift-finite-ess-bias-20260826`
Date/time: 2026-08-27T00:20:00+03:00
Tool: Claude Code; Model: claude-opus-5
Status: **LOCKED BEFORE EXECUTION.** Hypotheses, estimands, seeds, decision
rules, and kill conditions below are fixed prior to seeing any result on the
confirmatory seeds.

## Motivation and its exploratory origin (declared, not hidden)

An exploratory look at the existing 30 stationary units (seeds 1000–1029) of
`experiments/2026-08-26_codex_local_kais_memory_matched` showed a signed
deviation of −0.00544 at `w = 20` whose bootstrap interval excluded zero, with
the same magnitude for the window and fading kernels. That analysis was not
pre-registered and is treated here as hypothesis-generating only. It is not
evidence for any claim in this protocol, and its seeds are excluded from the
confirmatory analysis.

A timing pilot additionally showed that the 3000-observation empirical target
used by the existing runner itself deviates from a 100000-observation reference
by +0.00534 on one seed — the same order of magnitude as the effect under test.
The confirmatory design therefore measures deviation against **two** references
rather than one.

## Scientific question

MemSoftMCC is a nonlinear function (ratio with a square-root denominator) of
four weighted soft counts. Under a finite effective sample size the expectation
of that function need not equal the function of the expectations. The question
is whether, **in a stationary stream with no drift at all**, a short-memory
MemSoftMCC systematically departs from the population SoftMCC, how that
departure scales with the memory setting, and whether the ESS-matching rule
λ = (w−1)/(w+1) equalizes it across kernels.

This stays inside the Drift paper's contribution boundary (temporal memory on
the four SoftMCC counts). It does not touch the Theory paper's batch/Brier
characterization, the Cost paper's static cost/prevalence interface, or any
change-point alarm.

## Hypotheses (pre-specified)

- **H1 (primary).** In the stationary regime the signed deviation
  `E[MemSoftMCC(w)] − SoftMCC_population` is non-zero, and its magnitude
  increases as `w` decreases. Primary comparison: `w = 20`.
- **H2 (scaling).** `|bias| ∝ w^(−β)` with `β ≈ 1`, the first-order delta-method
  prediction.
- **H3 (kernel equivalence).** Under the ESS-matching rule, window and fading
  deviations at the same `w` are indistinguishable.
- **H4 (mechanism).** The deviation grows as class imbalance increases, because
  the weakest confusion cell governs the effective sample size of the ratio.

## Design

Stage 1 — H1/H2/H3
- Regime: `stationary` only (no drift, so any deviation is a finite-ESS effect).
- Generator: `make_synthetic` from the frozen run's `memsoftmcc_matched.py`,
  unchanged, `n_pre = n_post = 3000`, `d = 10`.
- Estimators: `cumulative` plus matched window/fading at
  `w ∈ {20, 40, 100, 200, 500}` — the same five memory scales as the manuscript.
- Seeds: **2000–2499** (500 units), disjoint from the exploratory 1000–1029.
- References, both reported:
  - `R_emp`: SoftMCC of the 3000-observation evaluation segment — the target
    definition the existing runner uses.
  - `R_pop`: SoftMCC of an independent 200000-observation draw from the same
    seed's generating weights — the clean population reference.
- Statistic per unit and estimator: mean of the trace over the 1000-position
  evaluation window, minus each reference.

Stage 2 — H4
- The frozen generator has no imbalance parameter, so the confirmatory run uses
  a locally defined generator that adds an intercept `b` to the linear score and
  is otherwise identical. **Parity gate:** at `b = 0` it must reproduce the
  frozen generator bit-for-bit on the same seeds; if it does not, Stage 2 is
  abandoned and reported as not run.
- Intercept grid: `b ∈ {0, −1, −2, −3}` (decreasing positive-class prevalence).
- Seeds: 3000–3199 (200 units per grid point), disjoint from Stage 1.

Stage 3 — theory
- Second-order delta-method expansion of the SoftMCC ratio in the four weighted
  counts, evaluated at the generator's population moments, giving a predicted
  bias per memory setting. Compared against the Stage 1 measurement in sign and
  order of magnitude.

## Inference

- Bootstrap BCa 95% intervals, 20000 resamples, over units.
- Holm correction across the five memory scales within each kernel family.
- H2 fitted as `log|bias| ~ log w` with a bootstrap interval on the slope.
- No result-dependent selection of `w`, reference, regime, or estimator.

## Kill conditions (written before execution)

- **K1.** If the `w = 20` deviation against `R_pop` does not exclude zero after
  Holm correction, H1 is rejected and the finding is reported as negative.
- **K2.** If no consistent scaling in `w` is found — sign inconsistent or slope
  interval covering zero — H2 is rejected; the effect, if any, is reported
  without a mechanism.
- **K3.** If the Stage 3 delta-method prediction fails to match the measured
  sign or order of magnitude, the theoretical contribution is dropped and only
  the empirical observation stands.
- **K4.** If the Stage 2 parity gate fails, Stage 2 is reported as not run.
- Any fired kill condition is reported as such. No hypothesis is re-specified
  after seeing the confirmatory data.

## Execution environment

Local workstation, the same host and interpreter as the frozen run
(`<workstation>`, Windows, Python 3.12.7, system numpy), executed with process
parallelism over units. Rationale for not using MTA/VPS: measured unit cost is
4.6 s, so 500 units complete in roughly five minutes on 12 local cores, and
keeping the host and library stack identical to the frozen run preserves
numerical comparability with the existing 30 units. A remote host would add an
environment difference and the stale-heartbeat failure mode documented in
`Akis1_AnaPipeline/EXPERIMENT_DURABILITY_AND_RECOVERY.md` without buying wall
clock.

## What this protocol does not authorize

No manuscript edit, no gate transition, no claim of novelty, and no change to
the frozen `2026-08-26_codex_local_kais_memory_matched` evidence. A positive
result would license a separate, author-approved decision about whether and how
to use it.
