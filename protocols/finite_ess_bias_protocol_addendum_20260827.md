# Finite-ESS bias — protocol addendum (stages 4 and 5)

Operation ID: `softmcc-drift-finite-ess-bias-20260826`
Date/time: 2026-08-27T01:40:00+03:00
Status: **LOCKED BEFORE EXECUTION**, extending
`finite_ess_bias_protocol_20260826.md` (SHA-256
`A53E6E1D8E94426AA1AF6865FDE869C428D69574993011DB89B4AB85DB26A2D3`).

Stages 1–3 established that a short-memory MemSoftMCC is downward biased in a
stationary stream, that the bias scales as `C/ESS`, that ESS matching aligns it
across kernels, and that imbalance amplifies it. Two gaps remain before the
result can carry any manuscript claim, and this addendum closes both.

## Gap 1 — an observation is not yet a usable correction

Stage 3 computed `C` from population moments, which an operator does not have.
A claim of practical value requires a **plug-in** estimator that uses only what
the monitor already holds.

**Stage 4.** At each evaluation point, form the weighted first and second
moments of the four soft-count contributions, estimate
`C_hat = ½·tr(H(m_hat)·Cov_hat)` from them, and report
`MemSoftMCC_corrected = MemSoftMCC_raw − C_hat/ESS_hat`.
All quantities are 4-dimensional, so the correction adds constant cost and does
not change the O(1) update character of the estimator.

- Kernel: sliding window (Stage 1 showed the fading kernel is indistinguishable
  under ESS matching, so the window carries the test).
- Seeds: 2000–2499, the same units as Stage 1, so raw and corrected are compared
  on identical streams.
- Evaluation: 10 equally spaced points inside the locked 1000-position window.
- **H5.** The corrected estimator has smaller absolute bias than the raw one at
  `w = 20` and `w = 40`.
- **K5.** If the corrected estimator's residual bias is not credibly smaller
  than the raw bias, the correction is reported as not practically useful and
  only the theoretical observation stands.
- Variance cost is reported alongside; a correction that removes bias while
  inflating the trace standard deviation is reported as such, not hidden.

## Gap 2 — everything so far is synthetic

**Stage 5.** External validity on the manuscript's own real streams, without
introducing a new dataset or retraining any model. The recorded probability and
label sequences of the eight ordered units (Electricity, Ozone, INSECTS
abrupt/gradual, each with two online learners) are reused as-is.

Each stream is **randomly permuted**. Permutation destroys temporal structure —
so any drift is removed — while preserving the empirical joint distribution of
`(p, y)`, including the real class imbalance and real calibration error. Under
permutation the whole-stream SoftMCC is the correct population reference.

- 20 independent permutations per unit, 8 units, 160 permuted streams.
- Evaluation window: the final 1000 positions.
- **H6.** The same downward bias appears, with the same sign and comparable
  magnitude, on permuted real streams.
- **K6.** If the real-stream bias does not exclude zero at `w = 20`, the finding
  is reported as bounded to synthetic conditions and no general claim is made.

## Unchanged commitments

Seeds, windows, the λ = (w−1)/(w+1) rule, BCa inference with 20000 resamples,
and Holm correction within a kernel family carry over. No manuscript edit, gate
transition, or novelty claim is authorized by this addendum. Any fired kill
condition is reported as fired.
