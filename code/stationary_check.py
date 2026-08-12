"""stationary_check.py - descriptive sensitivity check in a stationary stream.

Quantifies fixed-memory deviations from the batch SoftMCC target when there is no
drift. The generated BCa intervals are retained as legacy/internal diagnostics;
the reader-facing manuscript reports stored means and trace variability only and
does not treat this finite-seed check as a formal equivalence test.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from streaming_mcc import soft_mcc, BatchSoftMCC, WindowSoftMCC, FadingSoftMCC
from drift_experiment import run_estimators, WINDOWS, LAMBDAS, SEEDS
from analyze_drift import bca_ci, stable_seed

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
N = 6000
D = 10
TAIL = 1000


def make_stationary(seed):
    rng = np.random.default_rng(1000 + seed)
    w = rng.normal(0, 1, D)
    X = rng.normal(0, 1, size=(N, D))
    z = X @ w
    pt = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(0, 1, N) < pt).astype(float)
    p = pt  # well-matched, calibrated deployed model
    return p, y


def main():
    rows = []
    for seed in SEEDS:
        p, y = make_stationary(seed)
        oracle = soft_mcc(p, y)
        traces = run_estimators(p, y)
        for est, tr in traces.items():
            ss = float(np.nanmean(tr[-TAIL:]))
            rows.append({"seed": seed, "estimator": est,
                         "steady_state": ss, "cumulative_full": oracle,
                         "abs_diff_vs_batch": abs(ss - oracle),
                         "trace_sd": float(np.nanstd(tr[-TAIL:]))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "stationary_equivalence_raw.csv"), index=False)

    out = []
    for est, g in df.groupby("estimator"):
        m, lo, hi = bca_ci(g["abs_diff_vs_batch"].values,
                           seed=stable_seed("stationary", est, "abs_diff_vs_batch"))
        out.append({"estimator": est, "mean_abs_diff_vs_batch": m,
                    "ci_lo": lo, "ci_hi": hi,
                    "mean_trace_sd": float(g["trace_sd"].mean())})
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(RESULTS, "stationary_equivalence_summary.csv"), index=False)
    print("=== Stationary (no-drift) equivalence: |steady-state - batch SoftMCC| ===")
    print(summ.round(4).to_string(index=False))
    return summ


if __name__ == "__main__":
    main()
