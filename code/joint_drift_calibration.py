"""joint_drift_calibration.py - Jointly drifting concept + calibration experiment.

Paper 3 (Streaming SoftMCC drift), supplementary experiment. Real fits on fixed seeds, resumable
per (regime, seed) block. No fabricated numbers.

Motivation
----------
The main experiments hold the deployed model (and its calibration map) fixed and let
only the *concept* P(y|x) drift. A common, harder regime in practice is that concept
drift and *calibration* drift occur together: as the data-generating process moves,
the model's emitted probabilities not only become less accurate but also become
miscalibrated (over- or under-confident). Because SoftMCC is computed on the
continuous probability mass, calibration distortion changes the soft confusion counts
even when the implied ranking is unchanged. This experiment asks whether the windowed
and fading streaming SoftMCC estimators still track the current (post-drift) concept
when calibration drifts jointly, or whether the joint distortion breaks the
tracking-gap advantage seen under pure concept drift.

Design
------
We reuse the synthetic abrupt concept-drift construction of
drift_experiment.make_synth_stream (concept A -> concept B; the boundary inverts on
half the coordinates at t0). On top of the concept drift we apply, to the post-drift
segment only, a deterministic calibration distortion of the model's emitted
probabilities:

    p' = sigma( g * ( logit(p) + b0 ) ),     p in (0, 1),

a logit-temperature (gain g) plus bias (b0) map. g < 1 makes the model
under-confident (probabilities pulled toward 0.5); g > 1 makes it over-confident;
b0 shifts the operating point. The map is strictly monotone, so it preserves the
ranking of p and leaves hard MCC at any fixed quantile threshold essentially
unchanged, while moving the soft confusion mass; this isolates the calibration-drift
effect from the concept-drift effect already present.

Three joint regimes (all with abrupt concept drift at t0):
    under : g = 0.45, b0 =  0.0   (post-drift model becomes under-confident)
    over  : g = 2.20, b0 =  0.0   (post-drift model becomes over-confident)
    shift : g = 1.00, b0 = +1.1   (post-drift operating-point / bias shift)

For each regime/seed we compute, exactly as in analyze_drift.py:
    - oracle_post_mcc : soft_mcc over the clearly-post-drift tail computed on the
      *distorted* post-drift probabilities, i.e. the true current SoftMCC of what the
      model now emits (this is the target the estimator should track);
    - tracking_gap    : |estimator steady state on the post-drift tail - oracle|;
    - trace_sd        : SD of the estimator over the post-drift tail (trace variability).

We aggregate with means, BCa 95% CIs, Wilcoxon signed-rank vs cumulative, and Cliff's
delta (the same helpers as analyze_drift.py), pooled across the three joint regimes.
Outputs:
    ../results/joint_calib_raw.csv
    ../results/joint_calib_summary.csv
    ../results/joint_calib_paired_vs_cum.csv
    ../results/trace_joint_<regime>.npz   (seed 42 traces, for the figure)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from streaming_mcc import soft_mcc
from drift_experiment import (make_synth_stream, run_estimators, detection_delay,
                              WINDOWS, LAMBDAS, SEEDS)
from analyze_drift import bca_ci, cliffs_delta, stable_seed

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS, exist_ok=True)

TAIL = 1000  # steady-state tail length used for oracle and SD (matches analyze_drift)
EPS = 1e-9

# Joint calibration-drift regimes applied to the post-drift segment.
CALIB_REGIMES = {
    "under": dict(g=0.45, b0=0.0),
    "over":  dict(g=2.20, b0=0.0),
    "shift": dict(g=1.00, b0=1.1),
}


def _logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def apply_calibration_drift(p, t0, g, b0):
    """Apply the monotone logit-temperature+bias map to the post-drift segment."""
    p = np.asarray(p, dtype=float).copy()
    z = _logit(p[t0:])
    p[t0:] = _sigmoid(g * (z + b0))
    return p


def collect():
    rows = []
    trace_bundle = {}
    for cregime, params in CALIB_REGIMES.items():
        for seed in SEEDS:
            # abrupt concept drift, then joint calibration distortion post-drift
            p0, y, t0, _meta = make_synth_stream("abrupt", seed)
            p = apply_calibration_drift(p0, t0, params["g"], params["b0"])

            # clearly-post-drift region (abrupt: full drift at t0); offset 500 as in
            # analyze_drift to avoid the immediate transition window
            post_full = t0 + 500
            oracle = soft_mcc(p[post_full:], y[post_full:])

            traces = run_estimators(p, y)
            if seed == SEEDS[0]:
                trace_bundle[cregime] = {"t0": t0, "oracle": oracle,
                                         **{k: v for k, v in traces.items()}}
            for est_name, tr in traces.items():
                tail = tr[-TAIL:]
                gap = abs(float(np.nanmean(tail)) - oracle)
                sd = float(np.nanstd(tail))
                delay, detected = detection_delay(tr, t0)
                rows.append({"calib_regime": cregime, "seed": seed,
                             "estimator": est_name, "oracle_post_mcc": oracle,
                             "tracking_gap": gap, "trace_sd": sd,
                             "delay": delay, "detected": int(detected)})
        print(f"[joint] calib={cregime} done ({len(SEEDS)} seeds)")
    # save seed-0 traces for the figure
    for cregime, bundle in trace_bundle.items():
        np.savez_compressed(os.path.join(RESULTS, f"trace_joint_{cregime}.npz"),
                            **bundle)
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    # per calib regime
    for (cregime, est), g in df.groupby(["calib_regime", "estimator"]):
        for metric in ["tracking_gap", "trace_sd"]:
            m, lo, hi = bca_ci(g[metric].values,
                               seed=stable_seed("joint", cregime, est, metric))
            out.append({"calib_regime": cregime, "estimator": est, "metric": metric,
                        "mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(g)})
    # pooled across all calib regimes
    for est, g in df.groupby("estimator"):
        for metric in ["tracking_gap", "trace_sd"]:
            m, lo, hi = bca_ci(g[metric].values,
                               seed=stable_seed("joint", "pooled", est, metric))
            out.append({"calib_regime": "pooled", "estimator": est, "metric": metric,
                        "mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(g)})
    return pd.DataFrame(out)


def paired_vs_cum(df):
    """Wilcoxon + Cliff's delta of each estimator's tracking_gap vs cumulative,
    paired across (calib_regime, seed), pooled."""
    out = []
    base = df[df["estimator"] == "cum"].set_index(["calib_regime", "seed"])["tracking_gap"]
    for est in sorted(df["estimator"].unique()):
        if est == "cum":
            continue
        sub = df[df["estimator"] == est].set_index(["calib_regime", "seed"])["tracking_gap"]
        common = base.index.intersection(sub.index)
        a = sub.loc[common].values
        b = base.loc[common].values
        try:
            stat, pval = wilcoxon(a, b)
        except ValueError:
            stat, pval = np.nan, np.nan
        out.append({"estimator": est, "n_pairs": len(common),
                    "mean_gap_est": float(np.mean(a)),
                    "mean_gap_cum": float(np.mean(b)),
                    "mean_diff": float(np.mean(a - b)),
                    "wilcoxon_p": float(pval),
                    "cliffs_delta": cliffs_delta(a, b)})
    return pd.DataFrame(out)


def main():
    df = collect()
    df.to_csv(os.path.join(RESULTS, "joint_calib_raw.csv"), index=False)
    summ = summarize(df)
    summ.to_csv(os.path.join(RESULTS, "joint_calib_summary.csv"), index=False)
    paired = paired_vs_cum(df)
    paired.to_csv(os.path.join(RESULTS, "joint_calib_paired_vs_cum.csv"), index=False)

    print("\n=== pooled tracking_gap mean by estimator (lower=better) ===")
    print(df.groupby("estimator")["tracking_gap"].mean().round(3)
            .sort_values().to_string())
    print("\n=== paired vs cumulative (pooled) ===")
    print(paired.round(4).to_string(index=False))
    print("\n[OK] wrote joint_calib_{raw,summary,paired_vs_cum}.csv")
    return df, summ, paired


if __name__ == "__main__":
    main()
