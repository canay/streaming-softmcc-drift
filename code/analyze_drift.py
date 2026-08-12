"""analyze_drift.py - Analysis of Streaming SoftMCC drift experiments.

Re-runs the streams (fixed seeds) to compute, per (stream, regime, seed, estimator):
  - tracking_gap: |estimator steady-state value - oracle post-drift batch SoftMCC|,
    where the oracle is soft_mcc over the clearly-post-drift tail. Lower = better
    reflects the current concept.
  - trace_sd: SD of the estimator over the stationary post-drift tail (variance/
    noise cost of short memory).
  - delay: descriptive monitoring-response diagnostic from a pre-drift control band.

Then aggregates with means, BCa 95% bootstrap CIs, Wilcoxon signed-rank vs the
cumulative estimator, and Cliff's delta. Reuses the BCa/stat helpers' logic from
Paper 1 (re-implemented compactly here). No fabricated numbers.

Honest framing target: streaming (windowed/fading) estimators track the post-drift
concept far more closely than the cumulative estimator (lower gap), at the cost of
higher trace variability for very short memory (the tracking-gap / trace-variability
trade-off discussed in the manuscript).
"""
from __future__ import annotations
import hashlib
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from streaming_mcc import soft_mcc, BatchSoftMCC, WindowSoftMCC, FadingSoftMCC
from drift_experiment import (make_synth_stream, make_real_stream,
                              run_estimators, detection_delay,
                              WINDOWS, LAMBDAS, SEEDS, GRADUAL_G)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
TAIL = 1000  # steady-state tail length used for oracle and SD


def stable_seed(*parts):
    """Return a process-independent 32-bit seed for deterministic resampling."""
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def bca_ci(x, B=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        m = float(np.mean(x)) if n else np.nan
        return m, m, m
    rng = np.random.default_rng(seed)
    theta = float(np.mean(x))
    boot = np.array([np.mean(rng.choice(x, n, replace=True)) for _ in range(B)])
    z0 = _norm_ppf((np.sum(boot < theta) + 0.5) / B)
    jack = np.array([np.mean(np.delete(x, i)) for i in range(n)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5) + 1e-12
    a = num / den
    zl, zu = _norm_ppf(alpha / 2), _norm_ppf(1 - alpha / 2)
    def adj(z):
        return _norm_cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo = np.quantile(boot, np.clip(adj(zl), 0, 1))
    hi = np.quantile(boot, np.clip(adj(zu), 0, 1))
    return theta, float(lo), float(hi)


def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def _norm_ppf(p):
    from scipy.special import ndtri
    return float(ndtri(np.clip(p, 1e-9, 1 - 1e-9)))


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def collect():
    rows = []
    synth = ["abrupt", "gradual", "recurring"]
    real = [("creditcard_pi10.npz", "real_cc"), ("iotid20_compact.npz", "real_iot")]

    def process(stream, regime, seed, p, y, t0, post_full_start):
        oracle = soft_mcc(p[post_full_start:], y[post_full_start:])
        traces = run_estimators(p, y)
        for est_name, tr in traces.items():
            tail = tr[-TAIL:]
            gap = abs(float(np.nanmean(tail)) - oracle)
            sd = float(np.nanstd(tail))
            delay, detected = detection_delay(tr, t0)
            rows.append({"stream": stream, "regime": regime, "seed": seed,
                         "estimator": est_name, "oracle_post_mcc": oracle,
                         "tracking_gap": gap, "trace_sd": sd,
                         "delay": delay, "detected": int(detected)})

    for regime in synth:
        for seed in SEEDS:
            p, y, t0, meta = make_synth_stream(regime, seed)
            # clearly-post-drift region: after the transition fully completes
            post_full = t0 + (GRADUAL_G if regime == "gradual" else 0) + 500
            process("synth", regime, seed, p, y, t0, post_full)
        print(f"[analyze] synth {regime} done")

    for npz_name, tag in real:
        for seed in SEEDS:
            out = make_real_stream(npz_name, seed)
            if out is None:
                print(f"[analyze] {tag} missing")
                break
            p, y, t0, meta = out
            process("real", tag, seed, p, y, t0, t0 + 500)
        print(f"[analyze] real {tag} done")

    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for (stream, regime, est), g in df.groupby(["stream", "regime", "estimator"]):
        for metric in ["tracking_gap", "trace_sd", "delay"]:
            m, lo, hi = bca_ci(g[metric].values,
                               seed=stable_seed(stream, regime, est, metric))
            out.append({"stream": stream, "regime": regime, "estimator": est,
                        "metric": metric, "mean": m, "ci_lo": lo, "ci_hi": hi,
                        "n": len(g)})
    return pd.DataFrame(out)


def paired_vs_cum(df):
    """Wilcoxon + Cliff's delta of each estimator's tracking_gap vs cumulative,
    paired across (stream, regime, seed)."""
    out = []
    base = df[df["estimator"] == "cum"].set_index(["stream", "regime", "seed"])["tracking_gap"]
    for est in sorted(df["estimator"].unique()):
        if est == "cum":
            continue
        sub = df[df["estimator"] == est].set_index(["stream", "regime", "seed"])["tracking_gap"]
        common = base.index.intersection(sub.index)
        a = sub.loc[common].values   # estimator gap
        b = base.loc[common].values  # cumulative gap
        diff = a - b
        try:
            stat, pval = wilcoxon(a, b)
        except ValueError:
            stat, pval = np.nan, np.nan
        out.append({"estimator": est, "n_pairs": len(common),
                    "mean_gap_est": float(np.mean(a)), "mean_gap_cum": float(np.mean(b)),
                    "mean_diff": float(np.mean(diff)), "wilcoxon_p": float(pval),
                    "cliffs_delta": cliffs_delta(a, b)})
    return pd.DataFrame(out)


def main():
    df = collect()
    df.to_csv(os.path.join(RESULTS, "drift_metrics_raw.csv"), index=False)
    summ = summarize(df)
    summ.to_csv(os.path.join(RESULTS, "drift_metrics_summary.csv"), index=False)
    paired = paired_vs_cum(df)
    paired.to_csv(os.path.join(RESULTS, "drift_paired_vs_cum.csv"), index=False)

    print("\n=== tracking_gap mean by estimator (lower=better), all streams ===")
    print(df.groupby("estimator")["tracking_gap"].mean().round(3).sort_values().to_string())
    print("\n=== paired vs cumulative (tracking_gap) ===")
    print(paired.round(4).to_string(index=False))
    print(f"\n[OK] wrote drift_metrics_raw.csv, drift_metrics_summary.csv, drift_paired_vs_cum.csv")
    return df, summ, paired


if __name__ == "__main__":
    main()
