"""q1_audit_stats.py - Q1 audit statistical summaries.

Reads existing raw result CSV files and writes secondary summaries needed by the
Q1 action-register revision:

  - paired mean differences versus cumulative SoftMCC / cumulative family metric;
  - BCa 95% confidence intervals for paired differences;
  - Holm-adjusted p-values within each table-defined comparison family.

This is not a new experiment. It does not regenerate streams or change raw
results; it only summarizes stored result artifacts.
"""
from __future__ import annotations

import os
import zlib

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import wilcoxon


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")


def _stable_seed(label: str) -> int:
    return zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    from math import erf, sqrt

    if np.isscalar(x):
        return 0.5 * (1.0 + erf(float(x) / sqrt(2.0)))
    return np.asarray([_norm_cdf(v) for v in np.asarray(x, dtype=float)])


def _norm_ppf(p: float) -> float:
    return float(ndtri(np.clip(p, 1e-9, 1.0 - 1e-9)))


def bca_mean_ci(x, label: str, b: int = 2000, alpha: float = 0.05):
    """BCa CI for the mean of a one-dimensional vector."""
    vals = np.asarray(x, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n < 3:
        theta = float(np.mean(vals)) if n else np.nan
        return theta, theta, theta

    rng = np.random.default_rng(_stable_seed(label))
    theta = float(np.mean(vals))
    boot = np.array([np.mean(rng.choice(vals, n, replace=True)) for _ in range(b)])
    z0 = _norm_ppf((np.sum(boot < theta) + 0.5) / b)
    jack = np.array([np.mean(np.delete(vals, i)) for i in range(n)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5) + 1e-12
    acc = num / den
    zl, zu = _norm_ppf(alpha / 2.0), _norm_ppf(1.0 - alpha / 2.0)

    def adj(z):
        return _norm_cdf(z0 + (z0 + z) / (1.0 - acc * (z0 + z)))

    lo = np.quantile(boot, np.clip(adj(zl), 0.0, 1.0))
    hi = np.quantile(boot, np.clip(adj(zu), 0.0, 1.0))
    return theta, float(lo), float(hi)


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * p[idx]
        running = max(running, adjusted)
        out[idx] = min(running, 1.0)
    return out


def paired_effects(df, index_cols, estimator_col="estimator", family_label="all"):
    """Paired tracking-gap effects for all non-cumulative estimators."""
    base = df[df[estimator_col] == "cum"].set_index(index_cols)["tracking_gap"]
    rows = []
    for est in sorted(df[estimator_col].dropna().unique()):
        if est == "cum":
            continue
        sub = df[df[estimator_col] == est].set_index(index_cols)["tracking_gap"]
        common = base.index.intersection(sub.index)
        a = sub.loc[common].astype(float).values
        b = base.loc[common].astype(float).values
        diff = a - b
        try:
            _stat, pval = wilcoxon(a, b)
        except ValueError:
            pval = np.nan
        mean_diff, ci_lo, ci_hi = bca_mean_ci(
            diff, f"{family_label}:{est}:paired_diff")
        mean_est = float(np.mean(a))
        mean_cum = float(np.mean(b))
        rows.append({
            "family": family_label,
            "estimator": est,
            "n_pairs": int(len(common)),
            "mean_gap_est": mean_est,
            "mean_gap_cum": mean_cum,
            "mean_diff_est_minus_cum": mean_diff,
            "diff_bca95_lo": ci_lo,
            "diff_bca95_hi": ci_hi,
            "relative_gap_closure": float((mean_cum - mean_est) / (mean_cum + 1e-12)),
            "wilcoxon_p": float(pval),
            "cliffs_delta": cliffs_delta(a, b),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_p"] = holm_adjust(out["wilcoxon_p"].values)
    return out


def summarize_trace_sd(df, group_cols):
    return (
        df.groupby(group_cols + ["estimator"], as_index=False)["trace_sd"]
        .mean()
        .rename(columns={"trace_sd": "mean_trace_sd"})
    )


def main():
    drift = pd.read_csv(os.path.join(RESULTS, "drift_metrics_raw.csv"))
    joint = pd.read_csv(os.path.join(RESULTS, "joint_calib_raw.csv"))
    metric = pd.read_csv(os.path.join(RESULTS, "metric_family_comparator_raw.csv"))

    drift_overall = paired_effects(
        drift, ["stream", "regime", "seed"], family_label="drift_all")
    drift_sd = summarize_trace_sd(drift, [])
    drift_overall = drift_overall.merge(drift_sd, on="estimator", how="left")

    regime_rows = []
    for regime, g in drift.groupby("regime"):
        tmp = paired_effects(g, ["stream", "seed"], family_label=f"regime_{regime}")
        tmp.insert(0, "regime", regime)
        regime_rows.append(tmp)
    drift_regime = pd.concat(regime_rows, ignore_index=True)

    joint_overall = paired_effects(
        joint, ["calib_regime", "seed"], family_label="joint_calibration")
    joint_sd = summarize_trace_sd(joint, [])
    joint_overall = joint_overall.merge(joint_sd, on="estimator", how="left")

    metric_rows = []
    for family, g in metric.groupby("family"):
        tmp = paired_effects(g, ["stream", "regime", "seed"], family_label=family)
        tmp.insert(0, "metric_family", family)
        metric_rows.append(tmp)
    metric_effects = pd.concat(metric_rows, ignore_index=True)

    outputs = {
        "q1_drift_paired_effects.csv": drift_overall,
        "q1_drift_regime_effects.csv": drift_regime,
        "q1_joint_paired_effects.csv": joint_overall,
        "q1_metric_family_paired_effects.csv": metric_effects,
    }
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(RESULTS, name), index=False)

    summary_path = os.path.join(RESULTS, "q1_audit_stats_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Q1 Audit Statistical Summaries\n\n")
        f.write("Generated from existing raw result CSV files; no streams were regenerated.\n\n")
        for title, frame in [
            ("Drift paired effects", drift_overall),
            ("Joint calibration paired effects", joint_overall),
            ("Metric-family paired effects", metric_effects),
        ]:
            cols = [
                c for c in [
                    "metric_family", "family", "estimator", "n_pairs",
                    "mean_gap_est", "mean_gap_cum", "mean_diff_est_minus_cum",
                    "diff_bca95_lo", "diff_bca95_hi", "wilcoxon_p", "holm_p",
                    "cliffs_delta", "mean_trace_sd",
                ] if c in frame.columns
            ]
            f.write(f"## {title}\n\n")
            f.write(frame[cols].round(6).to_markdown(index=False))
            f.write("\n\n")

    print("[OK] wrote Q1 audit statistical summaries:")
    for name in outputs:
        print(f"  - {os.path.join(RESULTS, name)}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
