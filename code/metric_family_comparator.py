"""metric_family_comparator.py - close comparator probe for the Streaming SoftMCC drift paper.

This script asks whether the main tracking-gap reduction is just "windowing any
metric" or whether the time-local SoftMCC object has a distinct operating profile.

Families:
  - SoftMCC: cumulative, sliding windows, and fading estimators from streaming_mcc.
  - HardMCC@0.5: the same additive estimators after thresholding p at 0.5.
  - AUPRC: cumulative and sliding-window average precision. AUPRC is not a
    four-count additive statistic, so no fading estimator is reported for it.

The endpoint mirrors analyze_drift.py: tracking gap to the clearly-post-drift
batch target, lower is better. Outputs are written under 03_experiments/results.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, matthews_corrcoef

import drift_experiment as de
from analyze_drift import bca_ci, cliffs_delta, stable_seed
from streaming_mcc import soft_mcc


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
TAIL = 1000
AP_TAIL_STRIDE = 10


def _active_paper1_data_dir() -> str:
    """Resolve Paper-1 NPZ caches through the shared Drift experiment resolver."""
    return de.resolve_paper1_data_dir()


def _ap_safe(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(average_precision_score(y, p))
        except ValueError:
            return np.nan


def _hard_mcc_batch(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if len(y) == 0:
        return np.nan
    return float(matthews_corrcoef(y, pred))


def _soft_or_hard_records(stream, regime, seed, family, p_trace, y, t0, post_full):
    traces = de.run_estimators(p_trace, y)
    if family == "SoftMCC":
        oracle = float(soft_mcc(y[post_full:], p_trace[post_full:]))
    elif family == "HardMCC@0.5":
        oracle = _hard_mcc_batch(y[post_full:], p_trace[post_full:])
    else:
        raise ValueError(family)

    rows = []
    for est_name, tr in traces.items():
        tail = tr[-TAIL:]
        delay, detected = de.detection_delay(tr, t0)
        rows.append({
            "stream": stream,
            "regime": regime,
            "seed": seed,
            "family": family,
            "estimator": est_name,
            "oracle_post": oracle,
            "tracking_gap": abs(float(np.nanmean(tail)) - oracle),
            "trace_sd": float(np.nanstd(tail)),
            "delay": delay,
            "detected": int(detected),
            "n_tail": int(np.sum(~np.isnan(tail))),
        })
    return rows


def _auprc_records(stream, regime, seed, p, y, post_full):
    oracle = _ap_safe(y[post_full:], p[post_full:])
    n = len(y)
    tail_start = max(0, n - TAIL)
    rows = []

    specs = [("cum", None)] + [(f"win{w}", int(w)) for w in de.WINDOWS]
    for est_name, window in specs:
        vals = []
        grid = list(range(tail_start, n, AP_TAIL_STRIDE))
        if not grid or grid[-1] != n - 1:
            grid.append(n - 1)
        for i in grid:
            lo = 0 if window is None else max(0, i - window + 1)
            vals.append(_ap_safe(y[lo:i + 1], p[lo:i + 1]))
        vals = np.asarray(vals, dtype=float)
        rows.append({
            "stream": stream,
            "regime": regime,
            "seed": seed,
            "family": "AUPRC",
            "estimator": est_name,
            "oracle_post": oracle,
            "tracking_gap": abs(float(np.nanmean(vals)) - oracle),
            "trace_sd": float(np.nanstd(vals)),
            "delay": np.nan,
            "detected": np.nan,
            "n_tail": int(np.sum(~np.isnan(vals))),
        })
    return rows


def _process_stream(stream, regime, seed, p, y, t0, post_full):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p_hard = (p >= 0.5).astype(float)
    rows = []
    rows.extend(_soft_or_hard_records(stream, regime, seed, "SoftMCC", p, y, t0, post_full))
    rows.extend(_soft_or_hard_records(stream, regime, seed, "HardMCC@0.5", p_hard, y, t0, post_full))
    rows.extend(_auprc_records(stream, regime, seed, p, y, post_full))
    return rows


def collect():
    rows = []
    data_dir = _active_paper1_data_dir()
    de.E1_DATA = data_dir

    for regime in ["abrupt", "gradual", "recurring"]:
        for seed in de.SEEDS:
            p, y, t0, _meta = de.make_synth_stream(regime, seed)
            post_full = t0 + (de.GRADUAL_G if regime == "gradual" else 0) + 500
            rows.extend(_process_stream("synth", regime, seed, p, y, t0, post_full))
        print(f"[metric-family] synth {regime} done", flush=True)

    for npz_name, tag in [("creditcard_pi10.npz", "real_cc"), ("iotid20_compact.npz", "real_iot")]:
        for seed in de.SEEDS:
            out = de.make_real_stream(npz_name, seed)
            if out is None:
                print(f"[metric-family] {npz_name} missing in {data_dir}, skipped", flush=True)
                break
            p, y, t0, _meta = out
            rows.extend(_process_stream("real", tag, seed, p, y, t0, t0 + 500))
        print(f"[metric-family] real {tag} done", flush=True)

    return pd.DataFrame(rows)


def summarize(df):
    rows = []
    for (family, est), g in df.groupby(["family", "estimator"]):
        for metric in ["tracking_gap", "trace_sd"]:
            m, lo, hi = bca_ci(g[metric].values,
                               seed=stable_seed("metric-family", family, est, metric))
            rows.append({
                "family": family,
                "estimator": est,
                "metric": metric,
                "mean": m,
                "ci_lo": lo,
                "ci_hi": hi,
                "n": len(g),
            })
    return pd.DataFrame(rows)


def paired_vs_cum(df):
    rows = []
    for family, gf in df.groupby("family"):
        base = gf[gf["estimator"] == "cum"].set_index(["stream", "regime", "seed"])["tracking_gap"]
        for est in sorted(gf["estimator"].unique()):
            if est == "cum":
                continue
            sub = gf[gf["estimator"] == est].set_index(["stream", "regime", "seed"])["tracking_gap"]
            common = base.index.intersection(sub.index)
            if len(common) == 0:
                continue
            a = sub.loc[common].values
            b = base.loc[common].values
            try:
                _stat, pval = wilcoxon(a, b)
            except ValueError:
                pval = np.nan
            rows.append({
                "family": family,
                "estimator": est,
                "n_pairs": len(common),
                "mean_gap_est": float(np.mean(a)),
                "mean_gap_cum": float(np.mean(b)),
                "mean_diff": float(np.mean(a - b)),
                "relative_gap_closure": float((np.mean(b) - np.mean(a)) / (np.mean(b) + 1e-12)),
                "wilcoxon_p": float(pval),
                "cliffs_delta": cliffs_delta(a, b),
            })
    return pd.DataFrame(rows)


def best_local(summary):
    gap = summary[summary["metric"] == "tracking_gap"].copy()
    gap = gap[gap["estimator"] != "cum"]
    return gap.sort_values(["family", "mean"]).groupby("family", as_index=False).first()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = collect()
    summary = summarize(df)
    paired = paired_vs_cum(df)
    best = best_local(summary)

    df.to_csv(os.path.join(RESULTS, "metric_family_comparator_raw.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS, "metric_family_comparator_summary.csv"), index=False)
    paired.to_csv(os.path.join(RESULTS, "metric_family_comparator_paired.csv"), index=False)
    best.to_csv(os.path.join(RESULTS, "metric_family_comparator_best.csv"), index=False)

    print("\n=== Mean tracking gap by family/estimator (lower=better) ===")
    gap = summary[summary["metric"] == "tracking_gap"].sort_values(["family", "mean"])
    print(gap[["family", "estimator", "mean", "ci_lo", "ci_hi", "n"]].round(4).to_string(index=False))

    print("\n=== Paired tracking gap vs cumulative ===")
    print(paired.sort_values(["family", "mean_gap_est"]).round(4).to_string(index=False))

    print("\n[OK] wrote metric_family_comparator_*.csv")
    return df, summary, paired, best


if __name__ == "__main__":
    main()
