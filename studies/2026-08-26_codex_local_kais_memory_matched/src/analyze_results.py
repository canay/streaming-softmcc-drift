"""Verified aggregation and inference for the memory-matched run."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, wilcoxon

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))

from memsoftmcc_matched import (  # noqa: E402
    WINDOWS, atomic_csv, atomic_json, matched_lambda, soft_mcc,
)


def bca_median(x, b=5000, alpha=0.05, seed=20260826):
    x = np.asarray(x, dtype=float)
    n = len(x)
    theta = float(np.median(x))
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(x, size=(b, n), replace=True), axis=1)
    prop = np.clip(np.mean(boot < theta), 1.0 / (2 * b), 1.0 - 1.0 / (2 * b))
    z0 = norm.ppf(prop)
    jack = np.array([np.median(np.delete(x, i)) for i in range(n)])
    center = jack.mean()
    num = np.sum((center - jack) ** 3)
    den = 6.0 * np.sum((center - jack) ** 2) ** 1.5
    acc = float(num / den) if den > 0 else 0.0
    qs = []
    for a in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = norm.ppf(a)
        adj = norm.cdf(z0 + (z0 + z) / (1.0 - acc * (z0 + z)))
        qs.append(float(np.clip(adj, 0.0, 1.0)))
    return theta, float(np.quantile(boot, qs[0])), float(np.quantile(boot, qs[1]))


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_rank_biserial(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[diff != 0]
    if not len(diff):
        return 0.0
    ranks = rankdata(np.abs(diff))
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def summarize_synthetic(raw):
    return (raw.groupby(["regime", "estimator", "kernel", "w", "lambda"], as_index=False)
            .agg(n=("seed", "nunique"), median_iae1000=("iae1000", "median"),
                 mean_iae1000=("iae1000", "mean"), median_iae_2w=("iae_2w", "median"),
                 median_tail_gap=("tail_gap", "median"), median_trace_sd=("trace_sd", "median")))


def primary_effects(raw):
    abrupt = raw[(raw.regime == "abrupt") & (raw.kernel.isin(["window", "fading"]))]
    rows = []
    for w in WINDOWS:
        sub = abrupt[abrupt.w == w].pivot(index="seed", columns="kernel", values="iae1000").dropna()
        diff = (sub["window"] - sub["fading"]).to_numpy()
        med, lo, hi = bca_median(diff, seed=20260826 + w)
        try:
            p = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p = 1.0
        rows.append({
            "w": w, "lambda": matched_lambda(w), "n_pairs": len(diff),
            "median_window_minus_fading_iae1000": med,
            "bca95_low": lo, "bca95_high": hi, "wilcoxon_p": p,
            "paired_rank_biserial": paired_rank_biserial(diff),
            "window_better_count": int(np.sum(diff < 0)),
            "fading_better_count": int(np.sum(diff > 0)),
            "ties": int(np.sum(diff == 0)),
        })
    out = pd.DataFrame(rows)
    out["holm_p"] = holm(out.wilcoxon_p)
    return out


def stationary_ratios(raw):
    stat = raw[(raw.regime == "stationary") & raw.kernel.isin(["window", "fading"])]
    rows = []
    for w in WINDOWS:
        sub = stat[stat.w == w].pivot(index="seed", columns="kernel", values="trace_sd").dropna()
        ratio = sub.window / sub.fading
        rows.append({
            "w": w, "lambda": matched_lambda(w), "n_pairs": len(sub),
            "median_window_to_fading_sd_ratio": float(np.median(ratio)),
            "iqr_low": float(np.quantile(ratio, 0.25)),
            "iqr_high": float(np.quantile(ratio, 0.75)),
        })
    return pd.DataFrame(rows)


def ordered_contrasts(raw):
    local = raw[raw.kernel.isin(["window", "fading"])]
    rows = []
    for (dataset, learner, w), sub in local.groupby(["dataset", "learner", "w"]):
        required = np.maximum(2, np.ceil(0.5 * sub["n_anchors"].to_numpy(dtype=float)))
        if np.any(sub["n_valid_anchors"].to_numpy(dtype=float) < required):
            continue
        vals = sub.set_index("kernel")["mean_future_gap"]
        if not {"window", "fading"}.issubset(vals.index):
            continue
        rows.append({
            "dataset": dataset, "learner": learner, "w": int(w),
            "lambda": matched_lambda(int(w)),
            "window_mean_future_gap": float(vals.window),
            "fading_mean_future_gap": float(vals.fading),
            "window_minus_fading": float(vals.window - vals.fading),
        })
    return pd.DataFrame(rows)


def metric_family_summary(raw):
    return (raw.groupby(["metric", "w", "lambda", "k_over_w", "kernel"], as_index=False)
            .agg(n=("seed", "nunique"), median_absolute_gap=("absolute_gap", "median"),
                 mean_absolute_gap=("absolute_gap", "mean")))


def make_figure(primary, synthetic_raw):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), constrained_layout=True)
    w = 200
    lam = matched_lambda(w)
    x = np.linspace(0, 2.0, 401)
    axes[0].plot(x, np.maximum(1.0 - x, 0.0), color="#17324D", lw=2.0, label="Window")
    axes[0].plot(x, lam ** (x * w), color="#238A8D", lw=2.0, label="Fading")
    axes[0].axvline(1.0, color="#9AA8B2", lw=0.8, ls="--")
    axes[0].set(xlabel=r"Post-change age $k/w$", ylabel="Old-regime mass",
                title="A  Matched memory, different kernels", xlim=(0, 2), ylim=(-0.02, 1.02))
    axes[0].legend(frameon=False, fontsize=8)

    errors = {"cum": [], "win200": [], "fad200": []}
    for path in sorted((RUN / "raw/synthetic_units").glob("abrupt_*.npz")):
        with np.load(path, allow_pickle=False) as dat:
            p, y = dat["p"], dat["y"]
            t0 = int(dat["t0"])
            target = soft_mcc(y[t0:], p[t0:])
            names = list(dat["method_names"].astype(str))
            traces = dat["traces"]
            for name in errors:
                errors[name].append(np.abs(traces[names.index(name), t0:t0 + 1000] - target))
    labels = {"cum": "Cumulative", "win200": "Window $w=200$", "fad200": "Fading matched to $w=200$"}
    colors = {"cum": "#617484", "win200": "#17324D", "fad200": "#238A8D"}
    for name in errors:
        med = np.nanmedian(np.vstack(errors[name]), axis=0)
        axes[1].plot(np.arange(1, len(med) + 1), med, lw=1.7,
                     color=colors[name], label=labels[name])
    axes[1].set(xlabel="Post-change observations", ylabel="Median absolute tracking error",
                title="B  Abrupt-drift tracking", xlim=(1, 1000))
    axes[1].legend(frameon=False, fontsize=7.5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D8DEE3", lw=0.6, alpha=0.7)
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(9)
    out = RUN / "evidence/fig_memory_matched.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    raw_dir, evidence = RUN / "raw", RUN / "evidence"
    required = [raw_dir / "synthetic_metrics.csv", raw_dir / "metric_family.csv",
                raw_dir / "ordered_metrics.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing raw aggregate(s): {missing}")
    synthetic = pd.read_csv(required[0])
    family = pd.read_csv(required[1])
    ordered = pd.read_csv(required[2])
    synth_summary = summarize_synthetic(synthetic)
    primary = primary_effects(synthetic)
    stationary = stationary_ratios(synthetic)
    ordered_pairs = ordered_contrasts(ordered)
    family_summary = metric_family_summary(family)
    atomic_csv(evidence / "synthetic_summary.csv", synth_summary)
    atomic_csv(evidence / "primary_paired_effects.csv", primary)
    atomic_csv(evidence / "stationary_variance_ratios.csv", stationary)
    atomic_csv(evidence / "ordered_summary.csv", ordered)
    atomic_csv(evidence / "ordered_matched_contrasts.csv", ordered_pairs)
    atomic_csv(evidence / "metric_family_matched_summary.csv", family_summary)

    calibration = pd.read_csv(evidence / "memory_calibration.csv")
    table = calibration.merge(primary, on=["w", "lambda"], how="left")
    table = table.merge(stationary, on=["w", "lambda"], how="left", suffixes=("", "_stationary"))
    atomic_csv(evidence / "table_memory_matched.csv", table)
    abrupt = synth_summary[synth_summary.regime == "abrupt"]
    atomic_csv(evidence / "figure_memory_shape.csv", abrupt)
    figure = make_figure(primary, synthetic)

    theory = json.loads((evidence / "theory_verification.json").read_text(encoding="utf-8"))
    verdict = {
        "status": "verified" if theory.get("status") == "passed" else "blocked",
        "synthetic_units": int(synthetic[["regime", "seed"]].drop_duplicates().shape[0]),
        "synthetic_seeds_per_regime": int(synthetic.groupby("regime").seed.nunique().min()),
        "ordered_dataset_learner_strata": int(ordered[["dataset", "learner"]].drop_duplicates().shape[0]),
        "primary_holm_significant_count": int((primary.holm_p < 0.05).sum()),
        "theory_status": theory.get("status"),
        "figure": str(figure.relative_to(RUN)),
    }
    atomic_json(evidence / "analysis_verdict.json", verdict)

    best = primary.loc[primary.median_window_minus_fading_iae1000.abs().idxmax()]
    ordered_signs = ordered_pairs.window_minus_fading
    lines = [
        "# Verified result narrative", "",
        f"- Theory gate: `{theory['status']}`; maximum identity error "
        f"{theory['max_memory_identity_error']:.3e}; simulated valid-state coverage "
        f"{theory['coverage_over_valid_states']:.3f}.",
        f"- Synthetic evidence: {verdict['synthetic_units']} independent regime-seed units "
        f"({verdict['synthetic_seeds_per_regime']} seeds per regime).",
        f"- Primary matched family: {verdict['primary_holm_significant_count']}/5 window-fading "
        "contrasts remain significant after Holm correction.",
        f"- Largest median window-minus-fading IAE contrast occurs at w={int(best.w)}: "
        f"{best.median_window_minus_fading_iae1000:.4f} "
        f"(BCa 95% [{best.bca95_low:.4f}, {best.bca95_high:.4f}]).",
        f"- Ordered-stream robustness: window has lower mean future gap in "
        f"{int((ordered_signs < 0).sum())}/{len(ordered_signs)} dataset-learner-memory strata; "
        "heterogeneity is retained rather than converted into a universal winner claim.",
    ]
    (evidence / "verified_result_narrative.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(verdict, indent=2))
    if verdict["status"] != "verified":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
