"""make_figures_drift.py - Figures for the Streaming SoftMCC drift manuscript.

fig1_drift_traces.png : streaming-MCC traces (cumulative vs window vs fading) on the
                        three synthetic regimes, with the drift onset marked.
fig2_gap_tradeoff.png : tracking-gap vs trace-SD scatter (the tracking-gap /
                        trace-variability trade-off), all estimators, with
                        BCa-mean markers.
fig3_real_traces.png  : streaming-MCC traces on the real-data stream simulations.

Reads precomputed traces (trace_*.npz) and summary CSVs from ../results/.
No fabricated numbers; everything plotted comes from the result files.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
if os.path.basename(HERE).lower() == "scripts":
    default_fig_dir = os.path.normpath(os.path.join(os.path.dirname(HERE), "..", "manuscript"))
else:
    default_fig_dir = os.path.join(os.path.dirname(HERE), "figures")
FIGS = os.environ.get("DRIFTMCC_FIG_DIR") or os.environ.get(
    "SOFTMCC_DRIFT_FIG_DIR", default_fig_dir)
os.makedirs(FIGS, exist_ok=True)

SHOW = ["cum", "win250", "fad0.99", "win100"]
COLORS = {"cum": "#444444", "win250": "#1f77b4", "fad0.99": "#2ca02c",
          "win100": "#d62728"}
LABELS = {"cum": "cumulative", "win250": "window w=250",
          "fad0.99": "fading $\\lambda$=0.99", "win100": "window w=100"}


def _smooth_plot(ax, tr, label, color):
    ax.plot(np.arange(len(tr)), tr, color=color, lw=1.0, label=label, alpha=0.9)


def fig1_synth():
    regimes = ["abrupt", "gradual", "recurring"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, reg in zip(axes, regimes):
        path = os.path.join(RESULTS, f"trace_synth_{reg}.npz")
        d = np.load(path); t0 = int(d["t0"])
        for k in SHOW:
            _smooth_plot(ax, d[k], LABELS[k], COLORS[k])
        ax.axvline(t0, color="k", ls="--", lw=1.0)
        ax.set_title(f"{reg} drift (onset at t={t0})")
        ax.set_xlabel("stream index t")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("streaming SoftMCC")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig1_drift_traces.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", out)


def fig2_tradeoff():
    df = pd.read_csv(os.path.join(RESULTS, "drift_metrics_raw.csv"))
    g = df.groupby("estimator").agg(gap=("tracking_gap", "mean"),
                                    sd=("trace_sd", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for _, r in g.iterrows():
        ax.scatter(r["sd"], r["gap"], s=70)
        ax.annotate(r["estimator"], (r["sd"], r["gap"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("trace variability  (mean trace SD on post-drift tail)")
    ax.set_ylabel("tracking gap  |steady-state $-$ oracle post-drift MCC|")
    ax.set_title("Tracking-gap / trace-variability trade-off")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig2_gap_tradeoff.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", out)


def fig3_real():
    tags = [("real_cc", "credit-card stream"), ("real_iot", "IoTID20 stream")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, (tag, title) in zip(axes, tags):
        path = os.path.join(RESULTS, f"trace_{tag}.npz")
        if not os.path.exists(path):
            ax.set_visible(False); continue
        d = np.load(path); t0 = int(d["t0"])
        for k in SHOW:
            _smooth_plot(ax, d[k], LABELS[k], COLORS[k])
        ax.axvline(t0, color="k", ls="--", lw=1.0)
        ax.set_title(f"{title} (injected drift at t={t0})")
        ax.set_xlabel("stream index t")
        ax.set_ylabel("streaming SoftMCC")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig3_real_traces.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig1_synth()
    fig2_tradeoff()
    fig3_real()
