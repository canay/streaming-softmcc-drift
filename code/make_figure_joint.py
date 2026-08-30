"""make_figure_joint.py - Figure for the jointly-drifting calibration experiment.

fig4_joint_calib.png : streaming-MCC traces (cumulative vs window vs fading) under
abrupt concept drift combined with a post-drift calibration distortion, for the three
joint regimes (under-confident, over-confident, operating-point shift). The dashed
line marks drift onset; the dotted line marks the oracle post-drift SoftMCC that the
estimators should track. No fabricated numbers; all values read from trace_joint_*.npz.
"""
from __future__ import annotations
import os
import numpy as np
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
TITLES = {"under": "under-confident calibration drift",
          "over": "over-confident calibration drift",
          "shift": "operating-point shift"}


def main():
    regimes = ["under", "over", "shift"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, reg in zip(axes, regimes):
        path = os.path.join(RESULTS, f"trace_joint_{reg}.npz")
        d = np.load(path)
        t0 = int(d["t0"]); oracle = float(d["oracle"])
        for k in SHOW:
            ax.plot(np.arange(len(d[k])), d[k], color=COLORS[k], lw=1.0,
                    label=LABELS[k], alpha=0.9)
        ax.axvline(t0, color="k", ls="--", lw=1.0)
        ax.axhline(oracle, color="k", ls=":", lw=1.0)
        ax.set_title("%s\n(onset t=%d, oracle=%.3f)" % (TITLES[reg], t0, oracle),
                     fontsize=9)
        ax.set_xlabel("stream index t")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("streaming SoftMCC")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig4_joint_calib.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
