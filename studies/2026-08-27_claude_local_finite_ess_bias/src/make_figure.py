"""Bulgu ozetini tek sayfalik bir figure olarak cizer (kanit gorseli, manuscript figuru degil)."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "studies/2026-08-27_claude_local_finite_ess_bias"
s1 = json.loads((BASE / "evidence/stage1_raw.json").read_text(encoding="utf-8"))
s3 = json.loads((BASE / "evidence/stage3_theory.json").read_text(encoding="utf-8"))
s2 = json.loads((BASE / "evidence/stage2_summary.json").read_text(encoding="utf-8"))

W = s1["windows"]
rows = s1["rows"]
NAVY, TEAL, GREY = "#17324D", "#1F7A7A", "#8C99A6"

obs_w = np.array([np.mean([r["estimators"]["win%d" % w]["dev_pop"] for r in rows]) for w in W])
obs_f = np.array([np.mean([r["estimators"]["fad%d" % w]["dev_pop"] for r in rows]) for w in W])
ci = []
for w in W:
    x = np.array([r["estimators"]["win%d" % w]["dev_pop"] for r in rows])
    se = x.std(ddof=1) / np.sqrt(len(x))
    ci.append(1.96 * se)
ci = np.array(ci)
C = s3["C_mean"]
rho = np.mean([r["R_pop"] for r in rows])
C_cl = -(rho * (1 - rho ** 2) / 2.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axes[0]
ax.axhline(0, color=GREY, lw=1, ls="--")
ax.errorbar(W, obs_w, yerr=ci, fmt="o-", color=NAVY, capsize=3, label="olculen (pencere)", zorder=3)
ax.plot(W, obs_f, "s--", color=TEAL, mfc="none", label="olculen (fading)", zorder=2)
ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(W); ax.set_xticklabels(W)
ax.set_xlabel("hafiza olcegi $w$  (ESS = $w$)")
ax.set_ylabel("yanlilik: E[MemSoftMCC] - populasyon MCC")
ax.set_title("(a) Duragan rejim, drift YOK\n500 onceden kayitli birim", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
ax.plot(W, np.abs(obs_w), "o-", color=NAVY, label="olculen", zorder=3)
ax.plot(W, np.abs([C / w for w in W]), "-", color=TEAL, lw=2, label="bu calisma: $C/\\mathrm{ESS}$", zorder=2)
ax.plot(W, np.abs([C_cl / w for w in W]), ":", color="#B03A2E", lw=2, label="klasik Pearson formulu", zorder=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
ax.set_xticks(W); ax.set_xticklabels(W)
ax.set_xlabel("hafiza olcegi $w$")
ax.set_ylabel("|yanlilik|")
ax.set_title("(b) Teori vs olcum\nsoft-sayim turetmesi 3.2 kat daha dogru", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = axes[2]
bs = sorted(s2["bias_by_b_window"], key=lambda k: -float(k))
prevs = [s2["prevalence_by_b"][b] for b in bs]
for w, col, mk in ((20, NAVY, "o"), (40, TEAL, "s"), (100, GREY, "^")):
    vals = [s2["bias_by_b_window"][b][str(w)] for b in bs]
    ax.plot(prevs, vals, mk + "-", color=col, label="$w=%d$" % w)
ax.axhline(0, color=GREY, lw=1, ls="--")
ax.invert_xaxis()
ax.set_xlabel("pozitif sinif orani  (saga dogru daha dengesiz)")
ax.set_ylabel("yanlilik")
ax.set_title("(c) Dengesizlik ekseni\n800 birim, parite kapisi PASS", fontsize=10)
ax.legend(fontsize=8, frameon=False)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = BASE / "evidence/finite_ess_bias_summary.png"
fig.savefig(out, dpi=200)
print("figure:", out)
