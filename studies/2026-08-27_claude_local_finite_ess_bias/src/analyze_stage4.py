"""Asama 4 analizi — H5: tak-calistir duzeltme kalan yanliligi azaltiyor mu?"""
import json
import pathlib

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "studies/2026-08-27_claude_local_finite_ess_bias"
d = json.loads((BASE / "evidence/stage4_raw.json").read_text(encoding="utf-8"))
rows, W = d["rows"], d["windows"]


def bca(x, nboot=20000, alpha=0.05, seed=20260827):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = x.size; theta = x.mean()
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, size=(nboot, n))].mean(axis=1)
    fr = (boots < theta).mean()
    z0 = stats.norm.ppf(fr) if 0 < fr < 1 else 0.0
    jack = (x.sum() - x) / (n - 1); jbar = jack.mean()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    acc = ((jbar - jack) ** 3).sum() / den if den else 0.0
    adj = lambda z: stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))  # noqa: E731
    return theta, np.percentile(boots, 100 * adj(stats.norm.ppf(alpha / 2))), \
        np.percentile(boots, 100 * adj(stats.norm.ppf(1 - alpha / 2)))


print("=" * 96)
print("ASAMA 4 — tak-calistir yanlilik duzeltmesi, pencere cekirdegi, %d birim" % d["n_units"])
print("  C_hat her degerlendirme noktasinda AGIRLIKLI ORNEKLEM momentlerinden; populasyon bilgisi kullanilmadi")
print("=" * 96)
print()
print("%-6s %14s %14s %10s   %12s %12s %8s" % ("w", "ham yanlilik", "duzeltilmis", "azalma", "ham SD", "duz. SD", "SD bedeli"))
print("-" * 96)
summary = {}
for w in W:
    key = str(w)
    raw = np.array([r["w"][key]["dev_raw"] for r in rows if key in r["w"]])
    cor = np.array([r["w"][key]["dev_corr"] for r in rows if key in r["w"]])
    rsd = np.array([r["w"][key]["raw_sd"] for r in rows if key in r["w"]])
    csd = np.array([r["w"][key]["corr_sd"] for r in rows if key in r["w"]])
    tr, lr, hr = bca(raw)
    tc, lc, hc = bca(cor)
    red = 100 * (1 - abs(tc) / abs(tr)) if tr else np.nan
    summary[w] = {"raw": tr, "raw_ci": [lr, hr], "corr": tc, "corr_ci": [lc, hc],
                  "reduction_pct": red, "raw_sd": float(rsd.mean()), "corr_sd": float(csd.mean())}
    print("%-6d %+14.6f %+14.6f %9.1f%%   %12.5f %12.5f %7.2fx"
          % (w, tr, tc, red, rsd.mean(), csd.mean(), csd.mean() / rsd.mean()))

print()
print("Duzeltilmis tahmin edicinin CI'lari (sifiri iceriyorsa yanlilik kapanmis demektir):")
for w in W:
    s = summary[w]
    zc = "sifiri ICERIYOR" if s["corr_ci"][0] <= 0 <= s["corr_ci"][1] else "sifiri disliyor"
    zr = "sifiri ICERIYOR" if s["raw_ci"][0] <= 0 <= s["raw_ci"][1] else "sifiri disliyor"
    print("  w=%-4d ham [%+.6f, %+.6f] %s | duzeltilmis [%+.6f, %+.6f] %s"
          % (w, s["raw_ci"][0], s["raw_ci"][1], zr, s["corr_ci"][0], s["corr_ci"][1], zc))

print()
print("H5 karari (w=20 ve w=40'ta mutlak yanlilik azaldi mi):")
ok = True
for w in (20, 40):
    paired = np.array([abs(r["w"][str(w)]["dev_corr"]) - abs(r["w"][str(w)]["dev_raw"])
                       for r in rows if str(w) in r["w"]])
    t, lo, hi = bca(paired)
    verdict = "AZALDI" if hi < 0 else ("degismedi/artti" if lo > 0 else "ayirt edilemez")
    ok &= hi < 0
    print("  w=%-4d eslesmis |duzeltilmis|-|ham| = %+.6f  [%+.6f, %+.6f]  -> %s" % (w, t, lo, hi, verdict))
print()
print("  K5: %s" % ("ATESLEMEDI - duzeltme pratik olarak isliyor" if ok else "ATESLEDI - duzeltme pratik degil"))

(BASE / "evidence/stage4_summary.json").write_text(
    json.dumps({str(k): v for k, v in summary.items()}, indent=1), encoding="utf-8")
