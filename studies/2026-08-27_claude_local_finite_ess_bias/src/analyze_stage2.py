"""Asama 2 analizi — H4: yanlilik sinif dengesizligiyle buyuyor mu?"""
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "studies/2026-08-27_claude_local_finite_ess_bias"
sys.path.insert(0, str(BASE / "src"))
raw = json.loads((BASE / "evidence/stage2_raw.json").read_text(encoding="utf-8"))
rows, WINDOWS = raw["rows"], raw["windows"]


def bca(x, nboot=20000, alpha=0.05, seed=20260827):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return np.nan, np.nan, np.nan
    theta = x.mean()
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, size=(nboot, n))].mean(axis=1)
    frac = (boots < theta).mean()
    z0 = stats.norm.ppf(frac) if 0 < frac < 1 else 0.0
    jack = (x.sum() - x) / (n - 1)
    jbar = jack.mean()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    acc = ((jbar - jack) ** 3).sum() / den if den else 0.0
    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    return theta, np.percentile(boots, 100 * adj(stats.norm.ppf(alpha / 2))), \
        np.percentile(boots, 100 * adj(stats.norm.ppf(1 - alpha / 2)))


by_b = {}
for r in rows:
    by_b.setdefault(r["b"], []).append(r)

print("=" * 92)
print("ASAMA 2 — sinif dengesizligi ekseni (H4). parite kapisi: %s" % ("PASS" if raw["parity_ok"] else "FAIL"))
print("=" * 92)
print()
print("%-7s %10s %10s   %s" % ("b", "poz. oran", "pop. MCC", "tanimsiz iz orani (w=20)"))
print("-" * 66)
for b in sorted(by_b, reverse=True):
    rs = by_b[b]
    prev = np.mean([r["prevalence"] for r in rs])
    pop = np.mean([r["R_pop"] for r in rs])
    und = np.mean([1.0 - r["estimators"]["win20"]["n_finite"] / raw["horizon"] for r in rs])
    print("%-7.1f %10.4f %10.4f   %10.4f" % (b, prev, pop, und))

print()
print("Yanlilik (populasyon referansina gore), pencere cekirdegi:")
print("%-7s %10s" % ("b", "poz.oran") + "".join("%16s" % ("w=%d" % w) for w in WINDOWS))
print("-" * 92)
scale_rows = {}
for b in sorted(by_b, reverse=True):
    rs = by_b[b]
    prev = np.mean([r["prevalence"] for r in rs])
    line = "%-7.1f %10.4f" % (b, prev)
    vals = []
    for w in WINDOWS:
        x = np.array([r["estimators"]["win%d" % w]["dev_pop"] for r in rs
                      if r["estimators"]["win%d" % w]["dev_pop"] is not None])
        th, lo, hi = bca(x)
        vals.append(th)
        mark = "*" if (lo > 0 or hi < 0) else " "
        line += "%15.6f%s" % (th, mark)
    scale_rows[b] = vals
    print(line)
print("(* = BCa %95 CI sifiri disliyor)")

print()
print("H4 karari — w=20'de yanlilik dengesizlikle buyuyor mu?")
base = None
for b in sorted(by_b, reverse=True):
    x = np.array([r["estimators"]["win20"]["dev_pop"] for r in by_b[b]
                  if r["estimators"]["win20"]["dev_pop"] is not None])
    th, lo, hi = bca(x)
    prev = np.mean([r["prevalence"] for r in by_b[b]])
    if base is None:
        base = th
        print("  b=%+.0f (poz.oran %.3f): %+.6f  [%+.6f, %+.6f]   (referans)" % (b, prev, th, lo, hi))
    else:
        ratio = th / base if base else np.nan
        print("  b=%+.0f (poz.oran %.3f): %+.6f  [%+.6f, %+.6f]   b=0'in %.2f kati"
              % (b, prev, th, lo, hi, ratio))

print()
mags = [abs(scale_rows[b][0]) for b in sorted(by_b, reverse=True)]
mono = all(mags[i] <= mags[i + 1] + 1e-9 for i in range(len(mags) - 1))
print("  buyukluk dengesizlikle monoton artiyor mu: %s" % ("EVET" if mono else "HAYIR"))
print("  H4: %s" % ("DESTEKLENDI" if mono and mags[-1] > 1.5 * mags[0] else
                    "kismen/desteklenmedi — asagidaki degerlere bak"))

out = BASE / "evidence/stage2_summary.json"
out.write_text(json.dumps({
    "parity_ok": raw["parity_ok"],
    "windows": WINDOWS,
    "bias_by_b_window": {str(b): {str(w): v for w, v in zip(WINDOWS, scale_rows[b])} for b in scale_rows},
    "prevalence_by_b": {str(b): float(np.mean([r["prevalence"] for r in by_b[b]])) for b in by_b},
    "monotone_in_imbalance": bool(mono),
}, indent=1), encoding="utf-8")
print()
print("cikti:", out)
