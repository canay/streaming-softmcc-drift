"""Asama 1 analizi — protokoldeki H1/H2/H3 kararlari.

BCa bootstrap, Holm duzeltmesi, log-log olcekleme. Protokolde kilitlenen
kurallar disinda hicbir secim yapilmaz.
"""
import json
import pathlib

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[3]
RAW = ROOT / "studies/2026-08-27_claude_local_finite_ess_bias/evidence/stage1_raw.json"
d = json.loads(RAW.read_text(encoding="utf-8"))
rows = d["rows"]
WINDOWS = d["windows"]
NBOOT = 20000


def bca(x, nboot=NBOOT, alpha=0.05, seed=20260827):
    x = np.asarray(x, dtype=float)
    n = x.size
    theta = x.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    boots = x[idx].mean(axis=1)
    z0 = stats.norm.ppf((boots < theta).mean()) if 0 < (boots < theta).mean() < 1 else 0.0
    jack = (x.sum() - x) / (n - 1)
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    acc = num / den if den != 0 else 0.0
    zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    lo = np.percentile(boots, 100 * adj(zl))
    hi = np.percentile(boots, 100 * adj(zu))
    return theta, lo, hi


def collect(key):
    out = {}
    for r in rows:
        for nm, v in r["estimators"].items():
            out.setdefault(nm, []).append(v[key])
    return {k: np.array(v) for k, v in out.items()}


def holm(pvals, names):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        run = max(run, val)
        adj[i] = min(1.0, run)
    return dict(zip(names, adj))


print("=" * 84)
print("ASAMA 1 — duragan rejim, %d birim, seed %d-%d" % (d["n_units"], *d["seeds"]))
print("populasyon referansi: %d gozlem | degerlendirme ufku: %d" % (d["pop_n"], d["horizon"]))
prev = np.array([r["prevalence"] for r in rows])
print("populasyon pozitif orani: ort %.3f (min %.3f, maks %.3f)" % (prev.mean(), prev.min(), prev.max()))
print("=" * 84)

for ref, key in (("R_pop (populasyon, 200k)", "dev_pop"), ("R_emp (ampirik, 3k)", "dev_emp")):
    dev = collect(key)
    print()
    print("### Referans: %s" % ref)
    print("%-8s %-7s %5s  %12s  %-26s %10s" % ("tahmin", "kernel", "w", "ort sapma", "BCa %95 CI", "Holm p"))
    print("-" * 84)
    fam = {}
    for kern, pfx in (("window", "win"), ("fading", "fad")):
        names, ps = [], []
        for w in WINDOWS:
            nm = "%s%d" % (pfx, w)
            x = dev[nm]
            _, p = stats.wilcoxon(x)
            names.append(nm); ps.append(p)
        fam[kern] = holm(np.array(ps), names)
    theta, lo, hi = bca(dev["cum"])
    print("%-8s %-7s %5s  %+12.6f  [%+9.6f, %+9.6f] %10s" % ("cum", "cumul", "-", theta, lo, hi, "-"))
    for w in WINDOWS:
        for kern, pfx in (("window", "win"), ("fading", "fad")):
            nm = "%s%d" % (pfx, w)
            theta, lo, hi = bca(dev[nm])
            ph = fam[kern][nm]
            star = " *" if (lo > 0 or hi < 0) and ph < 0.05 else ""
            print("%-8s %-7s %5d  %+12.6f  [%+9.6f, %+9.6f] %10.2e%s" % (nm, kern, w, theta, lo, hi, ph, star))

    # H2 olcekleme
    print()
    print("H2 olcekleme (log|bias| ~ log w):")
    for kern, pfx in (("window", "win"), ("fading", "fad")):
        ws = np.array(WINDOWS, dtype=float)
        bs = np.array([dev["%s%d" % (pfx, w)].mean() for w in WINDOWS])
        if np.all(bs > 0) or np.all(bs < 0):
            sl = np.polyfit(np.log(ws), np.log(np.abs(bs)), 1)[0]
            rng = np.random.default_rng(11)
            sls = []
            for _ in range(2000):
                bb = [np.mean(dev["%s%d" % (pfx, w)][rng.integers(0, len(rows), len(rows))]) for w in WINDOWS]
                bb = np.array(bb)
                if np.all(bb > 0) or np.all(bb < 0):
                    sls.append(np.polyfit(np.log(ws), np.log(np.abs(bb)), 1)[0])
            ci = np.percentile(sls, [2.5, 97.5]) if len(sls) > 100 else (np.nan, np.nan)
            print("  %-7s egim %+.3f  CI [%+.3f, %+.3f]  (-1 iceriyor mu: %s)"
                  % (kern, sl, ci[0], ci[1], "EVET" if ci[0] <= -1 <= ci[1] else "hayir"))
        else:
            print("  %-7s isaret tutarsiz -> H2 bu referans icin reddedilir" % kern)

# H3
print()
print("### H3 — ESS-eslestirilmis cift farki (pencere - fading), R_pop")
dev = collect("dev_pop")
for w in WINDOWS:
    diff = dev["win%d" % w] - dev["fad%d" % w]
    theta, lo, hi = bca(diff)
    print("  w=%-4d %+.6f  [%+.6f, %+.6f]  %s" % (w, theta, lo, hi,
          "FARKLI" if (lo > 0 or hi < 0) else "ayirt edilemez"))

# referanslarin kendi farki
print()
print("### Referans farki: R_emp - R_pop (hedef tanimi kendi yanliligini tasiyor mu?)")
de = np.array([r["R_emp"] - r["R_pop"] for r in rows])
theta, lo, hi = bca(de)
print("  %+.6f  [%+.6f, %+.6f]  %s" % (theta, lo, hi, "SIFIR DISINDA" if (lo > 0 or hi < 0) else "sifiri iceriyor"))
