"""Asama 3 — delta-metodu ile analitik yanlilik ve olculen yanlilikla karsilastirma.

MemSoftMCC dort agirlikli soft sayimin nonlineer fonksiyonudur:
    f(TP,FP,FN,TN) = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

Normalize agirliklarla X = sum(w_t c_t)/sum(w_t), Cov(X) = (1/ESS) * Cov(c),
ESS = (sum w)^2 / sum(w^2). Ikinci mertebe delta metodu:

    E[f(X)] - f(mu) ~= (1 / (2*ESS)) * tr(H_f(mu) . Cov(c))

Bu dogrudan bias ~ C/ESS ongorur; C tamamen populasyon momentlerinden gelir.
K3: ongorulen isaret ve buyukluk mertebesi olculenle uyusmazsa teorik katki
dusurulur.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
FROZEN = ROOT / "studies/2026-08-26_codex_local_kais_memory_matched/src"
sys.path.insert(0, str(FROZEN))
from memsoftmcc_matched import _sigmoid, mcc_from_counts  # noqa: E402

POP_N = 200_000
POP_OFFSET = 1_000_000
D = 10


def contributions(seed, b=0.0, n=POP_N, d=D):
    """Her gozlemin dort hucreye koydugu soft kutle vektoru."""
    rng = np.random.default_rng(seed)
    w_a = rng.normal(size=d)
    rng2 = np.random.default_rng(seed + POP_OFFSET)
    x = rng2.normal(size=(n, d))
    truth = _sigmoid(x @ w_a + b)
    y = (rng2.random(n) < truth).astype(float)
    p = _sigmoid(x @ w_a + b)
    return np.column_stack([p * y, p * (1 - y), (1 - p) * y, (1 - p) * (1 - y)])


def f(v):
    return mcc_from_counts(v[0], v[1], v[2], v[3])


def hessian(mu, h_rel=1e-4):
    n = len(mu)
    h = np.maximum(np.abs(mu) * h_rel, 1e-9)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = h[i]
            ej = np.zeros(n); ej[j] = h[j]
            H[i, j] = (f(mu + ei + ej) - f(mu + ei - ej) - f(mu - ei + ej) + f(mu - ei - ej)) \
                / (4 * h[i] * h[j])
    return H


def predicted_C(seeds):
    """C = 0.5 * tr(H . Cov(c)); bias ongorusu C/ESS."""
    vals = []
    for s in seeds:
        c = contributions(s)
        mu = c.mean(axis=0)
        cov = np.cov(c, rowvar=False)
        H = hessian(mu)
        vals.append(0.5 * float(np.trace(H @ cov)))
    return np.array(vals)


if __name__ == "__main__":
    raw = json.loads((ROOT / "studies/2026-08-27_claude_local_finite_ess_bias"
                      / "evidence/stage1_raw.json").read_text(encoding="utf-8"))
    rows, WINDOWS = raw["rows"], raw["windows"]

    seeds = [r["seed"] for r in rows[:60]]
    C = predicted_C(seeds)
    Cbar = C.mean()
    Cse = C.std(ddof=1) / np.sqrt(len(C))
    print("Delta-metodu sabiti  C = 0.5 * tr(H . Cov(c))")
    print("  %d seed uzerinden: C = %+.5f  (SE %.5f)" % (len(C), Cbar, Cse))
    print("  ongoru: bias(w) = C / ESS,  ESS = w  (eslestirme kurali geregi)")
    print()

    print("%-6s %10s %14s %14s %9s" % ("w", "ESS", "ongorulen", "olculen(win)", "oran"))
    print("-" * 60)
    ratios = []
    for w in WINDOWS:
        pred = Cbar / w
        obs = float(np.mean([r["estimators"]["win%d" % w]["dev_pop"] for r in rows]))
        ratios.append(obs / pred)
        print("%-6d %10d %+14.6f %+14.6f %9.2f" % (w, w, pred, obs, obs / pred))

    print()
    same_sign = all((Cbar / w) * np.mean([r["estimators"]["win%d" % w]["dev_pop"] for r in rows]) > 0
                    for w in WINDOWS[:3])
    within_om = all(0.1 <= abs(r) <= 10 for r in ratios[:3])
    print("K3 degerlendirmesi (ilk uc w, CI'si sifiri dislayan veya yakin olanlar):")
    print("  isaret uyumu           : %s" % ("EVET" if same_sign else "HAYIR"))
    print("  buyukluk mertebesi     : %s" % ("EVET" if within_om else "HAYIR"))
    print("  K3 atesledi mi         : %s" % ("HAYIR - teorik katki ayakta" if (same_sign and within_om) else "EVET - teorik katki dusuruldu"))

    out = ROOT / "studies/2026-08-27_claude_local_finite_ess_bias/evidence/stage3_theory.json"
    out.write_text(json.dumps({
        "C_mean": Cbar, "C_se": Cse, "n_seeds": len(C),
        "windows": WINDOWS,
        "predicted": {str(w): Cbar / w for w in WINDOWS},
        "observed_window": {str(w): float(np.mean([r["estimators"]["win%d" % w]["dev_pop"] for r in rows])) for w in WINDOWS},
        "ratios": {str(w): r for w, r in zip(WINDOWS, ratios)},
        "K3_fired": not (same_sign and within_om),
    }, indent=1), encoding="utf-8")
    print()
    print("cikti:", out)
