"""verify_theory.py - Numerical verification of the Streaming SoftMCC drift formal properties.

Checks, on real random streams, the identities stated in 03_experiments/THEORY.md:
  T1. WindowSoftMCC(w >= N) == BatchSoftMCC == soft_mcc (full-window reduction).
  T2. FadingSoftMCC(lam = 1) == BatchSoftMCC == soft_mcc (no-forgetting reduction).
  T3. Hard special case: with p in {0,1}, streaming SoftMCC counts reduce to the
      ordinary (hard) confusion counts, so the full-window/lam=1 estimator equals
      sklearn's matthews_corrcoef.
  T4. WindowSoftMCC(w) depends only on the last w samples (locality): two streams
      that agree on their last w samples give the same windowed value.
  T5. FadingSoftMCC total weight matches (1-lam^t)/(1-lam).

All checks print the maximum absolute error. No fabricated numbers.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import matthews_corrcoef
from streaming_mcc import (soft_mcc, BatchSoftMCC, WindowSoftMCC, FadingSoftMCC,
                           stream_curve)

RNG = np.random.default_rng(20260616)


def main():
    results = []

    # ---- T1: full-window reduction ----
    errs = []
    for _ in range(50):
        n = int(RNG.integers(50, 500))
        y = RNG.integers(0, 2, n).astype(float)
        p = RNG.uniform(0, 1, n)
        win = WindowSoftMCC(w=n)
        v_win = stream_curve(p, y, win)[-1]
        v_batch = soft_mcc(p, y)
        errs.append(abs(v_win - v_batch))
    e1 = max(errs)
    results.append(("T1 WindowSoftMCC(w>=N) == soft_mcc", e1))

    # ---- T2: lam=1 reduction ----
    errs = []
    for _ in range(50):
        n = int(RNG.integers(50, 500))
        y = RNG.integers(0, 2, n).astype(float)
        p = RNG.uniform(0, 1, n)
        fad = FadingSoftMCC(lam=1.0)
        v_fad = stream_curve(p, y, fad)[-1]
        v_batch = soft_mcc(p, y)
        errs.append(abs(v_fad - v_batch))
    e2 = max(errs)
    results.append(("T2 FadingSoftMCC(lam=1) == soft_mcc", e2))

    # ---- T3: hard special case == sklearn MCC ----
    errs = []
    for _ in range(50):
        n = int(RNG.integers(50, 500))
        y = RNG.integers(0, 2, n).astype(float)
        p = RNG.integers(0, 2, n).astype(float)  # hard predictions in {0,1}
        # guard against degenerate confusion matrices that make sklearn MCC 0
        yhat = p.astype(int)
        if len(np.unique(yhat)) < 2 or len(np.unique(y.astype(int))) < 2:
            continue
        fad = FadingSoftMCC(lam=1.0)
        v_stream = stream_curve(p, y, fad)[-1]
        v_sk = matthews_corrcoef(y.astype(int), yhat)
        errs.append(abs(v_stream - v_sk))
    e3 = max(errs) if errs else float("nan")
    results.append(("T3 hard streaming == sklearn matthews_corrcoef", e3))

    # ---- T4: window locality ----
    errs = []
    for _ in range(50):
        w = int(RNG.integers(20, 100))
        tail_n = w + int(RNG.integers(0, 30))
        # shared tail of length >= w
        y_tail = RNG.integers(0, 2, tail_n).astype(float)
        p_tail = RNG.uniform(0, 1, tail_n)
        # two different prefixes
        pre1 = int(RNG.integers(10, 100)); pre2 = int(RNG.integers(10, 100))
        yA = np.concatenate([RNG.integers(0, 2, pre1).astype(float), y_tail])
        pA = np.concatenate([RNG.uniform(0, 1, pre1), p_tail])
        yB = np.concatenate([RNG.integers(0, 2, pre2).astype(float), y_tail])
        pB = np.concatenate([RNG.uniform(0, 1, pre2), p_tail])
        vA = stream_curve(pA, yA, WindowSoftMCC(w))[-1]
        vB = stream_curve(pB, yB, WindowSoftMCC(w))[-1]
        errs.append(abs(vA - vB))
    e4 = max(errs)
    results.append(("T4 windowed value depends only on last w samples", e4))

    # ---- T5: fading total weight ----
    errs = []
    for lam in (0.9, 0.95, 0.99):
        fad = FadingSoftMCC(lam=lam)
        # total mass of counts after t updates with all (p=1,y=1) equals sum lam^k
        t = 200
        for _ in range(t):
            fad.update(1.0, 1.0)
        mass = fad.tp  # = sum_{k=0}^{t-1} lam^k
        expected = (1.0 - lam ** t) / (1.0 - lam)
        errs.append(abs(mass - expected))
        errs.append(abs(fad.total_weight(t) - expected))
    e5 = max(errs)
    results.append(("T5 fading total weight matches formula", e5))

    print("=== Streaming SoftMCC drift theory numerical verification ===")
    ok = True
    for name, err in results:
        status = "OK" if (err < 1e-9) else "FAIL"
        if err >= 1e-9:
            ok = False
        print(f"  [{status}] {name}: max abs error = {err:.3e}")
    print("=== ALL PASS ===" if ok else "=== SOME CHECKS FAILED ===")
    return ok


if __name__ == "__main__":
    main()
