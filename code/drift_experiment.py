"""drift_experiment.py - Prequential streaming-MCC experiments under concept drift.

Paper 3 (Streaming SoftMCC drift). Real fits, fixed seeds, resumable per
(regime, seed) block.
No fabricated numbers. Writes raw per-step traces and a summary CSV under
../results/.

Design
------
We compare three threshold-free soft-MCC estimators on data streams:
  - cumulative (batch-so-far), lam=1
  - sliding window (sizes w)
  - fading / forgetting factor (lam values)

Two stream families:
  A. SYNTHETIC drift streams (controlled, primary evidence). A fixed logistic-style
     scoring rule maps features to probabilities. Drift is injected by changing the
     true data-generating concept at a known change point, so response diagnostics
     can be measured against ground truth.
       * abrupt   : concept flips at t0 (sign of the decision boundary inverts for
                    a subset of features).
       * gradual  : concept transitions over a window [t0, t0+g] with linearly
                    increasing probability of drawing from the new concept.
       * recurring: concept alternates between two regimes.
  B. REAL-DATA stream simulation (secondary, scope-limited). Paper 1 caches
     (creditcard, IoTID20) are turned into a stream by sample order (NOTE: the raw
     data carry no native timestamp, so the streaming order is an explicit
     assumption). A model is trained on an initial warm-up prefix, then probabilities
     are emitted in stream order; drift is injected at a known point by permuting a
     block of feature columns of the post-drift segment, degrading the fixed model.

Monitoring-response diagnostic
------------------------------
For a monitored estimator we set a control band from its pre-drift behavior: mean
and SD over a reference window ending just before t0. A drop is "detected" at the
first post-drift step where the estimator stays below (ref_mean - k*ref_sd) for
`persist` consecutive steps. Delay = detection_step - t0 (in samples). If never
detected within the post-drift segment, delay is right-censored at the segment
length and flagged. This is a descriptive monitoring-response diagnostic, NOT a
calibrated change-point test; the manuscript's primary endpoint is the tracking
gap to a known post-drift target.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from streaming_mcc import (soft_mcc, BatchSoftMCC, WindowSoftMCC, FadingSoftMCC)

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PARENT = os.path.dirname(HERE)
if os.path.basename(HERE).lower() == "scripts" and os.path.basename(SCRIPT_PARENT).lower() == "03_experiments":
    ROOT = os.path.dirname(SCRIPT_PARENT)                 # SCI-SoftMCC_Drift
    RESULTS = os.path.join(SCRIPT_PARENT, "results")      # 03_experiments/results
else:
    ROOT = SCRIPT_PARENT                                  # replication-package root
    RESULTS = os.path.join(SCRIPT_PARENT, "results")      # package/results


def resolve_paper1_data_dir() -> str:
    """Resolve Paper-1 NPZ caches across current, configured, and legacy names."""
    names = ["creditcard_pi10.npz", "iotid20_compact.npz"]
    candidates = [
        os.environ.get("STREAMING_SOFTMCC_DRIFT_DATA_DIR"),
        os.environ.get("DRIFTMCC_DATA_DIR"),
        # Replication-package root first, canonical name before the legacy one,
        # so a clone that ships `data/` resolves without an environment variable.
        os.path.join(ROOT, "data"),
        os.path.join(ROOT, "02_data"),
        os.path.join(ROOT, "..", "SCI-SoftMCC_Theory", "02_data"),
        os.path.join(ROOT, "..", "..", "SCI-SoftMCC_Theory", "02_data"),
        os.path.join(ROOT, "..", "SCI-s1e1-SoftMCC", "02_data"),
    ]
    for cand in candidates:
        if not cand:
            continue
        norm = os.path.normpath(cand)
        if all(os.path.exists(os.path.join(norm, name)) for name in names):
            return norm
    return os.path.normpath(os.path.join(ROOT, "..", "SCI-SoftMCC_Theory", "02_data"))


E1_DATA = resolve_paper1_data_dir()
os.makedirs(RESULTS, exist_ok=True)

WINDOWS = [100, 250, 500]
LAMBDAS = [0.99, 0.95, 0.90]
SEEDS = [42, 43, 44, 45, 46]
N_PRE = 3000      # pre-drift samples
N_POST = 3000     # post-drift samples
GRADUAL_G = 1000  # gradual transition length
REC_PERIOD = 1000 # recurring regime period


# ----------------------------------------------------------------------------
# Synthetic concept-drift streams
# ----------------------------------------------------------------------------
def _concept_probs(X, w_vec, b):
    """Logistic scoring rule -> probabilities for a given concept (w_vec, b)."""
    z = X @ w_vec + b
    return 1.0 / (1.0 + np.exp(-z))


def make_synth_stream(regime, seed, d=10, n_pre=N_PRE, n_post=N_POST):
    """Return (p_stream, y_stream, t0, regime_meta).

    A *fixed deployed model* (concept A scoring rule) emits probabilities for every
    sample. Labels are drawn from the *true* concept, which drifts. Before t0 the
    truth is concept A, so the model is well matched (high MCC). After t0 the truth
    moves toward concept B while the model is unchanged, so its soft MCC should drop.
    """
    rng = np.random.default_rng(seed)
    wA = rng.normal(0, 1, d)
    # concept B: invert sign on half the coordinates (a genuine boundary change)
    wB = wA.copy()
    flip = rng.choice(d, size=d // 2, replace=False)
    wB[flip] *= -1.0
    b = 0.0

    def draw(concept_w, n):
        X = rng.normal(0, 1, size=(n, d))
        pt = _concept_probs(X, concept_w, b)            # true label prob
        y = (rng.uniform(0, 1, n) < pt).astype(float)
        p_model = _concept_probs(X, wA, b)              # deployed model (concept A)
        return X, y, p_model

    if regime == "abrupt":
        XA, yA, pA = draw(wA, n_pre)
        XB, yB, pB = draw(wB, n_post)
        p = np.concatenate([pA, pB]); y = np.concatenate([yA, yB])
        t0 = n_pre
    elif regime == "gradual":
        XA, yA, pA = draw(wA, n_pre)
        # transition: mix concept A and B with linearly increasing B-probability
        g = GRADUAL_G
        Xg = rng.normal(0, 1, size=(g, d))
        frac = np.linspace(0, 1, g)
        use_B = rng.uniform(0, 1, g) < frac
        pt = np.where(use_B,
                      _concept_probs(Xg, wB, b),
                      _concept_probs(Xg, wA, b))
        yg = (rng.uniform(0, 1, g) < pt).astype(float)
        pg = _concept_probs(Xg, wA, b)
        XB, yB, pB = draw(wB, n_post - g)
        p = np.concatenate([pA, pg, pB]); y = np.concatenate([yA, yg, yB])
        t0 = n_pre   # nominal onset; full drift reached at t0+g
    elif regime == "recurring":
        segs_p, segs_y = [], []
        per = REC_PERIOD
        total = n_pre + n_post
        concept = wA
        t = 0
        first_switch = None
        while t < total:
            n = min(per, total - t)
            _, yy, pp = draw(concept, n)
            segs_p.append(pp); segs_y.append(yy)
            if concept is wB and first_switch is None:
                first_switch = t
            concept = wB if concept is wA else wA
            t += n
        p = np.concatenate(segs_p); y = np.concatenate(segs_y)
        t0 = per   # first switch to concept B
    else:
        raise ValueError(regime)

    return p, y, int(t0), {"regime": regime, "d": d, "flip": flip.tolist()}


# ----------------------------------------------------------------------------
# Real-data stream simulation
# ----------------------------------------------------------------------------
def make_real_stream(npz_name, seed, warmup=4000, n_pre=N_PRE, n_post=N_POST):
    """Train a fixed model on a warm-up prefix, then stream; inject drift by
    permuting a block of feature columns in the post-drift segment.

    Returns (p_stream, y_stream, t0, meta) or None if the cache is missing.
    NOTE: streaming order = sample order in the cache (assumption; no native time).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    path = os.path.join(E1_DATA, npz_name)
    if not os.path.exists(path):
        return None
    dat = np.load(path)
    X = dat["X"].astype(float); y = dat["y"].astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X = X[idx]; y = y[idx]

    need = warmup + n_pre + n_post
    if len(X) < need:
        # scale down proportionally for smaller caches
        scale = len(X) / need
        warmup = int(warmup * scale); n_pre = int(n_pre * scale)
        n_post = len(X) - warmup - n_pre
    Xw, yw = X[:warmup], y[:warmup]
    Xs, ys = X[warmup:warmup + n_pre + n_post], y[warmup:warmup + n_pre + n_post]

    base = LogisticRegression(C=1.0, class_weight="balanced",
                              max_iter=2000, solver="liblinear")
    clf = CalibratedClassifierCV(base, method="isotonic", cv=2)
    clf.fit(Xw, yw)

    t0 = n_pre
    Xs_drift = Xs.copy()
    # inject drift: permute a block of feature columns for the post-drift segment
    d = Xs.shape[1]
    cols = rng.choice(d, size=max(2, d // 3), replace=False)
    shuffled = rng.permutation(cols)
    Xs_drift[t0:, cols] = Xs[t0:, shuffled]

    p = clf.predict_proba(Xs_drift)[:, 1]
    return p, ys, int(t0), {"source": npz_name, "warmup": warmup,
                            "n_pre": n_pre, "n_post": len(ys) - n_pre,
                            "cols_permuted": cols.tolist()}


# ----------------------------------------------------------------------------
# Estimator traces and monitoring-response diagnostic
# ----------------------------------------------------------------------------
def run_estimators(p, y):
    est = {"cum": BatchSoftMCC()}
    for w in WINDOWS:
        est[f"win{w}"] = WindowSoftMCC(w)
    for lam in LAMBDAS:
        est[f"fad{lam}"] = FadingSoftMCC(lam)
    traces = {k: np.empty(len(p)) for k in est}
    for i in range(len(p)):
        pi, yi = float(p[i]), float(y[i])
        for k, e in est.items():
            traces[k][i] = e.update(pi, yi)
    return traces


def detection_delay(trace, t0, ref_len=500, k=3.0, persist=20):
    """Return (delay, detected) using a pre-drift control band."""
    ref_start = max(0, t0 - ref_len)
    ref = trace[ref_start:t0]
    if len(ref) < 30:
        return np.nan, False
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-9)
    thr = mu - k * sd
    post = trace[t0:]
    run = 0
    for j, v in enumerate(post):
        if v < thr:
            run += 1
            if run >= persist:
                return float(j - persist + 1), True
        else:
            run = 0
    return float(len(post)), False  # right-censored at segment length


# ----------------------------------------------------------------------------
# Main driver (resumable)
# ----------------------------------------------------------------------------
def main():
    summary_path = os.path.join(RESULTS, "drift_delay_summary.csv")
    done = set()
    if os.path.exists(summary_path):
        prev = pd.read_csv(summary_path)
        done = set(zip(prev["stream"], prev["regime"], prev["seed"], prev["estimator"]))
        rows = prev.to_dict("records")
    else:
        rows = []

    synth_regimes = ["abrupt", "gradual", "recurring"]
    real_streams = [("creditcard_pi10.npz", "real_cc"),
                    ("iotid20_compact.npz", "real_iot")]

    # --- synthetic ---
    for regime in synth_regimes:
        for seed in SEEDS:
            p, y, t0, meta = make_synth_stream(regime, seed)
            traces = run_estimators(p, y)
            # save one trace bundle per (regime, seed=42) for figures
            if seed == 42:
                np.savez_compressed(
                    os.path.join(RESULTS, f"trace_synth_{regime}.npz"),
                    t0=t0, **{k: v for k, v in traces.items()})
            for est_name, tr in traces.items():
                key = ("synth", regime, seed, est_name)
                if key in done:
                    continue
                delay, detected = detection_delay(tr, t0)
                pre = tr[max(0, t0 - 500):t0]
                rows.append({
                    "stream": "synth", "regime": regime, "seed": seed,
                    "estimator": est_name, "t0": t0,
                    "delay": delay, "detected": int(detected),
                    "pre_mean": float(np.mean(pre)), "pre_sd": float(np.std(pre)),
                    "post_min": float(np.min(tr[t0:])),
                })
            print(f"[synth] {regime} seed={seed} done")

    # --- real ---
    for npz_name, tag in real_streams:
        for seed in SEEDS:
            out = make_real_stream(npz_name, seed)
            if out is None:
                print(f"[real] {npz_name} missing, skipped")
                break
            p, y, t0, meta = out
            traces = run_estimators(p, y)
            if seed == 42:
                np.savez_compressed(
                    os.path.join(RESULTS, f"trace_{tag}.npz"),
                    t0=t0, **{k: v for k, v in traces.items()})
            for est_name, tr in traces.items():
                key = ("real", tag, seed, est_name)
                if key in done:
                    continue
                delay, detected = detection_delay(tr, t0)
                pre = tr[max(0, t0 - 500):t0]
                rows.append({
                    "stream": "real", "regime": tag, "seed": seed,
                    "estimator": est_name, "t0": t0,
                    "delay": delay, "detected": int(detected),
                    "pre_mean": float(np.mean(pre)), "pre_sd": float(np.std(pre)),
                    "post_min": float(np.min(tr[t0:])),
                })
            print(f"[real] {tag} seed={seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    print(f"\n[OK] summary -> {summary_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    main()
