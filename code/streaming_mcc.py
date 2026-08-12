"""streaming_mcc.py - Streaming / prequential Matthews correlation coefficient.

Paper 3 (Streaming SoftMCC drift) core. Builds windowed and forgetting-factor (fading) estimators
of the soft MCC of Paper 1 (SoftMCC). The batch soft_mcc kernel is reused from the
Paper-1 Theory project (`SCI-SoftMCC_Theory`).

Soft confusion counts (Paper 1):
    TP = sum_i p_i y_i,  FP = sum_i p_i (1-y_i),
    FN = sum_i (1-p_i) y_i,  TN = sum_i (1-p_i)(1-y_i).
SoftMCC plugs these into the standard MCC formula. Hard MCC is the special case
p_i in {0,1}.

This module exposes three estimators, all driven only by the four running soft
counts, so each per-sample update is O(1) and threshold-free:

  - BatchSoftMCC      : cumulative (all samples seen so far).
  - WindowSoftMCC(w)  : sliding window of the last w samples.
  - FadingSoftMCC(lam): exponential forgetting with factor lam in (0,1].

Reduction (verified numerically in verify_theory.py):
  WindowSoftMCC(w >= N) and FadingSoftMCC(lam = 1) both equal the cumulative
  (batch) SoftMCC on the same stream.

No fabricated numbers; everything is computed from the running counts.
"""
from __future__ import annotations
import numpy as np
from collections import deque

EPS = 1e-12


def soft_mcc(y_true, y_prob, eps=EPS):
    """Batch threshold-free probabilistic MCC (Paper 1 kernel, verbatim)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    tp = np.sum(p * y)
    fp = np.sum(p * (1 - y))
    fn = np.sum((1 - p) * y)
    tn = np.sum((1 - p) * (1 - y))
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    return num / den


def _mcc_from_counts(tp, fp, fn, tn, eps=EPS):
    """SoftMCC evaluated from (possibly weighted) soft confusion counts.

    The product under the root is nonnegative in exact arithmetic; we clip tiny
    negative values from floating-point round-off (e.g. a window holding a single
    class) before the square root.
    """
    num = tp * tn - fp * fn
    prod = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    den = np.sqrt(max(prod, 0.0)) + eps
    return num / den


class BatchSoftMCC:
    """Cumulative soft MCC over all samples seen so far (running counts)."""

    def __init__(self, eps=EPS):
        self.eps = eps
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, p, y):
        p = float(p); y = float(y)
        self.tp += p * y
        self.fp += p * (1 - y)
        self.fn += (1 - p) * y
        self.tn += (1 - p) * (1 - y)
        return self.value()

    def value(self):
        return _mcc_from_counts(self.tp, self.fp, self.fn, self.tn, self.eps)


class WindowSoftMCC:
    """Sliding-window soft MCC over the last w samples.

    Maintains the four soft counts of the active window by adding the incoming
    contribution and subtracting the one that leaves the window. O(1) per update.
    """

    def __init__(self, w, eps=EPS):
        if w < 1:
            raise ValueError("window size w must be >= 1")
        self.w = int(w)
        self.eps = eps
        self.buf = deque()  # stores (tp,fp,fn,tn) contributions of each sample
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, p, y):
        p = float(p); y = float(y)
        c = (p * y, p * (1 - y), (1 - p) * y, (1 - p) * (1 - y))
        self.buf.append(c)
        self.tp += c[0]; self.fp += c[1]; self.fn += c[2]; self.tn += c[3]
        if len(self.buf) > self.w:
            old = self.buf.popleft()
            self.tp -= old[0]; self.fp -= old[1]; self.fn -= old[2]; self.tn -= old[3]
        return self.value()

    def value(self):
        return _mcc_from_counts(self.tp, self.fp, self.fn, self.tn, self.eps)


class FadingSoftMCC:
    """Forgetting-factor (fading) soft MCC.

    Each running count is decayed by lam before the new contribution is added:
        S <- lam * S + contribution.
    With lam = 1 this is exactly the cumulative estimator. With lam < 1 the
    total weight is geometric (~ 1/(1-lam)), so recent samples dominate. This
    quantity is not a Kish effective sample size.
    O(1) per update, no buffer needed.
    """

    def __init__(self, lam, eps=EPS):
        if not (0.0 < lam <= 1.0):
            raise ValueError("forgetting factor lam must be in (0, 1]")
        self.lam = float(lam)
        self.eps = eps
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, p, y):
        p = float(p); y = float(y)
        L = self.lam
        self.tp = L * self.tp + p * y
        self.fp = L * self.fp + p * (1 - y)
        self.fn = L * self.fn + (1 - p) * y
        self.tn = L * self.tn + (1 - p) * (1 - y)
        return self.value()

    def value(self):
        return _mcc_from_counts(self.tp, self.fp, self.fn, self.tn, self.eps)

    def total_weight(self, t):
        """Total fading weight after t updates: (1-lam^t)/(1-lam), -> 1/(1-lam)."""
        if self.lam == 1.0:
            return float(t)
        return (1.0 - self.lam ** t) / (1.0 - self.lam)


def stream_curve(p_stream, y_stream, estimator):
    """Run an estimator over a stream and return the trace of its value."""
    p_stream = np.asarray(p_stream, dtype=float)
    y_stream = np.asarray(y_stream, dtype=float)
    out = np.empty(len(p_stream), dtype=float)
    for i in range(len(p_stream)):
        out[i] = estimator.update(p_stream[i], y_stream[i])
    return out
