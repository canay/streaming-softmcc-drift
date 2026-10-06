"""Core utilities for the prospective memory-matched MemSoftMCC study."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

EPS = 1e-12
WINDOWS = (20, 40, 100, 200, 500)


def matched_lambda(w: int) -> float:
    return (float(w) - 1.0) / (float(w) + 1.0)


def memory_pairs(windows=WINDOWS):
    return [(int(w), matched_lambda(int(w))) for w in windows]


def soft_mcc(y_true, y_prob, weights=None, eps: float = EPS) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    wt = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    tp = np.sum(wt * p * y)
    fp = np.sum(wt * p * (1.0 - y))
    fn = np.sum(wt * (1.0 - p) * y)
    tn = np.sum(wt * (1.0 - p) * (1.0 - y))
    return mcc_from_counts(tp, fp, fn, tn, eps=eps)


def hard_mcc(y_true, y_prob, weights=None, threshold: float = 0.5) -> float:
    return soft_mcc(y_true, np.asarray(y_prob) >= threshold, weights=weights)


def weighted_ap(y_true, y_prob, weights=None) -> float:
    y = np.asarray(y_true, dtype=int)
    if np.unique(y).size < 2:
        return float("nan")
    wt = None if weights is None else np.asarray(weights, dtype=float)
    return float(average_precision_score(y, y_prob, sample_weight=wt))


def mcc_from_counts(tp, fp, fn, tn, eps: float = EPS) -> float:
    # Sliding subtraction can leave tiny negative residuals when a mathematically
    # empty cell exits a long-running buffer.  Project the four cells back onto
    # the nonnegative table before testing the MCC margins.  A zero margin makes
    # MCC undefined, so return NaN rather than converting round-off into an
    # unbounded pseudo-score through division by a fixed epsilon.
    counts = np.maximum(np.asarray([tp, fp, fn, tn], dtype=float), 0.0)
    tp, fp, fn, tn = counts
    total = float(counts.sum())
    margins = np.asarray([tp + fp, tp + fn, tn + fp, tn + fn])
    if total <= 0.0 or np.min(margins) <= eps * max(1.0, total):
        return float("nan")
    num = tp * tn - fp * fn
    den = math.sqrt(float(np.prod(margins)))
    return float(np.clip(num / den, -1.0, 1.0))


def moment_soft_mcc(theta, eps: float = EPS) -> float:
    a, b, c = np.asarray(theta, dtype=float)
    den = math.sqrt(max(a * (1.0 - a) * b * (1.0 - b), 0.0))
    return float((c - a * b) / (den + eps))


def moments(y_true, y_prob, weights=None):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    wt = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    wt = wt / np.sum(wt)
    return np.array([np.sum(wt * p), np.sum(wt * y), np.sum(wt * p * y)])


class Cumulative:
    def __init__(self):
        self.s = np.zeros(4, dtype=float)

    def update(self, p, y):
        self.s += contribution(p, y)
        return mcc_from_counts(*self.s)


class Window:
    def __init__(self, w):
        self.w = int(w)
        self.s = np.zeros(4, dtype=float)
        self.buf = deque()

    def update(self, p, y):
        c = contribution(p, y)
        self.buf.append(c)
        self.s += c
        if len(self.buf) > self.w:
            self.s -= self.buf.popleft()
        return mcc_from_counts(*self.s)


class Fading:
    def __init__(self, lam):
        self.lam = float(lam)
        self.s = np.zeros(4, dtype=float)

    def update(self, p, y):
        self.s *= self.lam
        self.s += contribution(p, y)
        return mcc_from_counts(*self.s)


def contribution(p, y):
    p, y = float(p), float(y)
    return np.array([p * y, p * (1.0 - y), (1.0 - p) * y,
                     (1.0 - p) * (1.0 - y)], dtype=float)


def estimator_specs(windows=WINDOWS):
    specs = [("cum", "cumulative", 0, 1.0)]
    for w, lam in memory_pairs(windows):
        specs.append((f"win{w}", "window", w, lam))
        specs.append((f"fad{w}", "fading", w, lam))
    return specs


def run_traces(p, y, windows=WINDOWS):
    trackers = {"cum": Cumulative()}
    for w, lam in memory_pairs(windows):
        trackers[f"win{w}"] = Window(w)
        trackers[f"fad{w}"] = Fading(lam)
    traces = {name: np.empty(len(p), dtype=float) for name in trackers}
    for i, (pi, yi) in enumerate(zip(p, y)):
        for name, tracker in trackers.items():
            traces[name][i] = tracker.update(pi, yi)
    return traces


def _sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def make_synthetic(regime: str, seed: int, n_pre=3000, n_post=3000,
                   gradual=1000, period=1000, d=10):
    rng = np.random.default_rng(seed)
    w_a = rng.normal(size=d)
    w_b = w_a.copy()
    flip = rng.choice(d, size=d // 2, replace=False)
    w_b[flip] *= -1.0

    def draw(w, n):
        x = rng.normal(size=(n, d))
        truth = _sigmoid(x @ w)
        y = (rng.random(n) < truth).astype(float)
        p = _sigmoid(x @ w_a)
        return p, y

    if regime == "abrupt":
        p0, y0 = draw(w_a, n_pre)
        p1, y1 = draw(w_b, n_post)
        p, y = np.r_[p0, p1], np.r_[y0, y1]
        t0, eval_start = n_pre, n_pre
    elif regime == "gradual":
        p0, y0 = draw(w_a, n_pre)
        x = rng.normal(size=(gradual, d))
        frac = np.linspace(0.0, 1.0, gradual)
        use_b = rng.random(gradual) < frac
        truth = np.where(use_b, _sigmoid(x @ w_b), _sigmoid(x @ w_a))
        yg = (rng.random(gradual) < truth).astype(float)
        pg = _sigmoid(x @ w_a)
        p1, y1 = draw(w_b, n_post - gradual)
        p, y = np.r_[p0, pg, p1], np.r_[y0, yg, y1]
        t0, eval_start = n_pre, n_pre + gradual
    elif regime == "recurring":
        ps, ys = [], []
        total = n_pre + n_post
        for start in range(0, total, period):
            w = w_a if (start // period) % 2 == 0 else w_b
            pp, yy = draw(w, min(period, total - start))
            ps.append(pp); ys.append(yy)
        p, y = np.concatenate(ps), np.concatenate(ys)
        t0 = period
        eval_start = max(0, total - period)
    elif regime == "stationary":
        p, y = draw(w_a, n_pre + n_post)
        t0, eval_start = n_pre, n_pre
    else:
        raise ValueError(f"unknown regime: {regime}")
    return p, y, int(t0), int(eval_start), {"flip": flip.tolist(), "d": d}


def kernel_weights(n_prefix: int, kernel: str, w: int, lam: float):
    if kernel == "cumulative":
        return np.ones(n_prefix, dtype=float)
    if kernel == "window":
        wt = np.zeros(n_prefix, dtype=float)
        wt[max(0, n_prefix - w):] = 1.0
        return wt
    if kernel == "fading":
        return np.power(lam, np.arange(n_prefix - 1, -1, -1, dtype=float))
    raise ValueError(kernel)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_csv(path, frame: pd.DataFrame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def valid_unit(output_path, status_path, expected_meta=None) -> bool:
    output_path, status_path = Path(output_path), Path(status_path)
    if not output_path.exists() or not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        meta = status.get("meta", {})
        meta_ok = all(meta.get(key) == value for key, value in (expected_meta or {}).items())
        return (status.get("status") == "complete" and meta_ok and
                status.get("sha256") == sha256_file(output_path))
    except (OSError, ValueError, KeyError):
        return False


def load_ordered_dataset(name: str, cache_dir: Path):
    if name == "electricity":
        with zipfile.ZipFile(cache_dir / "electricity.zip") as zf:
            with zf.open("electricity.csv") as fh:
                df = pd.read_csv(fh)
        y = (df.pop("class") == "UP").astype(int).to_numpy()
        x = df.to_numpy(dtype=float)
        meta = {"order": "chronological", "positive": "UP", "change_points": []}
    elif name == "ozone_level":
        data, _ = arff.loadarff(cache_dir / "ozone_level.arff")
        df = pd.DataFrame(data)
        y = df.pop("Class").astype(float).astype(int).to_numpy()
        for col in df:
            df[col] = pd.to_numeric(df[col].map(lambda v: v.decode() if isinstance(v, bytes) else v), errors="coerce")
        x = df.to_numpy(dtype=float)
        meta = {"order": "chronological source order", "positive": "ozone_day", "change_points": []}
    elif name in {"insects_abrupt_balanced", "insects_gradual_balanced"}:
        filename = "insects_abrupt_balanced.csv" if "abrupt" in name else "insects_gradual_balanced.csv"
        df = pd.read_csv(cache_dir / filename, header=None)
        labels = df.iloc[:, -1].astype(int).to_numpy()
        y = (labels == 2).astype(int)
        x = df.iloc[:, :-1].to_numpy(dtype=float)
        cps = [14352, 19500, 33240, 38682, 39510] if "abrupt" in name else [14028]
        meta = {"order": "controlled real-observation order", "positive": "class_code_2", "change_points": cps}
    else:
        raise ValueError(name)
    return x, y, meta


def online_probabilities(x, y, learner_name: str, warmup=500, seed=20260826):
    if len(y) <= warmup:
        raise ValueError("stream shorter than warm-up")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_warm = imputer.fit_transform(x[:warmup])
    x_warm = scaler.fit_transform(x_warm)
    if learner_name == "sgd_logistic":
        learner = SGDClassifier(loss="log_loss", alpha=1e-4, average=True,
                                learning_rate="constant", eta0=0.001,
                                random_state=seed, max_iter=1, tol=None)
    elif learner_name == "gaussian_nb":
        learner = GaussianNB(var_smoothing=1e-8)
    else:
        raise ValueError(learner_name)
    if learner_name == "sgd_logistic":
        # Repeated passes are confined to the excluded warm-up; evaluated
        # observations remain strictly predict-then-update.
        for _ in range(10):
            learner.partial_fit(x_warm, y[:warmup], classes=np.array([0, 1]))
    else:
        learner.partial_fit(x_warm, y[:warmup], classes=np.array([0, 1]))
    probs = np.full(len(y), np.nan, dtype=float)
    for i in range(warmup, len(y)):
        xi = imputer.transform(x[i:i + 1])
        xi = scaler.transform(xi)
        probs[i] = float(learner.predict_proba(xi)[0, 1])
        learner.partial_fit(xi, y[i:i + 1])
    return probs


def finite_fading_ess(lam: float, t: int) -> float:
    if lam == 1.0:
        return float(t)
    return ((1.0 + lam) / (1.0 - lam)) * ((1.0 - lam ** t) / (1.0 + lam ** t))


def finite_fading_old_mass(lam: float, tau: int, k: int) -> float:
    if k < 0 or tau < 1:
        raise ValueError("tau >= 1 and k >= 0 required")
    if lam == 1.0:
        return tau / (tau + k)
    return (lam ** k) * (1.0 - lam ** tau) / (1.0 - lam ** (tau + k))
