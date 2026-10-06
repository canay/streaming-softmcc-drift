"""Algebraic and Monte Carlo checks for the MemSoftMCC tracking proposition."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(HERE))

from memsoftmcc_matched import (  # noqa: E402
    WINDOWS, atomic_csv, atomic_json, finite_fading_ess,
    finite_fading_old_mass, matched_lambda, moment_soft_mcc, moments,
)

ALPHA = 0.05
DELTA = 0.10
TOL = 1e-12
REPS = 400
TAU = 3000


def theta(h, low, prevalence=0.5):
    return np.array([
        prevalence * h + (1.0 - prevalence) * low,
        prevalence,
        prevalence * h,
    ], dtype=float)


THETA0 = theta(0.75, 0.25)
THETA1 = theta(0.55, 0.45)


def draw(rng, reps, n, h, low):
    y = (rng.random((reps, n)) < 0.5).astype(float)
    p = np.where(y == 1.0, h, low)
    return p, y


def gradient(theta_vec):
    a, b, c = theta_vec
    f = moment_soft_mcc(theta_vec)
    den = math.sqrt(a * (1.0 - a) * b * (1.0 - b))
    da = -b / den - f * (1.0 - 2.0 * a) / (2.0 * a * (1.0 - a))
    db = -a / den - f * (1.0 - 2.0 * b) / (2.0 * b * (1.0 - b))
    dc = 1.0 / den
    return np.array([da, db, dc])


def main():
    evidence = RUN / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    calibration_rows = []
    max_identity_error = 0.0
    for w in WINDOWS:
        lam = matched_lambda(w)
        win_age = (w - 1.0) / 2.0
        fad_age = lam / (1.0 - lam)
        fad_ess = (1.0 + lam) / (1.0 - lam)
        age_err = abs(win_age - fad_age)
        ess_err = abs(w - fad_ess)
        max_identity_error = max(max_identity_error, age_err, ess_err)
        t = 3000
        weights = lam ** np.arange(t - 1, -1, -1, dtype=float)
        numeric_ess = weights.sum() ** 2 / np.square(weights).sum()
        finite_ess = finite_fading_ess(lam, t)
        max_identity_error = max(max_identity_error, abs(numeric_ess - finite_ess))
        calibration_rows.append({
            "w": w, "lambda": lam, "window_mean_age": win_age,
            "fading_mean_age": fad_age, "window_ess": float(w),
            "fading_asymptotic_ess": fad_ess,
            "fading_total_mass": 1.0 / (1.0 - lam),
            "fading_finite_ess_t3000": finite_ess,
        })

    rng = np.random.default_rng(20260826)
    c_delta = 3.0 / (2.0 * DELTA * (1.0 - DELTA))
    delta_theta = float(np.abs(THETA0 - THETA1).sum())
    sim_rows = []
    total_valid = total_covered = total_margin_fail = 0
    max_old_mass_error = 0.0

    for w in WINDOWS:
        lam = matched_lambda(w)
        for k in sorted({max(1, w // 2), w, 2 * w}):
            p0, y0 = draw(rng, REPS, TAU, 0.75, 0.25)
            p1, y1 = draw(rng, REPS, k, 0.55, 0.45)
            p = np.concatenate([p0, p1], axis=1)
            y = np.concatenate([y0, y1], axis=1)
            for kernel in ("window", "fading"):
                if kernel == "window":
                    weights = np.zeros(TAU + k, dtype=float)
                    weights[-w:] = 1.0
                    q_formula = max(1.0 - k / w, 0.0)
                else:
                    weights = lam ** np.arange(TAU + k - 1, -1, -1, dtype=float)
                    q_formula = finite_fading_old_mass(lam, TAU, k)
                v = weights / weights.sum()
                q_numeric = float(v[:TAU].sum())
                max_old_mass_error = max(max_old_mass_error, abs(q_numeric - q_formula))
                ess = float(1.0 / np.square(v).sum())
                a = (p * v).sum(axis=1)
                b = (y * v).sum(axis=1)
                c = (p * y * v).sum(axis=1)
                empirical = np.column_stack([a, b, c])
                valid = ((a >= DELTA) & (a <= 1.0 - DELTA) &
                         (b >= DELTA) & (b <= 1.0 - DELTA))
                f_hat = np.array([moment_soft_mcc(row) for row in empirical])
                err = np.abs(f_hat - moment_soft_mcc(THETA1))
                noise = 3.0 * math.sqrt(math.log(6.0 / ALPHA) / (2.0 * ess))
                bound = c_delta * (q_formula * delta_theta + noise)
                covered = err <= bound + TOL
                total_valid += int(valid.sum())
                total_covered += int((covered & valid).sum())
                total_margin_fail += int((~valid).sum())
                sim_rows.append({
                    "w": w, "lambda": lam, "k": k, "kernel": kernel,
                    "q_formula": q_formula, "q_numeric": q_numeric,
                    "ess": ess, "bound": bound,
                    "valid_replicates": int(valid.sum()),
                    "margin_failures": int((~valid).sum()),
                    "coverage": float(covered[valid].mean()) if valid.any() else float("nan"),
                    "median_error": float(np.median(err[valid])) if valid.any() else float("nan"),
                    "max_error": float(np.max(err[valid])) if valid.any() else float("nan"),
                    "max_error_bound_ratio": float(np.max(err[valid] / bound)) if valid.any() else float("nan"),
                })

    # Numerical stress check for the conservative coordinate-gradient bound.
    points = []
    while len(points) < 20000:
        a, b = rng.uniform(DELTA, 1.0 - DELTA, size=2)
        lower, upper = max(0.0, a + b - 1.0), min(a, b)
        c = rng.uniform(lower, upper)
        points.append((a, b, c))
    max_gradient_ratio = max(np.max(np.abs(gradient(point))) / c_delta for point in points)

    atomic_csv(evidence / "memory_calibration.csv", pd.DataFrame(calibration_rows))
    atomic_csv(evidence / "theory_bound_simulation.csv", pd.DataFrame(sim_rows))
    valid_coverage = total_covered / total_valid
    margin_failure_rate = total_margin_fail / (total_valid + total_margin_fail)
    verdict = {
        "status": "passed",
        "identity_tolerance": TOL,
        "max_memory_identity_error": max_identity_error,
        "max_old_mass_identity_error": max_old_mass_error,
        "delta": DELTA,
        "alpha": ALPHA,
        "C_delta": c_delta,
        "theta0": THETA0.tolist(),
        "theta1": THETA1.tolist(),
        "valid_replicates": total_valid,
        "coverage_over_valid_states": valid_coverage,
        "margin_failure_rate": margin_failure_rate,
        "max_gradient_to_bound_ratio": max_gradient_ratio,
        "scope": "independent bounded pairs and nondegenerate empirical margins",
    }
    if max_identity_error > TOL or max_old_mass_error > TOL:
        verdict["status"] = "failed_identity"
    if valid_coverage < 1.0 - ALPHA:
        verdict["status"] = "failed_coverage"
    if margin_failure_rate > 0.10:
        verdict["status"] = "failed_margin_gate"
    if max_gradient_ratio > 1.0 + 1e-10:
        verdict["status"] = "failed_gradient_gate"
    atomic_json(evidence / "theory_verification.json", verdict)
    print(json.dumps(verdict, indent=2))
    if verdict["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
