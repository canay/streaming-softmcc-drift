"""Asama 4 (tak-calistir duzeltme) ve Asama 5 (gercek akis permutasyonu).

Protokol eki: protocols/finite_ess_bias_protocol_addendum_20260827.md
"""
import argparse
import glob
import hashlib
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
FROZEN = ROOT / "studies/2026-08-26_codex_local_kais_memory_matched"
OUT = ROOT / "studies/2026-08-27_claude_local_finite_ess_bias"
sys.path.insert(0, str(FROZEN / "src"))

from memsoftmcc_matched import (  # noqa: E402
    WINDOWS, make_synthetic, mcc_from_counts, run_traces, soft_mcc,
)

STAGE1_SEEDS = tuple(range(2000, 2500))
N_PRE = N_POST = 3000
HORIZON = 1000
N_EVAL_POINTS = 10
N_PERM = 20
POP_N = 200_000
POP_OFFSET = 1_000_000
D = 10


def contrib(p, y):
    return np.column_stack([p * y, p * (1 - y), (1 - p) * y, (1 - p) * (1 - y)])


def f4(v):
    return mcc_from_counts(v[0], v[1], v[2], v[3])


def hess(m, h_rel=1e-4):
    h = np.maximum(np.abs(m) * h_rel, 1e-9)
    H = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            ei = np.zeros(4); ei[i] = h[i]
            ej = np.zeros(4); ej[j] = h[j]
            H[i, j] = (f4(m + ei + ej) - f4(m + ei - ej) - f4(m - ei + ej) + f4(m - ei - ej)) \
                / (4 * h[i] * h[j])
    return H


def population_target(seed, d=D, n=POP_N):
    from memsoftmcc_matched import _sigmoid
    rng = np.random.default_rng(seed)
    w_a = rng.normal(size=d)
    rng2 = np.random.default_rng(seed + POP_OFFSET)
    x = rng2.normal(size=(n, d))
    truth = _sigmoid(x @ w_a)
    y = (rng2.random(n) < truth).astype(float)
    p = _sigmoid(x @ w_a)
    return float(soft_mcc(y, p))


def stage4_unit(seed):
    p, y, _t0, es, _m = make_synthetic("stationary", seed, n_pre=N_PRE, n_post=N_POST)
    c = contrib(p, y)
    pop = population_target(seed)
    pts = np.linspace(es, es + HORIZON - 1, N_EVAL_POINTS).astype(int)
    out = {"seed": seed, "R_pop": pop, "w": {}}
    for w in WINDOWS:
        raws, corrs = [], []
        for t in pts:
            win = c[t - w + 1: t + 1]
            m = win.mean(axis=0)
            cov = np.cov(win, rowvar=False, ddof=1)
            raw = f4(m)
            if not np.isfinite(raw):
                continue
            C_hat = 0.5 * float(np.trace(hess(m) @ cov))
            raws.append(raw)
            corrs.append(raw - C_hat / w)
        if not raws:
            continue
        out["w"][str(w)] = {
            "raw_mean": float(np.mean(raws)), "corr_mean": float(np.mean(corrs)),
            "raw_sd": float(np.std(raws, ddof=1)), "corr_sd": float(np.std(corrs, ddof=1)),
            "dev_raw": float(np.mean(raws)) - pop, "dev_corr": float(np.mean(corrs)) - pop,
            "n_points": len(raws),
        }
    return out


def stage5_unit(args):
    path, rep = args
    d = np.load(path, allow_pickle=True)
    p, y = np.asarray(d["p"], float), np.asarray(d["y"], float)
    rng = np.random.default_rng(20260827 + rep * 7919 + (hash(pathlib.Path(path).stem) % 1000))
    idx = rng.permutation(len(p))
    pp, yy = p[idx], y[idx]
    pop = float(soft_mcc(yy, pp))
    traces = run_traces(pp, yy, windows=WINDOWS)
    out = {"unit": pathlib.Path(path).stem, "rep": rep, "n": int(len(p)),
           "prevalence": float(y.mean()), "R_pop": pop, "estimators": {}}
    for nm, tr in traces.items():
        seg = tr[-HORIZON:]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        out["estimators"][nm] = {"mean": float(seg.mean()), "dev_pop": float(seg.mean()) - pop,
                                 "n_finite": int(seg.size)}
    return out


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest().upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--seed-lo", type=int, default=None)
    ap.add_argument("--seed-hi", type=int, default=None)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    (OUT / "evidence").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if a.stage == 4:
        seeds = tuple(range(a.seed_lo, a.seed_hi)) if a.seed_lo is not None else STAGE1_SEEDS
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(stage4_unit, seeds, chunksize=4))
        payload = {"stage": 4, "kernel": "window", "n_units": len(rows), "seeds": [seeds[0], seeds[-1]],
                   "eval_points": N_EVAL_POINTS, "windows": list(WINDOWS), "rows": rows}
    elif a.stage == 5:
        units = sorted(glob.glob(str(FROZEN / "raw/ordered_units/*.npz")))
        jobs = [(u, r) for u in units for r in range(N_PERM)]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(stage5_unit, jobs, chunksize=1))
        payload = {"stage": 5, "n_units": len(units), "n_perm": N_PERM,
                   "n_rows": len(rows), "windows": list(WINDOWS), "rows": rows}
    else:
        raise SystemExit("unknown stage")

    payload["elapsed_seconds"] = time.perf_counter() - t0
    payload["addendum_sha256"] = sha(ROOT / "protocols/finite_ess_bias_protocol_addendum_20260827.md")
    path = OUT / "evidence" / ("stage%d%s_raw.json" % (a.stage, a.tag))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    print("asama %d bitti: %d satir, %.1f s" % (a.stage, len(rows), payload["elapsed_seconds"]))
    print("cikti:", path, sha(path)[:32])


if __name__ == "__main__":
    main()
