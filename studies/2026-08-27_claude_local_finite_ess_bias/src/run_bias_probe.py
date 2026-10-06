"""Finite-ESS bias of MemSoftMCC — confirmatory runner.

Protocol: protocols/finite_ess_bias_protocol_20260826.md (locked before run).
Reuses the frozen generator and estimators from the 2026-08-26 run without
modifying them.

Usage:
    python run_bias_probe.py --stage 1
    python run_bias_probe.py --stage 2
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
FROZEN_SRC = ROOT / "studies/2026-08-26_codex_local_kais_memory_matched/src"
OUT = ROOT / "studies/2026-08-27_claude_local_finite_ess_bias"
sys.path.insert(0, str(FROZEN_SRC))

from memsoftmcc_matched import (  # noqa: E402
    WINDOWS, _sigmoid, make_synthetic, matched_lambda, run_traces, soft_mcc,
)

STAGE1_SEEDS = tuple(range(2000, 2500))
STAGE2_SEEDS = tuple(range(3000, 3200))
STAGE2_INTERCEPTS = (0.0, -1.0, -2.0, -3.0)
POP_N = 200_000
POP_OFFSET = 1_000_000
N_PRE = N_POST = 3000
HORIZON = 1000
D = 10


def population_target(seed, b=0.0, n=POP_N, d=D):
    """Same generating weights as the frozen generator, independent large draw."""
    rng = np.random.default_rng(seed)
    w_a = rng.normal(size=d)
    rng2 = np.random.default_rng(seed + POP_OFFSET)
    x = rng2.normal(size=(n, d))
    truth = _sigmoid(x @ w_a + b)
    y = (rng2.random(n) < truth).astype(float)
    p = _sigmoid(x @ w_a + b)
    return float(soft_mcc(y, p)), float(y.mean())


def make_synthetic_b(seed, b, n=N_PRE + N_POST, d=D):
    """Frozen stationary generator plus an intercept. At b=0 it must match exactly."""
    rng = np.random.default_rng(seed)
    w_a = rng.normal(size=d)
    rng.choice(d, size=d // 2, replace=False)  # consume flip draw, as in the frozen code
    x = rng.normal(size=(n, d))
    truth = _sigmoid(x @ w_a + b)
    y = (rng.random(n) < truth).astype(float)
    p = _sigmoid(x @ w_a + b)
    return p, y


def stage1_unit(seed):
    p, y, _t0, es, _meta = make_synthetic("stationary", seed, n_pre=N_PRE, n_post=N_POST)
    traces = run_traces(p, y, windows=WINDOWS)
    r_emp = float(soft_mcc(y[es:], p[es:]))
    r_pop, prevalence = population_target(seed)
    out = {"seed": seed, "R_emp": r_emp, "R_pop": r_pop, "prevalence": prevalence,
           "estimators": {}}
    for name, tr in traces.items():
        seg = tr[es:es + HORIZON]
        seg = seg[np.isfinite(seg)]
        m = float(seg.mean())
        out["estimators"][name] = {"mean": m, "dev_emp": m - r_emp, "dev_pop": m - r_pop,
                                   "n_finite": int(seg.size)}
    return out


def stage2_unit(args):
    seed, b = args
    p, y = make_synthetic_b(seed, b)
    traces = run_traces(p, y, windows=WINDOWS)
    es = N_PRE
    r_pop, prevalence = population_target(seed, b=b)
    out = {"seed": seed, "b": b, "R_pop": r_pop, "prevalence": prevalence,
           "sample_prevalence": float(y.mean()), "estimators": {}}
    for name, tr in traces.items():
        seg = tr[es:es + HORIZON]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            out["estimators"][name] = {"mean": None, "dev_pop": None, "n_finite": 0}
            continue
        m = float(seg.mean())
        out["estimators"][name] = {"mean": m, "dev_pop": m - r_pop, "n_finite": int(seg.size)}
    return out


def parity_gate():
    """b=0 extended generator must reproduce the frozen generator bit-for-bit."""
    for seed in (2000, 2001, 3000):
        p_ref, y_ref, _, _, _ = make_synthetic("stationary", seed, n_pre=N_PRE, n_post=N_POST)
        p_new, y_new = make_synthetic_b(seed, 0.0)
        if not (np.array_equal(p_ref, p_new) and np.array_equal(y_ref, y_new)):
            return False, seed
    return True, None


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest().upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    (OUT / "evidence").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if a.stage == 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(stage1_unit, STAGE1_SEEDS, chunksize=4))
        payload = {"stage": 1, "seeds": [STAGE1_SEEDS[0], STAGE1_SEEDS[-1]],
                   "n_units": len(rows), "windows": list(WINDOWS),
                   "lambda_rule": "(w-1)/(w+1)",
                   "lambdas": {str(w): matched_lambda(w) for w in WINDOWS},
                   "pop_n": POP_N, "horizon": HORIZON, "rows": rows}
    elif a.stage == 2:
        ok, bad = parity_gate()
        if not ok:
            print("PARITE KAPISI BASARISIZ (seed %s) — K4 atesledi, asama 2 kosulmadi." % bad)
            (OUT / "evidence" / "stage2_parity_failed.json").write_text(
                json.dumps({"parity_ok": False, "seed": bad}, indent=2), encoding="utf-8")
            return
        print("parite kapisi: PASS (b=0 dondurulmus ureteci birebir yeniden uretiyor)")
        jobs = [(s, b) for b in STAGE2_INTERCEPTS for s in STAGE2_SEEDS]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(stage2_unit, jobs, chunksize=4))
        payload = {"stage": 2, "parity_ok": True, "intercepts": list(STAGE2_INTERCEPTS),
                   "seeds": [STAGE2_SEEDS[0], STAGE2_SEEDS[-1]],
                   "n_units": len(rows), "windows": list(WINDOWS),
                   "horizon": HORIZON, "rows": rows}
    else:
        raise SystemExit("unknown stage")

    elapsed = time.perf_counter() - t0
    payload["elapsed_seconds"] = elapsed
    payload["frozen_generator_sha256"] = sha(FROZEN_SRC / "memsoftmcc_matched.py")
    payload["protocol_sha256"] = sha(ROOT / "protocols/finite_ess_bias_protocol_20260826.md")
    out_path = OUT / "evidence" / ("stage%d_raw.json" % a.stage)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(out_path)
    print("asama %d bitti: %d birim, %.1f s, %d isci" % (a.stage, len(rows), elapsed, a.workers))
    print("cikti:", out_path)
    print("SHA-256:", sha(out_path))


if __name__ == "__main__":
    main()
