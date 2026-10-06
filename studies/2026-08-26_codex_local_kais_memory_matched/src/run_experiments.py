"""Durable unit runner for the prospective memory-matched MemSoftMCC study."""
from __future__ import annotations

import argparse
import os
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PROJECT = RUN.parents[1]
sys.path.insert(0, str(HERE))

from memsoftmcc_matched import (  # noqa: E402
    WINDOWS, atomic_csv, atomic_json, atomic_npz, estimator_specs,
    hard_mcc, kernel_weights, load_ordered_dataset, matched_lambda,
    online_probabilities, run_traces, sha256_file, soft_mcc, valid_unit,
    weighted_ap,
)

CACHE = Path(os.environ.get("SOFTMCC_DATA_CACHE", str(RUN / "local_data_cache")))
SEEDS = tuple(range(1000, 1030))
REGIMES = ("abrupt", "gradual", "recurring", "stationary")
DATASETS = ("electricity", "ozone_level", "insects_abrupt_balanced",
            "insects_gradual_balanced")
LEARNERS = ("sgd_logistic", "gaussian_nb")


def code_meta():
    return {
        "core_sha256": sha256_file(HERE / "memsoftmcc_matched.py"),
        "runner_sha256": sha256_file(HERE / "run_experiments.py"),
    }


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def jsonable_rows(rows):
    clean = []
    for row in rows:
        out = {}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            out[key] = value
        clean.append(out)
    return clean


def unit_paths(kind, unit_id, smoke=False):
    root = RUN / ("smoke" if smoke else "raw")
    status_root = RUN / ("smoke/status_units" if smoke else "status_units")
    root = root / f"{kind}_units"
    return root / f"{unit_id}.npz", status_root / f"{kind}_{unit_id}.json"


def complete_status(status_path, output_path, started, elapsed, meta):
    meta = {**meta, **code_meta()}
    payload = {
        "status": "complete", "started_at": started, "ended_at": now_iso(),
        "elapsed_seconds": elapsed, "output": str(output_path.relative_to(RUN)),
        "bytes": output_path.stat().st_size, "sha256": sha256_file(output_path),
        "meta": meta,
    }
    atomic_json(status_path, payload)


def parse_method(name):
    if name == "cum":
        return "cumulative", 0, 1.0
    w = int(name[3:])
    return ("window" if name.startswith("win") else "fading"), w, matched_lambda(w)


def synthetic_unit(regime, seed, smoke=False):
    output, status = unit_paths("synthetic", f"{regime}_{seed}", smoke=smoke)
    if valid_unit(output, status, expected_meta=code_meta()):
        return "skipped", output
    started, tic = now_iso(), time.perf_counter()
    if smoke:
        n_pre, n_post, gradual, period, windows = 400, 400, 100, 200, (20, 40, 100)
    else:
        n_pre, n_post, gradual, period, windows = 3000, 3000, 1000, 1000, WINDOWS
    from memsoftmcc_matched import make_synthetic
    p, y, t0, eval_start, meta = make_synthetic(
        regime, seed, n_pre=n_pre, n_post=n_post, gradual=gradual, period=period)
    traces = run_traces(p, y, windows=windows)
    names = list(traces)
    trace_matrix = np.vstack([traces[name] for name in names])
    target_y, target_p = y[eval_start:], p[eval_start:]
    target = soft_mcc(target_y, target_p)
    horizon = min(1000 if not smoke else 200, len(p) - eval_start)
    tail = min(1000 if not smoke else 200, len(p) - eval_start)
    metric_rows = []
    for name in names:
        kernel, w, lam = parse_method(name)
        tr = traces[name]
        h2 = min(2 * w if w else horizon, len(p) - eval_start)
        metric_rows.append({
            "stream": "synthetic", "regime": regime, "seed": seed,
            "estimator": name, "kernel": kernel, "w": w, "lambda": lam,
            "target": target,
            "iae1000": float(np.nanmean(np.abs(tr[eval_start:eval_start + horizon] - target))),
            "iae_2w": float(np.nanmean(np.abs(tr[eval_start:eval_start + h2] - target))),
            "tail_gap": float(abs(np.nanmean(tr[-tail:]) - target)),
            "trace_sd": float(np.nanstd(tr[-tail:])),
            "undefined_rate": float(np.mean(~np.isfinite(tr))),
            "eval_start": eval_start, "eval_horizon": horizon,
        })

    family_rows = []
    if regime == "abrupt":
        targets = {
            "soft_mcc": soft_mcc(y[t0:], p[t0:]),
            "hard_mcc": hard_mcc(y[t0:], p[t0:]),
            "auprc": weighted_ap(y[t0:], p[t0:]),
        }
        for w in windows:
            lam = matched_lambda(w)
            for k in sorted({max(1, w // 2), w, min(2 * w, len(p) - t0)}):
                stop = t0 + k
                for kernel in ("cumulative", "window", "fading"):
                    wt = kernel_weights(stop, kernel, w, lam)
                    estimates = {
                        "soft_mcc": soft_mcc(y[:stop], p[:stop], weights=wt),
                        "hard_mcc": hard_mcc(y[:stop], p[:stop], weights=wt),
                        "auprc": weighted_ap(y[:stop], p[:stop], weights=wt),
                    }
                    for metric, estimate in estimates.items():
                        family_rows.append({
                            "seed": seed, "w": w, "lambda": lam, "k": k,
                            "k_over_w": k / w, "kernel": kernel,
                            "metric": metric, "estimate": estimate,
                            "target": targets[metric],
                            "absolute_gap": abs(estimate - targets[metric]),
                        })

    atomic_npz(
        output, p=p, y=y, t0=np.array(t0), eval_start=np.array(eval_start),
        method_names=np.array(names), traces=trace_matrix,
        metrics_json=np.array(json.dumps(jsonable_rows(metric_rows))),
        family_json=np.array(json.dumps(jsonable_rows(family_rows))),
        metadata_json=np.array(json.dumps(meta)),
    )
    elapsed = time.perf_counter() - tic
    complete_status(status, output, started, elapsed,
                    {"kind": "synthetic", "regime": regime, "seed": seed})
    return "completed", output


def ordered_unit(dataset, learner, smoke=False):
    output, status = unit_paths("ordered", f"{dataset}_{learner}", smoke=smoke)
    if valid_unit(output, status, expected_meta=code_meta()):
        return "skipped", output
    started, tic = now_iso(), time.perf_counter()
    x, y, meta = load_ordered_dataset(dataset, CACHE)
    warmup = 200 if smoke else 500
    if smoke:
        n = min(len(y), 1600)
        x, y = x[:n], y[:n]
        meta = dict(meta)
        meta["change_points"] = [cp for cp in meta.get("change_points", []) if cp < n]
    probs = online_probabilities(x, y, learner, warmup=warmup)
    p, yy = probs[warmup:], y[warmup:].astype(float)
    windows = (20, 40, 100) if smoke else WINDOWS
    traces = run_traces(p, yy, windows=windows)
    names = list(traces)
    horizon = 200 if smoke else 500
    stride = horizon
    first_anchor = max(windows)
    anchor_rows = []
    for anchor in range(first_anchor, len(p) - horizon + 1, stride):
        target = soft_mcc(yy[anchor:anchor + horizon], p[anchor:anchor + horizon])
        for name in names:
            kernel, w, lam = parse_method(name)
            anchor_rows.append({
                "dataset": dataset, "learner": learner, "anchor": anchor + warmup,
                "estimator": name, "kernel": kernel, "w": w, "lambda": lam,
                "current_value": float(traces[name][anchor - 1]),
                "future_target": target,
                "future_gap": float(abs(traces[name][anchor - 1] - target)),
            })
    cp_rows = []
    for cp in meta.get("change_points", []):
        rel = cp - warmup
        if rel < first_anchor or rel + horizon > len(p):
            continue
        target = soft_mcc(yy[rel:rel + horizon], p[rel:rel + horizon])
        for name in names:
            kernel, w, lam = parse_method(name)
            cp_rows.append({
                "dataset": dataset, "learner": learner, "change_point": cp,
                "estimator": name, "kernel": kernel, "w": w, "lambda": lam,
                "post_target": target,
                "response_iae": float(np.nanmean(np.abs(traces[name][rel:rel + horizon] - target))),
            })
    anchors = pd.DataFrame(anchor_rows)
    cps = pd.DataFrame(cp_rows)
    summary_rows = []
    for name in names:
        kernel, w, lam = parse_method(name)
        sub = anchors[anchors.estimator == name]
        cp_sub = cps[cps.estimator == name] if not cps.empty else pd.DataFrame()
        summary_rows.append({
            "dataset": dataset, "learner": learner, "estimator": name,
            "kernel": kernel, "w": w, "lambda": lam,
            "n_anchors": int(len(sub)),
            "n_valid_anchors": int(sub.future_gap.notna().sum()),
            "mean_future_gap": float(sub.future_gap.mean()),
            "median_future_gap": float(sub.future_gap.median()),
            "undefined_trace_rate": float(np.mean(~np.isfinite(traces[name]))),
            "n_change_points": int(len(cp_sub)),
            "mean_change_response_iae": (float(cp_sub.response_iae.mean()) if len(cp_sub) else None),
        })
    atomic_npz(
        output, p=p, y=yy, method_names=np.array(names),
        traces=np.vstack([traces[name] for name in names]),
        summary_json=np.array(json.dumps(jsonable_rows(summary_rows))),
        anchors_json=np.array(json.dumps(jsonable_rows(anchor_rows))),
        change_points_json=np.array(json.dumps(jsonable_rows(cp_rows))),
        metadata_json=np.array(json.dumps(meta)),
    )
    elapsed = time.perf_counter() - tic
    complete_status(status, output, started, elapsed,
                    {"kind": "ordered", "dataset": dataset, "learner": learner,
                     "warmup": warmup,
                     "sgd_schedule": ("constant_eta0_0.001_10_warmup_passes"
                                      if learner == "sgd_logistic" else "not_applicable")})
    return "completed", output


def load_json_array(npz, key):
    return json.loads(str(npz[key].item()))


def aggregate(smoke=False):
    root = RUN / ("smoke" if smoke else "raw")
    synthetic_rows, family_rows, ordered_rows, anchor_rows, cp_rows = [], [], [], [], []
    for path in sorted((root / "synthetic_units").glob("*.npz")):
        with np.load(path, allow_pickle=False) as dat:
            synthetic_rows.extend(load_json_array(dat, "metrics_json"))
            family_rows.extend(load_json_array(dat, "family_json"))
    for path in sorted((root / "ordered_units").glob("*.npz")):
        with np.load(path, allow_pickle=False) as dat:
            ordered_rows.extend(load_json_array(dat, "summary_json"))
            anchor_rows.extend(load_json_array(dat, "anchors_json"))
            cp_rows.extend(load_json_array(dat, "change_points_json"))
    atomic_csv(root / "synthetic_metrics.csv", pd.DataFrame(synthetic_rows))
    atomic_csv(root / "metric_family.csv", pd.DataFrame(family_rows))
    atomic_csv(root / "ordered_metrics.csv", pd.DataFrame(ordered_rows))
    atomic_csv(root / "ordered_anchors.csv", pd.DataFrame(anchor_rows))
    atomic_csv(root / "ordered_change_points.csv", pd.DataFrame(cp_rows))


def write_environment():
    payload = {
        "timestamp": now_iso(), "platform": platform.platform(),
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "cache": str(CACHE),
    }
    atomic_json(RUN / "logs/00_environment.json", payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--interrupt-after", type=int, default=0)
    parser.add_argument("--max-units", type=int, default=0)
    args = parser.parse_args()
    write_environment()
    completed_now = 0
    total_seen = 0
    seeds = (1000,) if args.smoke else SEEDS
    datasets = ("electricity", "ozone_level") if args.smoke else DATASETS
    for regime in REGIMES:
        for seed in seeds:
            state, path = synthetic_unit(regime, seed, smoke=args.smoke)
            total_seen += 1
            completed_now += int(state == "completed")
            print(f"[{state}] synthetic {regime} seed={seed} -> {path.name}", flush=True)
            if args.interrupt_after and completed_now >= args.interrupt_after:
                print("[CONTROLLED_INTERRUPT] resumability test requested", flush=True)
                return 75
            if args.max_units and total_seen >= args.max_units:
                aggregate(smoke=args.smoke)
                return 0
    for dataset in datasets:
        for learner in LEARNERS:
            state, path = ordered_unit(dataset, learner, smoke=args.smoke)
            total_seen += 1
            completed_now += int(state == "completed")
            print(f"[{state}] ordered {dataset} learner={learner} -> {path.name}", flush=True)
            if args.interrupt_after and completed_now >= args.interrupt_after:
                print("[CONTROLLED_INTERRUPT] resumability test requested", flush=True)
                return 75
            if args.max_units and total_seen >= args.max_units:
                aggregate(smoke=args.smoke)
                return 0
    aggregate(smoke=args.smoke)
    print(f"[OK] aggregate complete; units seen={total_seen}, newly completed={completed_now}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
