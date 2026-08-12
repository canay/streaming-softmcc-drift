"""q1_repro_manifest.py - protocol manifest for the Drift Q1 revision.

Writes deterministic per-run protocol metadata for the manuscript-facing stream
experiments. This file records generator seeds, drift indices, oracle-segment
rules, evaluation-tail rules, synthetic flipped coordinates, and real-data
permuted-column indices. It does not compute manuscript metrics.
"""
from __future__ import annotations

import json
import os

import pandas as pd

import drift_experiment as de
from joint_drift_calibration import CALIB_REGIMES


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
TAIL = 1000


def _json(value):
    return json.dumps(value, separators=(",", ":"))


def main():
    rows = []

    for regime in ["abrupt", "gradual", "recurring"]:
        for seed in de.SEEDS:
            p, y, t0, meta = de.make_synth_stream(regime, seed)
            post_full_start = t0 + (de.GRADUAL_G if regime == "gradual" else 0) + 500
            rows.append({
                "experiment_family": "main_drift",
                "stream_type": "synthetic",
                "regime": regime,
                "seed": seed,
                "n_total": len(y),
                "n_pre": de.N_PRE,
                "n_post": de.N_POST,
                "t0": t0,
                "transition_width": de.GRADUAL_G if regime == "gradual" else 0,
                "recurring_period": de.REC_PERIOD if regime == "recurring" else "",
                "oracle_segment_start": post_full_start,
                "oracle_segment_end": len(y) - 1,
                "evaluation_tail_start": max(0, len(y) - TAIL),
                "evaluation_tail_end": len(y) - 1,
                "feature_count": meta["d"],
                "flipped_feature_indices": _json(meta["flip"]),
                "real_cache_source": "",
                "warmup": "",
                "permuted_feature_indices": "",
                "calibration_regime": "",
                "calibration_g": "",
                "calibration_b0": "",
                "notes": "Pooled post-onset target for recurring drift; interpreted with regime caveat.",
            })

    for npz_name, tag in [("creditcard_pi10.npz", "real_cc"), ("iotid20_compact.npz", "real_iot")]:
        for seed in de.SEEDS:
            out = de.make_real_stream(npz_name, seed)
            if out is None:
                raise FileNotFoundError(f"Missing cache for {npz_name} under {de.E1_DATA}")
            p, y, t0, meta = out
            rows.append({
                "experiment_family": "main_drift",
                "stream_type": "semi_synthetic_real_cache",
                "regime": tag,
                "seed": seed,
                "n_total": len(y),
                "n_pre": meta["n_pre"],
                "n_post": meta["n_post"],
                "t0": t0,
                "transition_width": 0,
                "recurring_period": "",
                "oracle_segment_start": t0 + 500,
                "oracle_segment_end": len(y) - 1,
                "evaluation_tail_start": max(0, len(y) - TAIL),
                "evaluation_tail_end": len(y) - 1,
                "feature_count": "",
                "flipped_feature_indices": "",
                "real_cache_source": npz_name,
                "warmup": meta["warmup"],
                "permuted_feature_indices": _json(meta["cols_permuted"]),
                "calibration_regime": "",
                "calibration_g": "",
                "calibration_b0": "",
                "notes": "Sample order is a simulation assumption; column permutation is applied only after t0.",
            })

    for regime, params in CALIB_REGIMES.items():
        for seed in de.SEEDS:
            p, y, t0, meta = de.make_synth_stream("abrupt", seed)
            rows.append({
                "experiment_family": "joint_concept_calibration",
                "stream_type": "synthetic",
                "regime": "abrupt",
                "seed": seed,
                "n_total": len(y),
                "n_pre": de.N_PRE,
                "n_post": de.N_POST,
                "t0": t0,
                "transition_width": 0,
                "recurring_period": "",
                "oracle_segment_start": t0 + 500,
                "oracle_segment_end": len(y) - 1,
                "evaluation_tail_start": max(0, len(y) - TAIL),
                "evaluation_tail_end": len(y) - 1,
                "feature_count": meta["d"],
                "flipped_feature_indices": _json(meta["flip"]),
                "real_cache_source": "",
                "warmup": "",
                "permuted_feature_indices": "",
                "calibration_regime": regime,
                "calibration_g": params["g"],
                "calibration_b0": params["b0"],
                "notes": "Calibration distortion is applied to post-drift probabilities only.",
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS, "q1_reproducibility_manifest.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} ({len(out)} rows)")
    return out


if __name__ == "__main__":
    main()
