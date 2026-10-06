"""Post-processing that reproduces the reported values of three analyses.

Why this exists. The frozen experiment runs under `experiments/` were executed
with earlier analysis choices, and their own entry points still summarise the
data that way. Three reported quantities were subsequently recomputed from the
SAME frozen raw outputs under corrected choices, and this script is the
executable path from those raw outputs to the values the manuscript prints.

It does NOT re-run any experiment, does NOT modify any frozen artifact, and
reads only the recorded raw files listed in MANIFEST.json.

The three analyses:

1. `common_anchor_contrasts`
   The ordered-stream contrast is a PAIRED difference over anchors at which
   both kernels are defined. The archived `analyze_results.py` instead averages
   each kernel over its own valid anchors, which totals 15 window / 18 fading.
   The paired form totals 9 / 24 and 8 of the 33 contrasts reverse between the
   two. Both totals are produced here so the difference is auditable, and the
   published per-kernel means are reproduced first as a positive control.

2. `bias_scaling_slopes`
   The log|bias| ~ log w slope interval resamples the 500 units ONCE per draw,
   so the five memory settings keep the pairing they have in the data, and it
   applies NO sign filter. The archived `analyze_stage1.py` draws a separate
   index set per memory setting and discards draws whose five means do not
   share a sign. Both are produced here.

3. `table7_mean_deviations`
   Regenerates every stationary mean-deviation cell of Table 7 from the raw
   unit outputs, rather than transcribing them.

Usage:  python reanalysis.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Two trees ship the same three inputs under different layouts: the working tree
# keeps them under experiments/<study>_reclosure/, the public replication package
# under studies/<study>/. The files are byte-identical, so one script serves both
# rather than a fork that would drift. The root is whichever ancestor contains a
# layout that resolves.
_LAYOUTS = (
    ("experiments/2026-08-26_codex_local_kais_memory_matched_reclosure/raw/ordered_anchors.csv",
     "experiments/2026-08-26_codex_local_kais_memory_matched_reclosure/evidence/ordered_matched_contrasts.csv",
     "experiments/2026-08-27_claude_local_finite_ess_bias/evidence/stage1_raw.json"),
    ("studies/2026-08-26_codex_local_kais_memory_matched/raw/ordered_anchors.csv",
     "studies/2026-08-26_codex_local_kais_memory_matched/evidence/ordered_matched_contrasts.csv",
     "studies/2026-08-27_claude_local_finite_ess_bias/evidence/stage1_raw.json"),
)


def _resolve() -> tuple[Path, str, str, str]:
    for base in (HERE, *HERE.parents):
        for layout in _LAYOUTS:
            if all((base / rel).is_file() for rel in layout):
                return (base, *layout)
    raise SystemExit(
        "cannot locate the three recorded inputs from " + str(HERE) + "; expected one of:\n  "
        + "\n  ".join(layout[0] for layout in _LAYOUTS))


ROOT, ORDERED, ORDERED_PUB, STAGE1 = _resolve()

SLOPE_SEED = 11        # the seed the archived slope analysis used
SLOPE_DRAWS = 2000     # the draw count the archived slope analysis used


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _num(text: str) -> float:
    text = (text or "").strip()
    return float("nan") if text == "" else float(text)


# --------------------------------------------------------------------------
# 1. ordered-stream contrasts on the common anchor set
# --------------------------------------------------------------------------
def common_anchor_contrasts() -> dict:
    rows = list(csv.DictReader((ROOT / ORDERED).open(encoding="utf-8-sig")))
    published = list(csv.DictReader((ROOT / ORDERED_PUB).open(encoding="utf-8-sig")))

    gaps: dict[tuple, dict[int, float]] = defaultdict(dict)
    declared: dict[tuple, set] = defaultdict(set)
    for r in rows:
        stratum = (r["dataset"], r["learner"])
        declared[stratum].add(int(r["anchor"]))
        if r["kernel"] == "window":
            gaps[stratum + ("window", int(r["w"]))][int(r["anchor"])] = _num(r["future_gap"])
        elif r["kernel"] == "fading":
            key = stratum + ("fading", round(float(r["lambda"]), 12))
            gaps[key][int(r["anchor"])] = _num(r["future_gap"])

    out, flips = [], 0
    sep = {"window": 0, "fading": 0}
    com = {"window": 0, "fading": 0}
    for p in published:
        ds, ln = p["dataset"], p["learner"]
        w, lam = int(p["w"]), round(float(p["lambda"]), 12)
        wmap = gaps[(ds, ln, "window", w)]
        fmap = gaps[(ds, ln, "fading", lam)]
        wv = np.array([v for v in wmap.values() if np.isfinite(v)])
        fv = np.array([v for v in fmap.values() if np.isfinite(v)])

        # positive control: the separate-subset means must reproduce the record
        assert abs(wv.mean() - float(p["window_mean_future_gap"])) < 1e-9, (ds, ln, w)
        assert abs(fv.mean() - float(p["fading_mean_future_gap"])) < 1e-9, (ds, ln, w)
        separate = wv.mean() - fv.mean()
        assert abs(separate - float(p["window_minus_fading"])) < 1e-9, (ds, ln, w)

        shared = sorted(a for a in wmap
                        if a in fmap and np.isfinite(wmap[a]) and np.isfinite(fmap[a]))
        paired = np.array([wmap[a] - fmap[a] for a in shared])
        common = float(paired.mean())
        n_declared = len(declared[(ds, ln)])
        eligible = paired.size >= 2 and paired.size >= n_declared / 2

        sep["window" if separate < 0 else "fading"] += 1
        com["window" if common < 0 else "fading"] += 1
        flipped = (common < 0) != (separate < 0)
        flips += flipped
        out.append(dict(dataset=ds, learner=ln, w=w,
                        separate_subsets=separate, common_anchors=common,
                        n_window_valid=int(wv.size), n_fading_valid=int(fv.size),
                        n_common=int(paired.size), n_declared=n_declared,
                        eligible_on_intersection=bool(eligible), flipped=bool(flipped)))

    assert len(out) == 33, len(out)
    assert all(r["eligible_on_intersection"] for r in out), "eligibility changed"
    return dict(
        estimand="paired difference over anchors at which both kernels are defined",
        eligibility_rule="at least two anchors and at least half of the declared anchors, "
                         "re-applied to the intersection",
        totals_common_anchors=com,
        totals_separate_subsets=sep,
        direction_flips=int(flips),
        n_contrasts=len(out),
        positive_control="all 33 published separate-subset means reproduced to <1e-9",
        contrasts=out)


# --------------------------------------------------------------------------
# 2. log|bias| ~ log w slope intervals
# --------------------------------------------------------------------------
def bias_scaling_slopes() -> dict:
    raw = json.loads((ROOT / STAGE1).read_text(encoding="utf-8-sig"))
    windows = raw["windows"]
    rows = raw["rows"]
    assert len(rows) == 500, len(rows)
    lw = np.log(np.array(windows, dtype=float))

    result = {}
    for kern, pfx in (("window", "win"), ("fading", "fad")):
        dev = {w: np.array([r["estimators"]["%s%d" % (pfx, w)]["dev_pop"] for r in rows],
                           dtype=float) for w in windows}
        means = np.array([dev[w].mean() for w in windows])
        point = float(np.polyfit(lw, np.log(np.abs(means)), 1)[0])

        # reported form: one index set per draw, no sign filter
        rng = np.random.default_rng(SLOPE_SEED)
        paired, shared_sign = [], 0
        for _ in range(SLOPE_DRAWS):
            idx = rng.integers(0, len(rows), len(rows))
            bb = np.array([dev[w][idx].mean() for w in windows])
            shared_sign += bool(np.all(bb > 0) or np.all(bb < 0))
            paired.append(np.polyfit(lw, np.log(np.abs(bb)), 1)[0])
        lo, hi = (float(x) for x in np.percentile(paired, [2.5, 97.5]))

        # archived form, reproduced so the difference is auditable
        rng2 = np.random.default_rng(SLOPE_SEED)
        kept = []
        for _ in range(SLOPE_DRAWS):
            bb = np.array([dev[w][rng2.integers(0, len(rows), len(rows))].mean()
                           for w in windows])
            if np.all(bb > 0) or np.all(bb < 0):
                kept.append(np.polyfit(lw, np.log(np.abs(bb)), 1)[0])
        alo, ahi = (float(x) for x in np.percentile(kept, [2.5, 97.5]))

        result[kern] = dict(
            point_slope=point,
            width_means={str(w): float(dev[w].mean()) for w in windows},
            reported=dict(resampling="one unit index set per draw, shared by all five "
                                     "memory settings; no sign filter",
                          seed=SLOPE_SEED, draws=SLOPE_DRAWS, kept=SLOPE_DRAWS,
                          ci_2p5_97p5=[lo, hi], contains_minus_one=lo <= -1 <= hi,
                          draws_whose_five_means_share_a_sign=int(shared_sign)),
            archived=dict(resampling="a separate index set per memory setting; draws whose "
                                     "five means do not share a sign are discarded",
                          seed=SLOPE_SEED, draws=SLOPE_DRAWS, kept=len(kept),
                          ci_2p5_97p5=[alo, ahi], contains_minus_one=alo <= -1 <= ahi),
            correlation_shortest_longest=float(
                np.corrcoef(dev[windows[0]], dev[windows[-1]])[0, 1]))
    return result


# --------------------------------------------------------------------------
# 3. Table 7 stationary mean deviations
# --------------------------------------------------------------------------
def table7_mean_deviations() -> dict:
    raw = json.loads((ROOT / STAGE1).read_text(encoding="utf-8-sig"))
    rows = raw["rows"]
    windows = raw["windows"]
    cells = {}
    for key in ["cum"] + ["%s%d" % (p, w) for p in ("win", "fad") for w in windows]:
        vals = np.array([r["estimators"][key]["dev_pop"] for r in rows], dtype=float)
        assert vals.size == 500, key
        mean = float(vals.mean())
        q = Decimal(repr(mean)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)
        cells[key] = dict(mean=mean, rounded_5dp="%+.5f" % q, n=int(vals.size))
    return dict(rounding="half-up at five decimal places, taken from the raw unit values "
                         "rather than from an intermediate rounded figure",
                cells=cells)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inputs = {rel: sha256(ROOT / rel) for rel in (ORDERED, ORDERED_PUB, STAGE1)}
    for rel in inputs:
        assert (ROOT / rel).is_file(), rel

    produced = {
        "common_anchor_contrasts.json": common_anchor_contrasts(),
        "bias_scaling_slopes.json": bias_scaling_slopes(),
        "table7_mean_deviations.json": table7_mean_deviations(),
    }
    outputs = {}
    for name, payload in produced.items():
        path = out / name
        path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2,
                                     default=float) + "\n").encode("utf-8"))
        outputs[name] = sha256(path)

    manifest = dict(
        schema_version=1,
        kind="post_processing_reanalysis",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        script=str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        script_sha256=sha256(Path(__file__).resolve()),
        scope="Recomputes three reported quantities from frozen raw outputs. No "
              "experiment is re-run and no frozen artifact is modified.",
        inputs=inputs,
        outputs=outputs,
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         platform=platform.platform()),
        reported_values=dict(
            ordered_stream_direction_count=produced[
                "common_anchor_contrasts.json"]["totals_common_anchors"],
            ordered_stream_direction_count_archived_form=produced[
                "common_anchor_contrasts.json"]["totals_separate_subsets"],
            ordered_stream_direction_flips=produced[
                "common_anchor_contrasts.json"]["direction_flips"],
            slope_ci=dict(
                window=produced["bias_scaling_slopes.json"]["window"]["reported"][
                    "ci_2p5_97p5"],
                fading=produced["bias_scaling_slopes.json"]["fading"]["reported"][
                    "ci_2p5_97p5"]),
        ),
    )
    (out / "MANIFEST.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    c = produced["common_anchor_contrasts.json"]
    s = produced["bias_scaling_slopes.json"]
    print("ordered-stream contrasts  common %s | archived form %s | flips %d/%d"
          % (c["totals_common_anchors"], c["totals_separate_subsets"],
             c["direction_flips"], c["n_contrasts"]))
    for kern in ("window", "fading"):
        r = s[kern]["reported"]
        a = s[kern]["archived"]
        print("slope %-7s point %+.4f | reported [%+.3f, %+.3f] kept %d "
              "| archived [%+.3f, %+.3f] kept %d"
              % (kern, s[kern]["point_slope"], r["ci_2p5_97p5"][0], r["ci_2p5_97p5"][1],
                 r["kept"], a["ci_2p5_97p5"][0], a["ci_2p5_97p5"][1], a["kept"]))
    t = produced["table7_mean_deviations.json"]["cells"]
    print("Table 7 cells:", " ".join("%s=%s" % (k, v["rounded_5dp"]) for k, v in t.items()))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
