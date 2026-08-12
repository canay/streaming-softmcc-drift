"""Build private real-data NPZ caches for the Streaming SoftMCC drift package.

This script intentionally does not download or redistribute raw datasets. Place
the raw public-source CSV files in a private local directory, then point
``SOFTMCC_RAW_DIR`` to that directory. The generated NPZ caches should also stay
outside the public GitHub release unless redistribution rights are explicitly
confirmed.

Expected private raw layout:

    <raw-dir>/creditcard.csv
    <raw-dir>/data/creditcard.csv
    <raw-dir>/IoTID20.csv
    <raw-dir>/IoTID20/data/IoTID20.csv

Outputs:

    creditcard_pi10.npz
    creditcard_pi5.npz
    iotid20_compact.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = Path(os.environ.get("SOFTMCC_RAW_DIR", PACKAGE_ROOT / "data_private" / "raw"))
DEFAULT_OUT_DIR = Path(
    os.environ.get("STREAMING_SOFTMCC_DRIFT_DATA_DIR", PACKAGE_ROOT / "data_private")
)
DEFAULT_SEED = 42


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    options = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(f"None of the expected raw-data paths exists:\n{options}")


def _subsample_to_prevalence(
    X: np.ndarray, y: np.ndarray, target_pi: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_neg = int(round(len(pos) * (1 - target_pi) / target_pi))
    n_neg = min(n_neg, len(neg))
    keep_neg = rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([pos, keep_neg])
    rng.shuffle(idx)
    return X[idx], y[idx]


def prep_creditcard(raw_dir: Path, out_dir: Path, seed: int) -> None:
    csv_path = _first_existing([raw_dir / "creditcard.csv", raw_dir / "data" / "creditcard.csv"])
    df = pd.read_csv(csv_path)
    y = df["Class"].astype(int).to_numpy()
    X = (
        df.drop(columns=["Class", "Time"])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy()
    )
    print(f"creditcard raw: {X.shape}, positives={int(y.sum())}, prevalence={y.mean():.5f}")
    for pi in (0.01, 0.005):
        Xs, ys = _subsample_to_prevalence(X, y, pi, seed)
        out = out_dir / f"creditcard_pi{int(pi * 1000)}.npz"
        np.savez_compressed(out, X=Xs.astype(np.float32), y=ys.astype(np.int8))
        print(f"saved {out}: {Xs.shape}, positives={int(ys.sum())}, prevalence={ys.mean():.5f}")


def prep_iotid20(raw_dir: Path, out_dir: Path, seed: int) -> None:
    csv_path = _first_existing(
        [raw_dir / "IoTID20.csv", raw_dir / "IoTID20" / "data" / "IoTID20.csv"]
    )
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    for chunk in pd.read_csv(csv_path, chunksize=100000, low_memory=False):
        if "Label" in chunk.columns:
            labels = chunk["Label"].astype(str)
        elif "Cat" in chunk.columns:
            labels = chunk["Cat"].astype(str)
        else:
            labels = chunk.iloc[:, -1].astype(str)
        y_chunk = (~labels.str.lower().str.startswith("normal")).astype(int).to_numpy()
        num = chunk.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors="coerce")
        num = num.replace([np.inf, -np.inf], np.nan).fillna(0)
        chunks.append((num.to_numpy(dtype=np.float32), y_chunk.astype(np.int8)))
    X = np.vstack([c[0] for c in chunks])
    y = np.concatenate([c[1] for c in chunks])
    print(f"iotid20 raw: {X.shape}, attack positives={int(y.sum())}, prevalence={y.mean():.4f}")

    if y.mean() > 0.5:
        y = 1 - y
        print(f"flipped label polarity: minority normal class is positive, prevalence={y.mean():.4f}")

    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_neg = min(len(neg), max(40000 - len(pos), len(pos) * 20))
    keep = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
    rng.shuffle(keep)
    Xs, ys = X[keep], y[keep]
    out = out_dir / "iotid20_compact.npz"
    np.savez_compressed(out, X=Xs.astype(np.float32), y=ys.astype(np.int8))
    print(f"saved {out}: {Xs.shape}, positives={int(ys.sum())}, prevalence={ys.mean():.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", choices=["creditcard", "iotid20", "all"], help="Cache family to build."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"raw_dir={raw_dir}")
    print(f"out_dir={out_dir}")

    if args.dataset in {"creditcard", "all"}:
        prep_creditcard(raw_dir, out_dir, args.seed)
    if args.dataset in {"iotid20", "all"}:
        prep_iotid20(raw_dir, out_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
