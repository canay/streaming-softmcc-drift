# Data access

This folder is where the real-data inputs go. It ships empty because the two
real-data experiments use public benchmark datasets that you download from their
original sources, and the prepared caches are built locally from them.

## You may not need any of this

The synthetic results and every manuscript figure can be reproduced without
external data.

- `../code/` generates the abrupt, gradual, recurring, stationary, and joint
  concept-calibration streams from scratch.
- `../results/` carries the summary CSV files and the trace NPZ artifacts from
  the canonical evidence run, so the figures can be redrawn without rerunning
  the experiments. These are study-generated results, not raw feature matrices.

Only the credit-card-fraud and IoTID20 experiments need the steps below.

## Sources

| Dataset | Source |
|---|---|
| Credit-card fraud (MLG-ULB) | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| IoTID20, official project page | https://sites.google.com/view/iot-network-intrusion-dataset/home |
| IoTID20, Kaggle mirror | https://www.kaggle.com/datasets/rohulaminlabid/iotid20-dataset |

The IoTID20 project page grants academic research use and asks to be cited.

## Building the caches

Download `creditcard.csv` and `IoTID20.csv` into a directory of your own, then
run the included builder. It never downloads anything and never writes into the
repository.

```powershell
$env:SOFTMCC_RAW_DIR="C:\path\to\raw"
$env:STREAMING_SOFTMCC_DRIFT_DATA_DIR="C:\path\to\caches"
python -u code\real_cache_prep.py all
```

Either raw layout is accepted:

```text
<raw-dir>/creditcard.csv                 <raw-dir>/data/creditcard.csv
<raw-dir>/IoTID20.csv                    <raw-dir>/IoTID20/data/IoTID20.csv
```

The builder writes `creditcard_pi10.npz`, `creditcard_pi5.npz` and
`iotid20_compact.npz`.

## Pointing the experiments at your caches

The experiment scripts resolve cache files by checking, in order,
`STREAMING_SOFTMCC_DRIFT_DATA_DIR`, the legacy `DRIFTMCC_DATA_DIR`, a local
`data` folder, and two legacy sibling project folders. For a standalone clone,
set `STREAMING_SOFTMCC_DRIFT_DATA_DIR` and ignore the rest.

## Confirming you rebuilt the same caches

These are the cache identities behind the numbers reported in the article:

| Cache | Shape | SHA-256 |
|---|---|---|
| `creditcard_pi10.npz` | `X=(49200, 29)`, `y=(49200,)` | `F25F9D529C04B43F96FFBFD3FD5345229B8BF710DE05C04C70E0022F6D04263B` |
| `iotid20_compact.npz` | `X=(40000, 79)`, `y=(40000,)` | `C91BCF66509F88244B78FDEA711F1FE3345E724D71C6CC3A7580E444CBF28623` |

A mismatch usually means the source file was revised at the origin or a
different subsampling seed was used; the builder defaults to seed 42. Full
provenance and the per-dataset notes are in
[`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md).
