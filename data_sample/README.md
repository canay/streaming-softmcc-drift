# Data access

This folder intentionally contains no real-data inputs. Obtain source datasets
under their original terms and keep all raw files and prepared caches in private
directories outside this repository.

## You may not need any of this

The earlier synthetic streams can be regenerated without external data, and
the previously released monitoring traces can be redrawn from saved outputs.
The later studies and their distinct requirements are documented in `../STUDIES.md`.

- `../code/` generates the abrupt, gradual, recurring, stationary, and joint
  concept-calibration streams from scratch.
- `../results/` carries the summary CSV files and the trace NPZ artifacts from
  the canonical evidence run, so the figures can be redrawn without rerunning
  the experiments. These are study-generated results, not raw feature matrices.

The following instructions concern the two historical cache-stream experiments.
The later ordered Electricity, Ozone and INSECTS studies use their own acquisition
code and cache variable; see `../STUDIES.md`.

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

## Identities of the historical analyzed caches

These are the cache identities behind the numbers reported in the article:

| Cache | Shape | SHA-256 |
|---|---|---|
| `creditcard_pi10.npz` | `X=(49200, 29)`, `y=(49200,)` | `F25F9D529C04B43F96FFBFD3FD5345229B8BF710DE05C04C70E0022F6D04263B` |
| `iotid20_compact.npz` | `X=(40000, 79)`, `y=(40000,)` | `C91BCF66509F88244B78FDEA711F1FE3345E724D71C6CC3A7580E444CBF28623` |

The original IoTID20 cache's exact row-selection provenance is incomplete.
These hashes identify the analyzed artifacts; they do not establish that the
current builder can recover the same selected rows or bytes. Seed 42 alone
does not close that gap. A newly built cache must be treated as a new input
unless its identity and construction are independently established. Full
provenance and the per-dataset notes are in
[`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md).
