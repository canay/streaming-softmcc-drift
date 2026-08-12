# Replication Notes

Last updated: 2026-06-23

## Environment

Create a Python environment and install the listed packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Evidence Files

The `results/` folder contains the manuscript-facing CSV, summary, and trace artifacts copied from the project workspace. The old imported benchmark output `softmcc_results_20260616_113151.csv` is intentionally excluded because it is not active Streaming SoftMCC drift manuscript evidence.

## Reproducing Core Results

If the local real-data caches are not already available, first obtain the raw
datasets from their original public sources and build local caches. Raw
CSV files and prepared real-data caches are intentionally not redistributed in
this package.

```powershell
$env:SOFTMCC_RAW_DIR="C:\path\to\private\raw"
$env:STREAMING_SOFTMCC_DRIFT_DATA_DIR="C:\path\to\private\caches"
python -u code\real_cache_prep.py all
```

Then, from `code/`, run:

```powershell
python -u verify_theory.py
python -u drift_experiment.py
python -u analyze_drift.py
python -u stationary_check.py
python -u joint_drift_calibration.py
python -u metric_family_comparator.py
python -u q1_audit_stats.py
python -u q1_repro_manifest.py
```

Then regenerate figures:

```powershell
python -u make_figures_drift.py
python -u make_figure_joint.py
```

The real-data-cache runs require the two NPZ caches described in `DATA_SOURCES.md`. For a standalone clone, set `STREAMING_SOFTMCC_DRIFT_DATA_DIR` to the folder containing those files before running the real-data-cache scripts. If the caches are absent, synthetic-only outputs can still be regenerated, but the manuscript tables will not be fully reproduced.

## Durable Transcripts

Canonical rerun transcripts were captured on 2026-06-23 and are included under:

```text
docs/run_transcripts_20260623/
```

The transcript summary is `docs/run_transcripts_20260623/README.md`. The run
used `STREAMING_SOFTMCC_DRIFT_DATA_DIR` pointing to the local sibling
`SCI-SoftMCC_Theory/02_data` cache directory and all listed commands exited with
code 0.

## Current Known Gaps

- Public release still requires the author to create a GitHub repository/tag and
  decide whether to archive the release with a DOI service.
- Raw CSV files and prepared real-data caches are deliberately excluded from the
  package; users must obtain source datasets and build local caches.
