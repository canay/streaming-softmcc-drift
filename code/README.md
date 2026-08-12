# Code

Active scripts copied from `03_experiments/scripts/`.

Recommended run order from this folder after preparing the private data caches described in `../docs/DATA_SOURCES.md`:

```powershell
$env:SOFTMCC_RAW_DIR="C:\path\to\private\raw"
$env:STREAMING_SOFTMCC_DRIFT_DATA_DIR="C:\path\to\private\caches"
python -u real_cache_prep.py all
```

Then run:

```powershell
python -u verify_theory.py
python -u drift_experiment.py
python -u analyze_drift.py
python -u stationary_check.py
python -u joint_drift_calibration.py
python -u metric_family_comparator.py
python -u q1_audit_stats.py
python -u q1_repro_manifest.py
python -u make_figures_drift.py
python -u make_figure_joint.py
```

The figure scripts write to `../figures/` when run from this package. Set `SOFTMCC_DRIFT_FIG_DIR` to override the output directory; `DRIFTMCC_FIG_DIR` is accepted only as a legacy fallback.

The current manuscript evidence is based on the CSV files in `../results/`. Durable transcripts of the 2026-06-23 canonical rerun are included under `../docs/run_transcripts_20260623/`.
