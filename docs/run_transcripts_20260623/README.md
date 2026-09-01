# Canonical Run Transcript Summary

Operation ID: softmcc-drift-canonical-transcripts-20260623
Date/time: 2026-06-23 07:52:24 +03:00
Working directory: <controlled-workspace>\SCI-SoftMCC_Drift
Data dir: <controlled-workspace>\SCI-SoftMCC_Theory\02_data

The canonical internal transcripts remain under `MD/_state/run_transcripts_20260623/`
with `.log` extensions. Public-package copies use `.txt` extensions so repository
hygiene checks do not confuse experiment transcripts with LaTeX build sidecars;
their contents are unchanged.

| ID | Command | Exit | Duration seconds | Log |
|---|---|---:|---:|---|
| 01_verify_theory | `python -u 03_experiments\scripts\verify_theory.py` | 0 | 11.098 | `01_verify_theory.txt` |
| 02_drift_experiment | `python -u 03_experiments\scripts\drift_experiment.py` | 0 | 35.182 | `02_drift_experiment.txt` |
| 03_analyze_drift | `python -u 03_experiments\scripts\analyze_drift.py` | 0 | 30.605 | `03_analyze_drift.txt` |
| 04_stationary_check | `python -u 03_experiments\scripts\stationary_check.py` | 0 | 5.131 | `04_stationary_check.txt` |
| 05_joint_drift_calibration | `python -u 03_experiments\scripts\joint_drift_calibration.py` | 0 | 15.877 | `05_joint_drift_calibration.txt` |
| 06_metric_family_comparator | `python -u 03_experiments\scripts\metric_family_comparator.py` | 0 | 46.163 | `06_metric_family_comparator.txt` |
| 07_q1_audit_stats | `python -u 03_experiments\scripts\q1_audit_stats.py` | 0 | 8.667 | `07_q1_audit_stats.txt` |
| 08_q1_repro_manifest | `python -u 03_experiments\scripts\q1_repro_manifest.py` | 0 | 5.998 | `08_q1_repro_manifest.txt` |
| 09_make_figures_drift | `python -u 03_experiments\scripts\make_figures_drift.py` | 0 | 6.342 | `09_make_figures_drift.txt` |
| 10_make_figure_joint | `python -u 03_experiments\scripts\make_figure_joint.py` | 0 | 3.282 | `10_make_figure_joint.txt` |
