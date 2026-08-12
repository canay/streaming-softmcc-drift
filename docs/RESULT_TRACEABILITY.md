# Result Traceability

| Manuscript item | Package CSV evidence | Project source script |
|---|---|---|
| Abstract tracking gaps and Table `tab:gap` | `results/drift_metrics_raw.csv`, `results/drift_paired_vs_cum.csv` | `code/drift_experiment.py`, `code/analyze_drift.py` |
| Regime breakdown Table `tab:regime` | `results/drift_metrics_raw.csv`, `results/drift_metrics_summary.csv` | `code/analyze_drift.py` |
| Stationary stability Table `tab:stationary` | `results/stationary_equivalence_raw.csv`, `results/stationary_equivalence_summary.csv`; stored five-seed means/SD only, with legacy process-dependent BCa intervals omitted | `code/stationary_check.py` |
| Joint concept-calibration Table `tab:joint` | `results/joint_calib_raw.csv`, `results/joint_calib_paired_vs_cum.csv` | `code/joint_drift_calibration.py` |
| Metric-family comparator Table `tab:metric-family` | `results/metric_family_comparator_raw.csv`, `results/metric_family_comparator_paired.csv`, `results/metric_family_comparator_best.csv` | `code/metric_family_comparator.py` |
| Q1 paired effects and adjusted inference | `results/q1_drift_paired_effects.csv`, `results/q1_joint_paired_effects.csv`, `results/q1_metric_family_paired_effects.csv`, `results/q1_audit_stats_summary.md` | `code/q1_audit_stats.py` |
| Oracle/evaluation-tail reproducibility manifest | `results/q1_reproducibility_manifest.csv` | `code/q1_repro_manifest.py` |
| Round B stationary and metric-family protocol identity | `results/q1_reproducibility_manifest_addendum_20260812.json`; static artifact/code record, no experiment execution | `code/analyze_drift.py`, `code/stationary_check.py`, `code/joint_drift_calibration.py`, `code/metric_family_comparator.py` |
| Drift and real-cache figures | `results/trace_synth_*.npz`, `results/trace_real_*.npz`, `figures/fig1_drift_traces.png`, `figures/fig2_gap_tradeoff.png`, `figures/fig3_real_traces.png` | `code/make_figures_drift.py` |
| Joint calibration figure | `results/trace_joint_*.npz`, `figures/fig4_joint_calib.png` | `code/make_figure_joint.py` |

Do not use `softmcc_results_20260616_113151.csv` for this manuscript. It is legacy provenance and is intentionally not copied into this package.
