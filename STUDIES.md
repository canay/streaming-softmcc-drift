# Study layers and reproduction

The saved results are organized by their original study identifiers. These
identifiers describe provenance, not independent samples to pool across studies.
Run all commands from the repository root unless a command changes directory.

| Layer | Saved evidence | Replication units |
|---|---|---|
| Earlier tail-gap diagnostics | `results/` | Synthetic drift, constructed cache shifts, stability and joint concept-calibration controls |
| `studies/2026-08-26_codex_local_kais_memory_matched/` | `raw/*.csv` contains replication-unit metrics; `evidence/` contains summaries | 30 seeds per synthetic regime; eight ordered dataset-learner units, with eligibility applied before the 33 reported contrasts |
| `studies/2026-08-27_claude_local_finite_ess_bias/` | `evidence/stage*.json` contains replication-unit summaries | 500 initial stationary units; 800 imbalance conditions; 2500 correction units, including the initial 500; eight fixed ordered pools with 20 permutations each |
| `studies/2026-09-04_codex_vps_dependence/` | Complete `results/analysis.json` | 384 blocks, 49152 trajectories, 120 kernel-specific conditions |
| `studies/2026-09-05_codex_vps_support_covariance/` | Complete `results/analysis.json` | 192 blocks, 12288 trajectories, 1320 score rows and 600 interval rows |

The historical implementation in `code/streaming_mcc.py` uses an epsilon-
regularized denominator. The later studies use explicit undefined-value masks.
The historical layer is descriptive and must not be pooled with the later
layers as a single admissibility analysis. Kernel comparisons, oracle controls,
negative findings and excluded ordered-stream strata remain in the saved output.

## Inspect saved results without running experiments

```text
python verify_saved_results.py
```

This read-only command checks the five paired tracking-error medians and their
sign counts, ordered-stream directions, a finite-bias summary, both September
primary endpoints, the nonuniform correction results, original configuration
bindings and the shipped-file checksums. It does not download data, simulate
trajectories, train learners or write results. It is a bounded consistency check,
not a claim that all original computations were rerun.

## Environments

Use a separate environment for each stack. The repository-root requirements
belong to the earlier diagnostics. The memory-matched run recorded Python
3.12.7, NumPy 2.3.5, pandas 2.3.3, SciPy 1.18.0 and scikit-learn 1.9.0 on
Windows; the finite-bias extension reused that frozen numerical implementation.
The September programs used Python 3.12.3 and NumPy 2.4.6 on Linux aarch64,
with one worker and one numerical thread. They use Linux `resource` and alarm
facilities; run them on Linux rather than assuming native Windows compatibility.
Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for
the September runs. Platform and library differences can affect floating-point
results. Execution timestamps and timing arrays are not reproducible byte for byte.

## Reproduce the memory-matched and finite-bias studies

```text
python studies/2026-08-26_codex_local_kais_memory_matched/src/run_experiments.py
python studies/2026-08-26_codex_local_kais_memory_matched/src/verify_theory.py
python studies/2026-08-26_codex_local_kais_memory_matched/src/analyze_results.py
```

The full matched run downloads the documented ordered benchmarks into a private
cache. Set `SOFTMCC_DATA_CACHE` to a directory outside the repository. Its first
120 units are synthetic; `--max-units 120` runs only that portion. Full analysis
and ordered-pool permutation require the complete ordered outputs. Keep a copy
of the shipped snapshots before running the producers, which write their own
results under the study directory.

```text
python studies/2026-08-27_claude_local_finite_ess_bias/src/run_bias_probe.py --stage 1
python studies/2026-08-27_claude_local_finite_ess_bias/src/run_bias_probe.py --stage 2
python studies/2026-08-27_claude_local_finite_ess_bias/src/stage3_theory.py
python studies/2026-08-27_claude_local_finite_ess_bias/src/run_stages_45.py --stage 4
python studies/2026-08-27_claude_local_finite_ess_bias/src/run_stages_45.py --stage 4 --seed-lo 5000 --seed-hi 7000 --tag b
python studies/2026-08-27_claude_local_finite_ess_bias/src/run_stages_45.py --stage 5
python studies/2026-08-27_claude_local_finite_ess_bias/src/analyze_stage1.py
python studies/2026-08-27_claude_local_finite_ess_bias/src/analyze_stage2.py
python studies/2026-08-27_claude_local_finite_ess_bias/src/analyze_stage4.py
```

The `protocols/` files preserve prospective choices, with public-safe path
substitutions. The correction extension reuses the initial 500 seeds; do not
count them twice as independent evidence. Permuting a fixed empirical pool is
sampling without replacement, not IID sampling. The original finite-bias CLI
transcript was unavailable; it was not reconstructed. Original artifact hashes
and later independent provenance checks provide the recorded evidence boundary.

## Reproduce the September studies on Linux

```text
python studies/2026-09-04_codex_vps_dependence/probe.py run
python studies/2026-09-04_codex_vps_dependence/probe.py analyze
cd studies/2026-09-05_codex_vps_support_covariance
python run_support.py run --output results
python analyze.py
```

Run the full frozen configurations for the reported cardinalities. Smoke and
interruption options test infrastructure and must not be used as manuscript
evidence. The support study imports the adjacent predecessor code but uses its
own seeds and configuration. The original generated-trajectory identities are
listed separately from the shipped-file manifest; those sample-level NPZ files
are intentionally not distributed in this update. Reproduction generates new
receipts bound to the included public adapters. `PUBLIC_COPY_PROVENANCE.json`
records the original and adapted code hashes. Adaptations change paths, not
scientific function bodies or frozen configuration values.

The observed coverage increase remains below nominal coverage. Exact covariance
can produce unstable empirical-Hessian corrections, and the support guard does
not uniformly improve RMSE. These are part of the results, not failed runs to omit.

## Data and release boundaries

Raw third-party datasets, prepared private caches, sample-level trajectory NPZ
files from the new studies, manuscript sources, publisher PDFs and internal
review records are excluded. Replication-unit metric CSVs and JSON summaries
are derived results, not raw feature matrices. The earlier released trace NPZs
contain aggregate monitoring curves and are retained. See
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for source access and the unresolved
original IoTID20 cache selection. Original artifact hashes identify analyzed
snapshots; they do not promise that an unavailable cache-selection procedure or
runtime timing metadata can be reconstructed.
