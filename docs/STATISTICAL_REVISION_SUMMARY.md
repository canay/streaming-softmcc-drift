# Statistical Revision Summary

Date: 2026-06-22  
Scope: Q1 audit approved revision for `manuscript/main.tex`  
Inputs: `03_experiments/results/q1_drift_paired_effects.csv`, `q1_drift_regime_effects.csv`, `q1_joint_paired_effects.csv`, `q1_metric_family_paired_effects.csv`, and the corresponding raw/summary CSV files under `03_experiments/results/`.

## Evidence / Inference / Interpretation

Evidence: the revised manuscript tables use stored q1-prefixed statistical CSV files, not smoke tests and not hand-set numbers. The active drift/joint/metric/stationary scripts and figure scripts were rerun during the approved revision so current CSV, trace NPZ, and PNG artifacts match the fixed resolver and active manuscript path.

Inference: paired mean differences are estimator tracking gap minus cumulative tracking gap on matched stream/regime/seed units. Negative values therefore mean smaller tracking gap than cumulative SoftMCC.

Interpretation: Holm-adjusted p-values prevent nominal p-values from being presented as stronger evidence than the comparison family supports. BCa intervals are uncertainty summaries for central paired gap differences.

## Drift SoftMCC Family

Primary family: six memory settings versus cumulative SoftMCC across 25 matched drift streams.

| Estimator | Mean gap | Mean cumulative gap | Paired mean diff [BCa 95% CI] | Wilcoxon p | Holm p | Cliff delta | Mean trace SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| fading lambda=0.90 | 0.108753 | 0.159618 | -0.050866 [-0.128287, 0.024361] | 0.219986 | 0.219986 | -0.369600 | 0.186425 |
| fading lambda=0.95 | 0.086641 | 0.159618 | -0.072978 [-0.137507, 0.000626] | 0.066702 | 0.133403 | -0.475200 | 0.150879 |
| fading lambda=0.99 | 0.047288 | 0.159618 | -0.112330 [-0.165808, -0.054273] | 0.001816 | 0.007263 | -0.728000 | 0.077733 |
| window w=100 | 0.078731 | 0.159618 | -0.080887 [-0.145380, -0.010563] | 0.036682 | 0.110046 | -0.520000 | 0.127459 |
| window w=250 | 0.044416 | 0.159618 | -0.115202 [-0.167998, -0.058256] | 0.000912 | 0.004559 | -0.753600 | 0.083133 |
| window w=500 | 0.033123 | 0.159618 | -0.126495 [-0.179560, -0.081257] | 0.000103 | 0.000618 | -0.836800 | 0.056878 |

## Regime Heterogeneity Check

Evidence: paired differences by regime for key moderate-memory settings show that the aggregate improvement is not uniform.

| Estimator | Abrupt | Gradual | Recurring | Credit-card cache | IoTID20 cache |
|---|---:|---:|---:|---:|---:|
| window w=250 | -0.229608 | -0.267427 | 0.094774 | -0.072774 | -0.100976 |
| window w=500 | -0.226800 | -0.268172 | 0.041003 | -0.077022 | -0.101485 |
| fading lambda=0.99 | -0.228852 | -0.267491 | 0.103735 | -0.066184 | -0.102858 |

Inference: positive recurring differences mean local memory has larger pooled-tail gap than cumulative memory under the recurring target definition.

Interpretation: the abstract and results should state that gains are strongest in abrupt/gradual and selected controlled real-data-cache stress tests, while recurring drift exposes the limits of a pooled post-switch target.

## Joint Concept-Calibration Probe

Secondary family: six memory settings versus cumulative SoftMCC across 15 matched joint concept-calibration streams.

| Estimator | Mean gap | Mean cumulative gap | Paired mean diff [BCa 95% CI] | Wilcoxon p | Holm p | Cliff delta | Mean trace SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| fading lambda=0.90 | 0.009786 | 0.242858 | -0.233073 [-0.275012, -0.193281] | 0.000061 | 0.000366 | -1.000000 | 0.136355 |
| fading lambda=0.95 | 0.010865 | 0.242858 | -0.231993 [-0.274440, -0.192097] | 0.000061 | 0.000366 | -1.000000 | 0.091202 |
| fading lambda=0.99 | 0.011427 | 0.242858 | -0.231431 [-0.272869, -0.190145] | 0.000061 | 0.000366 | -1.000000 | 0.034557 |
| window w=100 | 0.011487 | 0.242858 | -0.231372 [-0.273435, -0.185111] | 0.000061 | 0.000366 | -1.000000 | 0.051702 |
| window w=250 | 0.010776 | 0.242858 | -0.232082 [-0.273637, -0.190126] | 0.000061 | 0.000366 | -1.000000 | 0.028124 |
| window w=500 | 0.013144 | 0.242858 | -0.229714 [-0.270143, -0.191330] | 0.000061 | 0.000366 | -1.000000 | 0.015538 |

## Metric-Family Comparator

Secondary boundary check: window `w=500` versus cumulative memory on the same 25 drift streams.

| Family | Cumulative gap | Window 500 gap | Paired mean diff [BCa 95% CI] | Closure | Wilcoxon p | Holm p within family | Cliff delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| SoftMCC | 0.159618 | 0.033123 | -0.126495 [-0.181330, -0.079038] | 79.2% | 0.000103 | 0.000618 | -0.836800 |
| HardMCC@0.5 | 0.199112 | 0.046783 | -0.152329 [-0.224056, -0.085178] | 76.5% | 0.000556 | 0.003338 | -0.699200 |
| AUPRC | 0.127430 | 0.029693 | -0.097737 [-0.137397, -0.065228] | 76.7% | 0.000018 | 0.000055 | -0.801600 |

## Audit Link

Relevant action register IDs: A-003, A-007, A-008, A-009, A-010, A-011, A-012.
