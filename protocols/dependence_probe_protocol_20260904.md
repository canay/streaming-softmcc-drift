# Dependency and finite-support probe: prospective protocol

Operation: SOFTMCC-RESEARCH-20260904. Change: MCC-DRIFT-20260904-DEPENDENCE.
Created 2026-09-04 +03:00 before code/compute. Status: locked scientific design.
Prior-art status: worth_testing, per ../01_literature/positioning_research_20260904.md.
This is a controlled mechanism experiment, not a validated new estimator or a
replacement for the existing drift/ordered-stream studies.

## Question and design

Does matching weight-only ESS make the sampling covariance and nonlinear
SoftMCC bias equivalent when observations are serially dependent?

Three calibrated discrete joint distributions use scores p=(0.01,0.25,0.90).
Their positive prevalences are (0.50,0.20,0.05), middle-score masses (0.20,0.20,0.10),
high-score mass (prevalence-0.01-0.24*middle_mass)/0.89 and remaining low mass.
Conditional y|p is Bernoulli(p). Thus all six state probabilities and all
population moments are exact finite sums, with no estimated population oracle.

At each step the state is retained with probability rho or independently
refreshed from that joint distribution; rho=(0,0.3,0.6,0.9). Initial state is
stationary. For z=(p,y,py), lag-h covariance is rho^h Sigma. This dependence
mechanism is deliberately simple and does not model an adaptive learner.

Each distribution/rho cell has 4096 independent trajectories of length 2500.
There are 32 blocks of 128 trajectories per cell, 384 atomic blocks total.
Seeds use NumPy SeedSequence([20260904, distribution_index, rho_index, block]).
Smoke seeds have a separate namespace. Both kernels see identical trajectories.
Windows w=(20,40,100,200,500), fading lambda=(w-1)/(w+1), normalized finite-prefix
weights, endpoint t=2500 only. No post-hoc window/seed/condition selection.

## Exact quantities and estimators

F(a,b,c)=(c-ab)/sqrt(a(1-a)b(1-b)). For normalized weights alpha:
K=sum(alpha^2)+2*sum_h rho^h sum_i alpha_i alpha_(i+h),
Cov(sum alpha*z)=K*Sigma; information ESS=1/K for this refresh-chain family.
Weight ESS remains 1/sum(alpha^2); do not redefine it silently.

Record weighted first/second moments and:
1. raw F;
2. iid plug-in correction F - 0.5 tr(H_F(theta_hat) Sigma_hat)*sum(alpha^2);
3. known-rho plug-in correction F - 0.5 tr(H_F(theta_hat) Sigma_hat)*K;
4. population-oracle second-order subtraction F - 0.5 tr(H_F(theta) Sigma)*K.

Sigma_hat is weighted second moment minus theta_hat theta_hat^T. It is not
claimed unbiased under dependence. Known rho and population moments are
diagnostic oracle inputs, not a deployable algorithm. Analytic Hessian must
match central finite differences on interior fixtures before compute.

F/corrections are undefined when either empirical marginal is outside
(1e-12,1-1e-12). Save null plus validity mask, never epsilon-divide, clip
corrected scores, or silently remove a cell. Record correction range excursions.
Primary covariance endpoint uses a=weighted p and is always defined.
Conditional-on-valid score bias/RMSE and validity probability are jointly
reported. A separately named zero-extension diagnostic maps undefined raw F
to zero; it is not the manuscript's estimator and cannot replace it.
Window one-class probability has the exact check:
(1-b)*(rho+(1-rho)*(1-b))^(w-1)+b*(rho+(1-rho)*b)^(w-1).

## Inference and decision rules

Primary cell: prevalence .50, rho .90, w20. Primary contrast is variance of
weighted p under fading/window, normalized by the exact population variance.
Report empirical ratio, exact ratio, and trajectory-paired bootstrap 95% CI,
5000 resamples, seed 94001. Secondary grids are descriptive, not additional
confirmatory discoveries; report MC SE for moments, bias, RMSE and validity.

SUPPORTED_MECHANISM only if exact primary ratio differs from 1 by >=0.10,
the paired 95% CI excludes 1, and the empirical covariance factors agree with
their exact values within five MC SE. This supports a bounded dependence
mechanism, not novelty or universal kernel superiority.
Otherwise NEGATIVE_OR_INCONCLUSIVE. All numerical discriminators must be saved.
For every correction separately report bias and RMSE changes; smaller bias
alone is not practical superiority. No score-bias expansion is declared
validated in a cell with >1% undefined endpoints.
KILL_IMPLEMENTATION if the Hessian/covariance/unit schema checks fail. Debugging
fixes preserve failed artifacts; scientific config changes need a fresh change ID.
No retuning or extra confirmatory seeds following a negative primary result.

## Review pre-mortem and scope

Classical autocorrelation-adjusted ESS and delta method are NOT new theory.
OEUVRE estimates current-model pointwise loss using two model evaluations;
our stationary emitted-pair generator cannot test its learning-stability gain.
Adding an unfair pseudo-OEUVRE baseline here would answer the wrong question;
a momentwise current-model comparator needs a separate learner experiment.
Kappa-Temporal evaluates skill against persistence, not this sampling bias.
Synthetic states have no human subjects, personal data, or external dataset
license requirement. No human evaluation or paid API is used.

Title/abstract/main manuscript/bib/public repo remain untouched. Existing
rounds and PDF approval remain historical evidence; any later scientific
integration will require affected independent review and render gates.
