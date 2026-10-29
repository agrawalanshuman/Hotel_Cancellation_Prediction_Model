# Actual measured experiments
| Experiment | Mean CV AUC | Fold SD | OOF Brier | Seconds |
|---|---:|---:|---:|---:|
| professor_baseline | 0.88936 | 0.00187 | 0.12608 | 6.3 |
| categorical_ids | 0.90740 | 0.00168 | 0.11727 | 7.9 |
| engineered_logistic | 0.91597 | 0.00165 | 0.11142 | 9.0 |
| conservative_logistic | 0.80280 | 0.00401 | 0.16599 | 6.5 |
| conservative_engineered_logistic | 0.81530 | 0.00343 | 0.16179 | 7.5 |
| decision_tree | 0.81937 | 0.00365 | 0.15581 | 4.7 |
| random_forest | 0.88252 | 0.00377 | 0.13252 | 33.2 |
| hist_conservative | 0.87340 | 0.00439 | 0.13493 | 10.9 |
| hist_exploratory | 0.94471 | 0.00106 | 0.09131 | 13.4 |
| hist_regularized | 0.88323 | 0.00342 | 0.12942 | 15.6 |

All runs: five stratified folds, shuffle=True, seed=835, on uploaded competition training data. IDs excluded; no external data. Fold SD is descriptive, not a confidence interval. Metrics.json records hashes/features/parameters; experiment logs preserve all runs. No Kaggle scores or uploads are inferred.

Categorical-ID improvement vs professor baseline: 0.01804. Engineered logistic vs categorical: 0.00857. Conservative engineered-feature uplift for logistic: 0.01250. Fixed-config hist exploratory minus conservative: 0.07131. This seven-field bundle effect is not a proof of leakage or individual causality. Regularized conservative hist vs default conservative hist: 0.00983.

Highest measured AUC: exploratory HistGradientBoosting, 0.94471; timing not established. Highest conservative candidate: regularized HistGradientBoosting, 0.88323; excludes the seven questioned fields but remaining timing still requires review. No candidate is certified leak-free; no final approved model is selected. No run exceeded 0.985. Optional external boosting/ensembles are unrun pending method approval. Secondary robustness and calibration reports are separate. Five-fold selection may be optimistic after comparisons.

Secondary validation for regularized conservative hist: held-out-agent AUC 0.80541 ± 0.07402; estimated-booking-time AUC 0.77825 ± 0.01932 with 93.53% row coverage. These are diagnostic, not primary benchmarks; missing-agent group is large, estimated dates can reflect revisions, and earliest temporal rows are unscored. This performance drop limits deployment claims. Actual stratified OOF calibration plot and Brier/log loss are available.
