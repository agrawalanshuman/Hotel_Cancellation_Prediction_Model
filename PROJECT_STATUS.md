# PARTIALLY COMPLETE
Suffolk University · ISOM 835 Fall 2026

Created runnable source, nine notebooks, pinned environment, tests, EDA, feature timing register and ten measured five-fold experiments. Data dimensions/schemas/IDs pass. All four members credited; user confirms instructor approval and official team name Ctrl_+_Alt_+_Defeat on October 29, 2026.

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

Best measured exploratory candidate: hist 0.94471. Best conservative candidate: regularized hist 0.88323. Neither is certified leak-free; booking-creation timing and remaining field provenance are unresolved. Candidate predictions may be technically validated in results/predictions; final classwork CSV remains pending review. No Kaggle/Canvas upload or leaderboard score is claimed. Memo/presentation contain real results where available but require team review; final PDF pending. Course sites beyond starter were blocked by network policy; domain additions saved in environment draft, not applied/published. GitHub publication not performed. See requirements audit and quality-control report for precise readiness.

Secondary validation for regularized conservative hist: held-out-agent AUC 0.80541 ± 0.07402; estimated-booking-time AUC 0.77825 ± 0.01932 with 93.53% row coverage. These are diagnostic, not primary benchmarks; missing-agent group is large, estimated dates can reflect revisions, and earliest temporal rows are unscored. This performance drop limits deployment claims. Actual stratified OOF calibration plot and Brier/log loss are available.

Update: official GitHub syllabus/session sources were reviewed at commit 112b263771011178f956525fc3d5585ce1901cfe despite rendered website blocks. Added notebook-top AI disclosure. Required suffolk.edu registration and at least one random-forest Kaggle submission remain pending. Kaggle private rules and Drive references remain inaccessible.

Final current-instance checks: 27 tests passed, 2 skipped (approved final CSV/PDF absent), 0 failed. Nine notebooks structurally valid; EDA, leakage and comparison notebooks executed. Two conservative candidate CSVs passed 35,817-row ID/probability contracts. One-page actual-results DRAFT memo is generated at 11pt. Exact files: docs/FILE_MANIFEST.txt. Environment install/start/network draft saved, not published.
