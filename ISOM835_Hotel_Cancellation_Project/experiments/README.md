# Experiment history
`experiment_log.csv` is append-only; UUID run IDs prevent overwrite. Failed runs record errors; successful rows describe measured results, not approval. `cv_fold_scores.csv` holds each fold. Run `python -m src.compare_models` to regenerate the derived table and plot. OOF predictions and versioned metadata live under results/reports/<run_id>/. Data hashes, features, estimator parameters, seed and elapsed seconds are recorded. No leaderboard score is inferred from CV.

`leakage_decisions.csv` is the reviewed decision register; observed columns are separately inventoried by `python -m src.leakage_audit`. Complete timing evidence before approving use. `feature_ablation.csv` compares paired five-fold runs. Do not delete unfavorable runs.
