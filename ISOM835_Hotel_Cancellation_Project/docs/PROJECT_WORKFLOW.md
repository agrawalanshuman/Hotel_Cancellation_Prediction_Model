# Workflow
1. Activate /workspace/isom835-venv (or create a local .venv and install requirements.txt).
2. From the project root run python -m src.quality_control; inspect missing inputs.
3. Validate data; run EDA and timing audit. Review every actual field against booking-creation timing.
4. Run professor-equivalent baseline: python -m src.train --exploratory --numeric-ids --description professor_baseline.
5. Run corrected categorical comparison: python -m src.train --exploratory --description categorical_ids.
6. Run conservative engineered logistic/tree/forest/hist; compare consistent folds. For controlled feature ablation compare the SAME model with/without --engineered. For suspicious-field ablation compare the SAME model and feature setting with/without --exploratory.
7. Run python -m src.compare_models; inspect OOF calibration and secondary grouped/time splits. Report optimistic selection bias after tuning; these are not untouched holdout estimates.
8. Copy docs/selection.example.json to a private reviewed selection file, replace the actual run ID, attach real review evidence, confirm course method permission and team eligibility. Do not toggle approvals without evidence.
9. Set export TEAM_NAME='verified_name'; python -m src.predict --selection path/to/selection.json. It refits the measured configuration, aligns sample order, validates and writes classwork CSV, hashes, provenance and model. Existing files are protected from overwrite.
10. Complete and review memo source, use python -m src.export_memo reviewed.md --reviewed to render one-page PDF. Export refuses PENDING text; verify business language and evidence manually.
11. Run pytest and quality control, then manually submit and record receipts. Five submissions/day maximum. No uploads are automated.

Additional source requirement: at least one random-forest Kaggle submission by Nov 9 4:30 PM ET, within five/day. Forest candidate is available under results/predictions; per-run candidate_validation.json records actual checks. Only submit after timing/rule review. Notebook-top AI disclosure is implemented. See official source review record.
