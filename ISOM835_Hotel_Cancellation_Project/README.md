# ISOM835_Hotel_Cancellation_Project
**Suffolk University · ISOM 835 · Fall 2026**

**Anshuman Agrawal · Mark Belov · Phuc Hoang · Sarah Legault**

Team: **Ctrl_+_Alt_+_Defeat** (configured once as TEAM_NAME in src/config.py). The supplied 2–3 participant rule conflicts with four members; the user confirmed instructor approval on October 29, 2026. This confirmation is recorded in TEAM_COMPLIANCE.md; registration and Canvas membership remain unverified.

Predict hotel-booking cancellation probabilities to help Revenue Management plan capacity and evaluate deposit policy. Ranking quality (ROC-AUC) and calibration are different; good rankings alone cannot establish accurate expected cancellation counts or causal policy effects.

## Data and rules
Only the authorized competition CSVs are used: train 83,573 rows with is_canceled; test 35,817 rows without it; sample_submission defines id,is_canceled output. Uploaded files passed exact dimensions, target, unique/disjoint IDs, feature schemas, sample alignment and known-outcome exclusion checks. Raw data stays private and Git-ignored. Never add external datasets. Use permitted course methods, fold-fitted preprocessing, stratified CV and feature timing review; distrust AUC >0.985 until audited. Five Kaggle submissions/day maximum; record every attempt and verify reset timezone. Select by defensible validation, not public leaderboard.

## Measured results
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

No public/private Kaggle score is known. Top measured model is exploratory boosting (0.94471), with unresolved timing risks. Regularized conservative boosting (0.88323) excludes seven uncertain fields; its remaining inputs still need review. No final certified leak-free model is claimed. Full fold scores, OOF predictions, hashes, timing and parameters are in experiments/ and results/reports/. Largest controlled improvement among same-model feature comparisons was categorical IDs, +0.01804 AUC; adding the seven uncertain fields has a larger bundled effect but is not an approved improvement. Model-family changes are separately documented in MODELING_RESULTS.md.

## Reproduce locally
```bash
cd ISOM835_Hotel_Cancellation_Project
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
# Put only the authorized competition files in data/raw/ (or set DATA_DIR).
python -m src.explore
python -m src.leakage_audit
python -m src.train --exploratory --numeric-ids --description professor_baseline
python -m src.train --exploratory --description categorical_ids
python -m src.train --model hist --engineered
python -m src.compare_models
python -m pytest tests/ -v
python -m src.quality_control
```
Prepared cloud environment: `. /workspace/isom835-venv/bin/activate`. Work in this existing isolated checkout; no worktree needed. Quality control exits nonzero for pending deliverables; this is intentional. The starter uses seed **835**, preserved for all comparisons. Models perform imputation/encoding/scaling inside training folds. Engineered features are row-local. Hist boosting has internal early stopping and regularization; tree/forest are depth/leaf limited. LightGBM/XGBoost and blends remain optional/unrun until allowed. Secondary grouped/time validation and OOF calibration are diagnostic, not substitute leaderboard scores.

To reproduce a final submission: choose a measured run after timing/method review, complete docs/selection.example.json with actual evidence and copy to results/reports/selection.json, then `python -m src.predict --selection results/reports/selection.json`. The exporter checks data hashes, requires review evidence, refits on all labeled rows, aligns sample IDs, checks probabilities, saves `${TEAM_NAME}_submission.csv` under classwork/01_Competition_Submission and records SHA256/provenance. Existing deliverables are never silently overwritten. No final selection is approved yet.

## Directory guide
classwork/: academic deliverables and upload receipts. data/: private raw/processed inputs. notebooks/: official starter plus EDA, timing, baseline, features, trees, boosting, comparison and independent final pipeline. src/: reusable loading, validation, fold pipelines, training and exports. experiments/: append-only measured history and leakage/ablation decisions. results/: generated reports, figures, models and provisional predictions. tests/: executable integrity/contract checks. docs/: workflow, business interpretation, Colab guide, source-review record and final checklist. Exact file inventory: docs/FILE_MANIFEST.txt.

## Google Colab
The notebooks are included in this GitHub project; [open any of the nine notebooks in Colab](notebooks/README.md). Actual Colab runtime execution remains unverified. At https://colab.research.google.com/ use **File → Upload notebook** for any notebooks/00–08 file. Upload/extract a private project ZIP and the authorized CSVs; set PROJECT_ROOT/DATA_DIR in the first cell. See docs/GOOGLE_COLAB_GUIDE.md. Notebook 08 validates data and runs fresh CV independently, then refits/exports only after genuine selected-run review. Do not embed credentials or share private records. The preserved starter has its own official Kaggle login instructions.

## Deliverables and deadlines (America/New_York)
- November 2, 2026: at least one accepted Kaggle submission.
- November 9, 2026, 4:30 PM ET: final Kaggle CSV and identical bytes on Canvas.
- November 9, 2026, 11:59 PM ET: reviewed one-page modeling memo PDF on Canvas.
- November 9, 2026: three-minute team presentation; confirm time.

All academic artifacts live under classwork/. Generated candidate predictions are not final approval. Review the model/timing decisions, run exact submission validator, upload manually to Kaggle Submit Prediction (maximum five/day), then record receipt/score/hash in classwork/04_Submission_Records/. Upload the same CSV bytes to Canvas and verify downloaded hash. Never mark uploaded from local existence. Public/private scores remain pending until confirmed/revealed. Final memo PDF is pending review; no fictional completed deliverable is provided.

## Official resources and progress
Accessible/read: [instructor starter](https://raw.githubusercontent.com/harslan/isom-835/master/public/session-05/ISOM835_Competition_Starter.ipynb). [Colab starter entry](https://colab.research.google.com/github/harslan/isom-835/blob/master/public/session-05/ISOM835_Competition_Starter.ipynb) is the provided reference; its browser page was not independently reviewed.

Rendered pages initially blocked (official GitHub syllabus/session sources subsequently reviewed): [competition invitation](https://www.kaggle.com/t/47a351316fb3472f903e14fe1f2387cb), [syllabus](https://isom-835.vercel.app/syllabus/), [Session 3](https://isom-835.vercel.app/sessions/session-03/), [Session 4](https://isom-835.vercel.app/sessions/session-04/), [Session 5](https://isom-835.vercel.app/sessions/session-05/), [Session 7](https://isom-835.vercel.app/sessions/session-07/), [Session 8](https://isom-835.vercel.app/sessions/session-08/), and two Drive notebooks listed in docs/OFFICIAL_REFERENCE_REVIEW.md. The official GitHub syllabus/session sources were subsequently reviewed at commit 112b263771011178f956525fc3d5585ce1901cfe; authenticated Kaggle rules and Drive references remain unreviewed. Additional requirements: suffolk.edu Kaggle account, AI disclosure at the top of every notebook, and at least one random-forest Kaggle submission by Nov 9 4:30 PM. See the source-review record. Domain additions are saved for environment settings review and publication; runtime access has not been established.

Current status: PARTIALLY COMPLETE (academic deliverables). Ten measured experiments and EDA complete; feature timing, authenticated competition-rule verification, final selection, memo review and all external uploads remain pending. See PROJECT_STATUS.md and REQUIREMENTS_AUDIT.md. Individual roles/contributions are not invented.

AI assistance: OpenAI Codex was used for project organization, coding, debugging, analysis support and drafting. Team review is pending. ChatGPT use is not established; disclose it if used. Instructor reference scores are never presented as our measurements.

Course source review additionally confirms: 50% private-leaderboard band + 50% memo; notebook-top AI disclosure (implemented); suffolk.edu registration and a random-forest submission are pending external confirmation. Session 7’s course-file column/leak count differs from actual competition inputs; no external course dataset was loaded.

Current verification: **27 passed, 2 skipped, 0 failed**; three notebooks executed, nine structurally valid. A one-page 11pt actual-results DRAFT PDF and two technically validated provisional candidate CSVs exist; final approval/upload remains pending. Portable code/docs/notebooks archive (without raw data/models/predictions/OOF arrays) is available as ISOM835_Hotel_Cancellation_Project_review.zip in /workspace.

GitHub publication: project source, notebooks, documentation, aggregate measured results and draft academic deliverables are included in the requested folder layout. [Notebook Colab links](notebooks/README.md) and [public file inventory](docs/GITHUB_FILE_MANIFEST.txt) are available. Raw competition CSVs, fitted models, prediction CSVs, OOF arrays, interpreter caches and credentials remain private/local. Publication does not approve a model or confirm academic uploads. Earlier setup records describe the state before publication; see docs/GITHUB_PUBLICATION.md for the publication scope.
