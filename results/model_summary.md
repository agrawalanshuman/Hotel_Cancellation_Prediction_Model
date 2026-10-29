# Hotel cancellation results
Team: Ctrl_+_Alt_+_Defeat
Data: 83,573 labeled bookings and 35,817 test bookings.
Five-fold stratified validation; seed 835; all learned preprocessing inside folds.

## Measured experiments
| Experiment | Mean ROC-AUC | Fold SD |
|---|---:|---:|
| Exploratory gradient boosting | 0.94471 | 0.00106 |
| Logistic with engineered features | 0.91597 | 0.00165 |
| Logistic with categorical IDs | 0.90740 | 0.00168 |
| Logistic baseline | 0.88936 | 0.00187 |
| Regularized conservative boosting | 0.88323 | 0.00342 |
| Random forest | 0.88252 | 0.00377 |
| Conservative gradient boosting | 0.87340 | 0.00439 |
| Decision tree | 0.81937 | 0.00365 |
| Conservative engineered logistic | 0.81530 | 0.00343 |
| Conservative logistic | 0.80280 | 0.00401 |

## Interpretation
Highest conservative candidate: Regularized conservative boosting. Timing approval remains pending.
Exploratory performance includes uncertain fields and is not certified leak-free.
Held-out-agent AUC: 0.80541; estimated-booking-time AUC: 0.77825.
Secondary robustness drops limit deployment claims. These are different validation scenarios.
Public Kaggle AUC: pending. Private Kaggle AUC: pending. No upload confirmed.
AI assistance: ChatGPT and OpenAI coding tools supported analysis and drafting; team review pending.
