# Source review — October 29, 2026
Read: https://raw.githubusercontent.com/harslan/isom-835/master/public/session-05/ISOM835_Competition_Starter.ipynb (HTTP 200; saved as notebook 00).
Verified there: stratified 5-fold seed 835; min_frequency=20; logistic max_iter=3000; numeric agent/company baseline; suggested categorical fix; HistGradientBoosting regularization/early stopping; grouped-by-agent robustness; five submissions/day; no public-leaderboard tuning; memo one page due Nov 9 11:59 PM; leaderboard Nov 9 4:30 PM. Reference-only values ~0.889 baseline CV, ~0.907 categorical CV, ~0.96 boosting are NOT our results. User's ~0.896 leaderboard baseline is a different statistic, not a measured local score.

Blocked by proxy 403: https://isom-835.vercel.app/syllabus/ and sessions/session-03/, session-04/, session-05/, session-07/, session-08/; https://www.kaggle.com/t/47a351316fb3472f903e14fe1f2387cb ; https://colab.research.google.com/drive/1-NT1byochvglHheil_xx9E-t0Iw23VkH ; https://colab.research.google.com/drive/1G5EtuWC7pPFk6g1QR_8kC2fH0SnJtA3o . Their contents were NOT reviewed. Domain additions were saved in environment draft but runtime access/publication has not been established. Browser-authenticated Drive references may still require access after network changes.

Requirements not verifiable in accessible starter (including team size and exact submission dimensions/deadlines) are attributed to the user-provided assignment pending official-page verification. No conflict is silently resolved. Retain the original starter implementation and use seed 835 consistently.

## Official GitHub source review (completed October 29, 2026)
Although the website remains proxy-blocked, equivalent official course source was retrieved from https://github.com/harslan/isom-835 at commit 112b263771011178f956525fc3d5585ce1901cfe. Reviewed src/data/syllabus.ts and src/content/sessions/session-03.json, session-04.json, session-05.json, session-07.json, session-08.json. These are source files, not a claim that the rendered website or authenticated Kaggle rules were accessed.

Additional requirements discovered:
- Syllabus AI policy: disclose tools in a short note at the TOP of each notebook; students must explain and defend every submitted line. Added to all nine covers; original starter attribution preserved.
- Session 5 homework: create Kaggle account using suffolk.edu email; form/register the team by Nov 2 and make at least one submission. Registration and account email are unverified. User-confirmed instructor exception covers four members despite the published 2–3 policy.
- Session 7 homework: at least one RANDOM-FOREST submission from the team before Nov 9, 4:30 PM; maintain dropped-feature reasons in memo draft. A forest experiment is measured locally; actual forest submission is PENDING.
- Syllabus midterm rubric: 50% private-leaderboard band, 50% one-page modeling memo, with a floor for a leak-free submission beating logistic baseline. Conservative AUC is below the professor-equivalent baseline; do not claim that grading floor was met. It depends on actual private score and leakage review.
- Sessions 7/8 explicitly teach random forests, HistGradientBoosting, LightGBM and XGBoost; the implemented model families are course-supported. Optional external libraries were not installed/run. Competition-specific rules still need authenticated verification.
- Session 5 confirms Nov 2 first submission; syllabus confirms identical Nov 9 4:30 PM Canvas/Kaggle file, 3-minute evening talks and 11:59 PM memo. Session 8 explicitly requests private AUC after reveal; keep pending until then.

Source discrepancy: Session 7 describes 32 columns/two leaks in a course practice file, while competition starter states three removed columns. Actual uploaded training has 30 columns, test 29 including id; all three known outcomes are absent. Use only uploaded competition data, never fetch the course practice hotel CSV. Record this prose mismatch for instructor clarification; do not alter competition files.

Still unreviewed: authenticated competition invitation/rules and two Drive notebooks. Their proxy 403 responses remain a concrete access limitation. Saved domain additions require settings review/save and publication, followed by retry. Public GitHub review does not establish the private competition rules.
