# DRAFT — Ctrl_+_Alt_+_Defeat modeling memo
Suffolk University · ISOM 835 Fall 2026 · Head of Revenue Management
Anshuman Agrawal · Mark Belov · Phuc Hoang · Sarah Legault

## 1. What You Tried
Ten five-fold experiments (seed 835) compared logistic regression, decision trees, random forests and gradient boosting. Baseline AUC was 0.88936; corrected categorical IDs reached 0.90740 and engineered logistic 0.91597. Exploratory boosting reached 0.94471. Conservative regularized boosting reached 0.88323, with fold SD 0.00342. These are validation results, not Kaggle scores.

## 2. What Worked
The largest controlled same-model feature improvement excluding uncertain-field inclusion was categorical agent/company handling: +0.01804 AUC. Treating IDs as labels avoids imposing numeric order. The largest comparable model-family change on conservative engineered inputs was logistic to random forest: +0.06721; it changes model capacity as well. Neither comparison establishes production benefit without timing review.

## 3. What Leaked or Almost Did
Known outcomes reservation_status, reservation_status_date and assigned_room_type were verified absent. The conservative candidate drops booking_changes, days_in_waiting_list, total_of_special_requests, agent, company, country and deposit_type. Adding them to fixed-config boosting raised AUC 0.07131; that is not proof of leakage. Booking creation is provisional; remaining fields such as ADR/parking still need timestamp evidence. No model is certified leak-free.

## 4. Your Score
Public Kaggle AUC: PENDING confirmed submission. Private Kaggle AUC: PENDING official reveal. Best conservative CV AUC: 0.88323; exploratory CV: 0.94471. No leaderboard comparison is possible. Candidate OOF Brier score is 0.12942; ranking and calibration answer different questions.

## 5. With Two More Weeks
Test booking-time validation and calibrate probabilities using training-only nested folds. Expected AUC improvement: not yet estimated; changes could be negative under time shift. Timing evidence is a prerequisite. Reliable probabilities could support expected cancellation counts, with capacity/cost review before decisions.

AI assistance: ChatGPT and OpenAI coding tools supported coding, analysis and drafting. Team review is pending. DRAFT; not ready for Canvas upload.
