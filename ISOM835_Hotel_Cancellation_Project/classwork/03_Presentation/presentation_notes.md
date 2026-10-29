# Speaker notes — actual-results draft, PENDING rehearsal
0:00–0:30: “We predict cancellation risk per booking to help plan capacity. ROC-AUC measures ranking; reliable expected cancellation counts also require calibration.”
0:30–1:15: “We ran ten experiments using the same five stratified folds, seed 835. The professor baseline measured 0.88936 AUC; correcting categorical IDs reached 0.90740. All preprocessing was fitted inside folds.”
1:15–2:00: “Categorical handling improved the same logistic model by 0.01804 AUC. Features raised it to 0.91597. Exploratory boosting reached 0.94471, but the conservative candidate was 0.88323; timing uncertainty matters more than the highest score.”
2:00–2:30: “Non Refund bookings canceled 99.37% versus 20.34% for Refundable bookings, whose sample was only 118. This is an association requiring investigation. Known outcome columns are absent; remaining state variables need provenance.”
2:30–3:00: “Agent holdout and estimated booking-time validation were lower, 0.8054 and 0.7782 for the conservative candidate. We recommend timing review and calibration before operational use. No leaderboard score or upload is claimed.”

All four members remain credited. Speaking allocation and actual three-minute rehearsal are unconfirmed. Do not present these notes as finalized before team review.
