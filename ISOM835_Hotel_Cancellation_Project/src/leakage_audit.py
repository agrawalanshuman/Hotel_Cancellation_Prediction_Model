from pathlib import Path
import pandas as pd
from .config import ROOT, KNOWN_LEAKS, UNCERTAIN

def audit_features(train):
    rows=[]
    for f in train.columns:
        if f=='is_canceled': continue
        if f in KNOWN_LEAKS: risk,decision,availability='high','drop','after outcome'
        elif f=='id': risk,decision,availability='high','drop','identifier; no business meaning established'
        elif f in UNCERTAIN: risk,decision,availability='uncertain','investigate','unknown: need timestamped provenance'
        else: risk,decision,availability='unverified','investigate','plausible at creation; requires course/team review'
        rows.append(dict(feature_name=f,feature_description=f.replace('_',' '),availability_at_prediction_time=availability,
           evidence='Column inspected; timing not proven by CSV',risk_level=risk,decision=decision,
           reason='Booking creation is provisional prediction moment',cv_auc_before_removal='',cv_auc_after_removal='',date_reviewed=''))
    return pd.DataFrame(rows)

def save_audit(train):
    p=ROOT/'results/reports/observed_feature_audit.csv';audit_features(train).to_csv(p,index=False);return p
if __name__=='__main__':
    from .data_loader import load_data
    print(save_audit(load_data()[0]))
