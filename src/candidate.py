"""Generate genuine predictions marked provisional; never official deliverable approval."""
import json,hashlib
from datetime import datetime,timezone
import pandas as pd,joblib
from .config import ROOT,TARGET
from .data_loader import load_data,data_hashes
from .preprocessing import build_pipeline
from .validate_submission import validate_submission

def generate_candidate(experiment_id):
    m=json.loads((ROOT/'results/reports'/experiment_id/'metrics.json').read_text())
    if m['data_hashes']!=data_hashes():raise ValueError('Data changed')
    train,test,sample=load_data()
    model=build_pipeline(m['model'],m['conservative'],m['engineered'],m['categorical_ids'],m['drop_features'],**m['params'])
    model.fit(train.drop(columns=TARGET),train[TARGET])
    probability=pd.Series(model.predict_proba(test)[:,1],index=test.id)
    sub=sample[['id']].copy();sub[TARGET]=sub.id.map(probability)
    validate_submission(sub,test,sample)
    path=ROOT/'results/predictions'/f'{experiment_id}_PROVISIONAL.csv'
    if path.exists():raise FileExistsError(path)
    sub.to_csv(path,index=False)
    validate_submission(pd.read_csv(path),test,sample)
    report={'status':'GENERATED_AND_TECHNICALLY_VALIDATED_NOT_APPROVED','experiment_id':experiment_id,'rows':len(sub),'columns':list(sub),
     'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'timestamp_utc':datetime.now(timezone.utc).isoformat(),
     'probability_min':float(sub[TARGET].min()),'probability_max':float(sub[TARGET].max()),'data_hashes':m['data_hashes'],
     'checks':'Exact row count; sample ID order; test ID coverage; unique IDs; finite continuous probabilities within [0,1]; read-back CSV validation',
     'limitations':'Timing and final model approval pending. No Kaggle/Canvas upload.'}
    (ROOT/'results/reports'/experiment_id/'candidate_validation.json').write_text(json.dumps(report,indent=2))
    summary=ROOT/'results/reports/candidate_validation.json'
    if not summary.exists(): summary.write_text(json.dumps(report,indent=2))
    joblib.dump(model,ROOT/'results/models'/f'{experiment_id}_PROVISIONAL.joblib')
    print(json.dumps(report,indent=2));return path
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('experiment_id');generate_candidate(p.parse_args().experiment_id)
