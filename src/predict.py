import argparse,json,hashlib
from pathlib import Path
import joblib,pandas as pd
from .config import ROOT,TARGET,team_name
from .data_loader import load_data,data_hashes
from .preprocessing import build_pipeline
from .validate_submission import validate_submission,validate_file

def generate(selection):
    name=team_name(); spec=json.loads(Path(selection).read_text())
    if not spec.get('timing_review_approved') or not spec.get('course_methods_approved') or not spec.get('team_eligibility_resolved'):
        raise ValueError('Selection requires explicit timing, course-method and team eligibility reviews')
    if not spec.get('reviewer') or not spec.get('evidence'): raise ValueError('Reviewer/evidence required')
    metrics=json.loads((ROOT/'results/reports'/spec['experiment_id']/'metrics.json').read_text())
    if metrics['data_hashes']!=data_hashes(): raise ValueError('Data changed since validation')
    if metrics['mean_auc']>0.985 and not spec.get('high_auc_review_evidence'): raise ValueError('High AUC leakage review required')
    train,test,sample=load_data()
    model=build_pipeline(metrics['model'],metrics['conservative'],metrics['engineered'],metrics['categorical_ids'],metrics['drop_features'],**metrics['params'])
    model.fit(train.drop(columns=TARGET),train[TARGET])
    p=pd.Series(model.predict_proba(test)[:,1],index=test.id)
    sub=sample[['id']].copy();sub[TARGET]=sub.id.map(p)
    validate_submission(sub,test,sample)
    dest=ROOT/'classwork/01_Competition_Submission'/f'{name}_submission.csv'
    if dest.exists(): raise FileExistsError('Preserve existing deliverable; archive deliberately before replacing')
    sub.to_csv(dest,index=False);print(validate_file(dest,test,sample))
    joblib.dump(model,ROOT/'results/models'/f'{spec["experiment_id"]}.joblib')
    (dest.with_suffix('.provenance.json')).write_text(json.dumps({'selection':spec,'metrics':metrics,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
    return dest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);a=p.parse_args();generate(a.selection)
