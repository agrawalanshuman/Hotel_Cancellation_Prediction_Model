import argparse, csv, json, time, uuid, traceback
from datetime import datetime, timezone
import numpy as np
from .config import ROOT, SEED, TARGET
from .data_loader import load_data, data_hashes
from .preprocessing import build_pipeline
from .evaluate import evaluate

def append_row(path, row):
    with path.open(newline='') as f: fields=next(csv.reader(f))
    with path.open('a',newline='') as f: csv.DictWriter(f,fieldnames=fields).writerow({k:row.get(k,'') for k in fields})

def run(model='logistic', conservative=True, engineered=False, categorical_ids=True, description='',drop_features=(),params=None):
    rid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'results/reports'/rid;out.mkdir(parents=True)
    row=dict(experiment_id=rid,run_date=datetime.now(timezone.utc).isoformat(),model_name=model,description=description,
             random_seed=SEED,cv_strategy='StratifiedKFold(5,shuffle=True,random_state=835)',decision='FAILED',notes='')
    started=time.monotonic()
    try:
        train,_,_=load_data(); X=train.drop(columns=TARGET);y=train[TARGET]
        pipeline=build_pipeline(model,conservative,engineered,categorical_ids,drop_features,**(params or {}))
        row.update(features_used=json.dumps(list(pipeline['features'].transform(X).columns)),features_dropped=json.dumps(sorted(set(X)-set(pipeline['features'].transform(X)))),
                   preprocessing=str(pipeline['prep']),hyperparameters=json.dumps(pipeline['clf'].get_params(),default=str))
        result=evaluate(pipeline,X,y)
        review='REQUIRED_HIGH_AUC' if result['mean_auc']>0.985 else 'PENDING_TIMING_REVIEW'
        row.update(mean_cv_auc=result['mean_auc'],std_cv_auc=result['std_auc'],leakage_review_status=review,decision='MEASURED_NOT_APPROVED')
        for i,s in enumerate(result['fold_auc'],1):
            row[f'fold_{i}_auc']=s
            append_row(ROOT/'experiments/cv_fold_scores.csv',dict(experiment_id=rid,fold=i,auc=s))
        np.save(out/'oof.npy',result['oof'])
        metadata={k:v for k,v in result.items() if k not in ('oof','fitted')}
        metadata.update(experiment_id=rid,model=model,conservative=conservative,engineered=engineered,categorical_ids=categorical_ids,
          drop_features=list(drop_features),params=params or {},data_hashes=data_hashes(),features=list(pipeline['features'].transform(X)),
          seed=SEED,duration_seconds=time.monotonic()-started,leakage_review_status=review,approved=False)
        (out/'metrics.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata,indent=2)); return rid
    except Exception as e:
        row['notes']=f'{type(e).__name__}: {e}';(out/'error.txt').write_text(traceback.format_exc());raise
    finally:
        row['training_duration_seconds']=time.monotonic()-started
        append_row(ROOT/'experiments/experiment_log.csv',row)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['logistic','tree','forest','hist'],default='logistic')
    p.add_argument('--exploratory',action='store_true');p.add_argument('--engineered',action='store_true');p.add_argument('--numeric-ids',action='store_true');p.add_argument('--description',default='');p.add_argument('--drop',nargs='*',default=[]);p.add_argument('--params',default='{}')
    a=p.parse_args();run(a.model,not a.exploratory,a.engineered,not a.numeric_ids,a.description,a.drop,json.loads(a.params))
