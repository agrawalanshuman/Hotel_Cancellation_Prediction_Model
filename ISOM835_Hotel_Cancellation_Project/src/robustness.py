import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold,TimeSeriesSplit
from sklearn.calibration import calibration_curve
from .config import ROOT,TARGET
from .data_loader import load_data
from .preprocessing import build_pipeline
from .evaluate import evaluate

def check(experiment_id):
    train,_,_=load_data();folder=ROOT/'results/reports'/experiment_id
    m=json.loads((folder/'metrics.json').read_text());p=np.load(folder/'oof.npy')
    frac,mean=calibration_curve(train[TARGET],p,n_bins=10,strategy='quantile')
    fig,ax=plt.subplots();ax.plot(mean,frac,'o-',label='OOF');ax.plot([0,1],[0,1],'--',label='Ideal');ax.set(title='Out-of-fold probability calibration',xlabel='Mean predicted probability',ylabel='Observed cancellation fraction');ax.legend();fig.savefig(ROOT/'results/figures/calibration.png');plt.close(fig)
    model=build_pipeline(m['model'],m['conservative'],m['engineered'],m['categorical_ids'],m['drop_features'],**m['params'])
    X=train.drop(columns=TARGET);y=train[TARGET]
    groups=train.agent.fillna(-1).astype(str)
    g=evaluate(model,X,y,GroupKFold(5),groups)
    # Estimated booking date uses only competition arrival date and lead time; dates can be revised.
    arrival=pd.to_datetime(train.arrival_date_year.astype(str)+'-'+train.arrival_date_month+'-'+train.arrival_date_day_of_month.astype(str),format='%Y-%B-%d')
    booking=arrival-pd.to_timedelta(train.lead_time,unit='D');order=np.argsort(booking.to_numpy(),kind='stable')
    dates=booking.iloc[order].reset_index(drop=True)
    # Do not split one calendar date across train/validation.
    unique=np.sort(dates.unique());splits=[]
    for a,b in TimeSeriesSplit(5).split(unique):
        splits.append((np.flatnonzero(dates.isin(unique[a])),np.flatnonzero(dates.isin(unique[b]))))
    class Fixed:
        def split(self,*args): return iter(splits)
    t=evaluate(model,X.iloc[order].reset_index(drop=True),y.iloc[order].reset_index(drop=True),Fixed())
    report={'experiment_id':experiment_id,'group_agent':{k:v for k,v in g.items() if k not in ('oof','fitted')},'estimated_booking_time':{k:v for k,v in t.items() if k not in ('oof','fitted')},'time_coverage':float(np.isfinite(t['oof']).mean()),'caveats':'Agent holdout includes a large missing-agent group; tests new intermediaries rather than ordinary bookings. Estimated booking dates may reflect revised arrival/lead-time fields. Temporal CV leaves earliest rows unscored. All results exploratory; not primary stratified benchmark.'}
    (ROOT/'results/reports/robustness.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('experiment_id');print(check(p.parse_args().experiment_id))
