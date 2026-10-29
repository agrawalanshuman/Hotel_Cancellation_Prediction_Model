import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from .config import SEED

def evaluate(model, X, y, splitter=None, groups=None):
    splitter=splitter or StratifiedKFold(5,shuffle=True,random_state=SEED)
    oof=np.full(len(y),np.nan); scores=[]; fitted=[]
    for fold,(a,b) in enumerate(splitter.split(X,y,groups),1):
        if len(set(y.iloc[b]))!=2: raise ValueError('A fold has only one target class')
        m=clone(model).fit(X.iloc[a],y.iloc[a]);p=m.predict_proba(X.iloc[b])[:,1]
        oof[b]=p;scores.append(float(roc_auc_score(y.iloc[b],p)));fitted.append(m)
    mask=np.isfinite(oof)
    return {'fold_auc':scores,'mean_auc':float(np.mean(scores)),'std_auc':float(np.std(scores)),
            'brier':float(brier_score_loss(y[mask],oof[mask])),'log_loss':float(log_loss(y[mask],oof[mask])),
            'oof':oof,'fitted':fitted}
