import numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from src.preprocessing import build_pipeline
from src.features import BookingFeatures
from src.config import SEED

# Deliberately small artificial unit fixtures; never training data or reported experiments.
def fixture():
    return pd.DataFrame({'lead_time':[0.,2.,np.nan,6.,100.,102.,np.nan,106.],
      'hotel':['A',None,'A','B','B','A','B','A'],'agent':[1.,2.,np.nan,1.,2.,3.,1.,2.]}).replace({None:np.nan}),pd.Series([0,1,0,1,0,1,0,1])

def test_missing_values_and_unknown_category():
    X,y=fixture();m=build_pipeline(conservative=False).fit(X,y)
    t=X.copy();t['hotel']='UNSEEN';p=m.predict_proba(t)[:,1]
    assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()

def test_statistics_learned_only_on_training_folds():
    X,y=fixture();cv=StratifiedKFold(2,shuffle=True,random_state=SEED)
    original=X.copy(deep=True)
    for a,b in cv.split(X,y):
        m=build_pipeline(conservative=False).fit(X.iloc[a],y.iloc[a])
        learned=m['prep'].named_transformers_['num']['imp'].statistics_[0]
        assert learned==np.nanmedian(X.iloc[a].lead_time)
        before=m['prep'].named_transformers_['num']['imp'].statistics_.copy()
        m.predict_proba(X.iloc[b].assign(lead_time=1e9))
        np.testing.assert_array_equal(before,m['prep'].named_transformers_['num']['imp'].statistics_)
    pd.testing.assert_frame_equal(X,original)

def test_categorical_ids_are_not_quantities():
    X,_=fixture();t=BookingFeatures(conservative=False).transform(X)
    assert t.agent.dtype==object and t.agent.isna().sum()==0

def test_engineering_is_row_local_and_zero_safe():
    X=pd.DataFrame({'stays_in_weekend_nights':[0,2],'stays_in_week_nights':[0,3],'adults':[2,1],'children':[0,1],'babies':[0,0],'lead_time':[0,9]})
    all_=BookingFeatures(engineered=True).transform(X)
    one=BookingFeatures(engineered=True).transform(X.iloc[[1]])
    pd.testing.assert_frame_equal(all_.iloc[[1]],one)
    assert np.isnan(all_.weekend_share.iloc[0]);assert all_.total_nights.iloc[1]==5
