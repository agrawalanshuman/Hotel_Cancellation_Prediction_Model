import pandas as pd
from src.features import BookingFeatures
from src.config import KNOWN_LEAKS,UNCERTAIN
from src.data_loader import load_data,validate_data
import pytest

def test_known_and_uncertain_fields_removed_conservatively():
    X=pd.DataFrame({c:[1] for c in ['id','is_canceled','lead_time',*KNOWN_LEAKS,*UNCERTAIN]})
    t=BookingFeatures().transform(X)
    assert set(t)=={'lead_time'}

def test_actual_outcome_columns_absent():
    try:train,test,_=load_data()
    except FileNotFoundError as e:pytest.skip(str(e))
    assert not set(KNOWN_LEAKS)&(set(train)|set(test))

def test_loader_rejects_inserted_leak():
    try:train,test,sample=load_data()
    except FileNotFoundError as e:pytest.skip(str(e))
    train=train.assign(reservation_status='Canceled');test=test.assign(reservation_status='Canceled')
    with pytest.raises(ValueError,match='Known outcome leakage'):validate_data(train,test,sample)
