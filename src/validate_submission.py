import argparse, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import TEST_ROWS, TARGET, ROOT

def validate_submission(sub,test,sample,expected_rows=TEST_ROWS):
    if list(sub.columns)!=['id',TARGET]: raise ValueError('Columns must be exactly id,is_canceled')
    if len(sub)!=expected_rows or len(test)!=expected_rows or len(sample)!=expected_rows: raise ValueError('Incorrect row count')
    for f in (sub,test,sample):
        if f.id.isna().any() or f.id.duplicated().any(): raise ValueError('Missing/duplicate IDs')
    if list(sample.columns)!=['id',TARGET]: raise ValueError('Sample columns invalid')
    if not sub.id.reset_index(drop=True).equals(sample.id.reset_index(drop=True)): raise ValueError('Must follow sample ID order exactly')
    if set(sub.id)!=set(test.id): raise ValueError('Test IDs differ')
    if not pd.api.types.is_numeric_dtype(sub[TARGET]): raise ValueError('Predictions must be numeric')
    p=sub[TARGET].to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('Nonfinite or out-of-range probabilities')
    if np.isin(p,[0,1]).all(): raise ValueError('Thresholded labels detected; continuous probabilities required')
    return True

def validate_file(path,test,sample):
    validate_submission(pd.read_csv(path),test,sample)
    text=f'VALIDATED\nUTC: {datetime.now(timezone.utc).isoformat()}\nFile: {path.name}\nSHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}\nRows: {TEST_ROWS}\nColumns, ID order/coverage/uniqueness, finite continuous probabilities in [0,1]: PASS\nUpload status: PENDING\n'
    (ROOT/'classwork/01_Competition_Submission/submission_validation.txt').write_text(text)
    return text
if __name__=='__main__':
    from .data_loader import load_data
    p=argparse.ArgumentParser();p.add_argument('path',type=__import__('pathlib').Path);a=p.parse_args()
    _,test,sample=load_data();print(validate_file(a.path,test,sample))
