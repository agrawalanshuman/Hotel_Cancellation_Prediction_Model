import hashlib
import pandas as pd
from .config import DATA_DIR, TARGET, TRAIN_ROWS, TEST_ROWS, KNOWN_LEAKS

def validate_data(train, test, sample, expected_rows=(TRAIN_ROWS, TEST_ROWS)):
    if (len(train), len(test)) != expected_rows:
        raise ValueError(f'Expected train/test rows {expected_rows}, got {(len(train), len(test))}')
    for frame, name in [(train,'train'), (test,'test'), (sample,'sample')]:
        if 'id' not in frame or frame.id.isna().any() or frame.id.duplicated().any():
            raise ValueError(f'{name}: missing or duplicate IDs')
        if frame.columns.duplicated().any(): raise ValueError('Duplicate columns')
    if TARGET not in train or TARGET in test: raise ValueError('Target schema invalid')
    if train[TARGET].isna().any() or set(train[TARGET].unique()) != {0,1}: raise ValueError('Target must contain both binary classes')
    if set(train.columns) - {TARGET} != set(test.columns): raise ValueError('Feature schema mismatch')
    if list(sample.columns) != ['id', TARGET] or len(sample) != len(test): raise ValueError('Sample schema/length invalid')
    if set(sample.id) != set(test.id): raise ValueError('Sample/test IDs differ')
    if set(train.id) & set(test.id): raise ValueError('Train/test IDs overlap')
    bad = set(KNOWN_LEAKS) & (set(train) | set(test))
    if bad: raise ValueError(f'Known outcome leakage present: {sorted(bad)}')
    return {'train_rows': len(train), 'test_rows': len(test), 'cancellation_rate': float(train[TARGET].mean())}

def load_data(directory=DATA_DIR):
    paths = [directory / n for n in ('train.csv','test.csv','sample_submission.csv')]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError('PENDING authorized competition files: ' + ', '.join(missing))
    frames = tuple(pd.read_csv(p) for p in paths)
    validate_data(*frames)
    return frames

def data_hashes(directory=DATA_DIR):
    return {n: hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ('train.csv','test.csv','sample_submission.csv')}
