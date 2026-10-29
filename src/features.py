import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .config import KNOWN_LEAKS, UNCERTAIN

class BookingFeatures(TransformerMixin, BaseEstimator):
    """Row-local transforms only; no population statistics or target inputs."""
    def __init__(self, conservative=True, engineered=False, categorical_ids=True, drop_features=()):
        self.conservative=conservative
        self.engineered=engineered
        self.categorical_ids=categorical_ids
        self.drop_features=drop_features
    def fit(self, X, y=None): return self
    def transform(self, X):
        X=X.copy().drop(columns=['id','is_canceled',*KNOWN_LEAKS,*self.drop_features],errors='ignore')
        if self.conservative: X=X.drop(columns=list(UNCERTAIN),errors='ignore')
        if self.categorical_ids:
            for c in ('agent','company'):
                if c in X: X[c]=X[c].map(lambda v: str(v) if not __import__('pandas').isna(v) else 'missing')
        if self.engineered:
            if {'stays_in_weekend_nights','stays_in_week_nights'} <= set(X):
                X['total_nights']=X.stays_in_weekend_nights+X.stays_in_week_nights
                X['weekend_share']=X.stays_in_weekend_nights / X.total_nights.replace(0,np.nan)
            if {'adults','children','babies'} <= set(X): X['total_guests']=X.adults+X.children+X.babies
            if 'previous_cancellations' in X: X['previous_cancel_indicator']=(X.previous_cancellations>0).astype(int)
            if 'is_repeated_guest' in X: X['returning_guest_indicator']=X.is_repeated_guest
            if 'lead_time' in X: X['log_lead_time']=np.log1p(X.lead_time.clip(lower=0))
        return X
