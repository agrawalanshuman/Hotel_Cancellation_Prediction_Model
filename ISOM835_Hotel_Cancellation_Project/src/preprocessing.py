from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from .features import BookingFeatures
from .config import SEED

def build_pipeline(model='logistic', conservative=True, engineered=False, categorical_ids=True, drop_features=(), **params):
    numeric=Pipeline([('imp',SimpleImputer(strategy='median',keep_empty_features=True)),('sc',StandardScaler())])
    encoding = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1) if model=='hist' else OneHotEncoder(handle_unknown='ignore',min_frequency=20)
    categorical=Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='missing',keep_empty_features=True)),('encode',encoding)])
    prep=ColumnTransformer([('num',numeric,make_column_selector(dtype_include='number')),('cat',categorical,make_column_selector(dtype_exclude='number'))])
    estimators={
      'logistic':lambda:LogisticRegression(max_iter=3000,random_state=SEED),
      'tree':lambda:DecisionTreeClassifier(max_depth=8,min_samples_leaf=20,random_state=SEED),
      'forest':lambda:RandomForestClassifier(n_estimators=200,max_depth=16,min_samples_leaf=5,n_jobs=2,random_state=SEED),
      'hist':lambda:HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,l2_regularization=2,early_stopping=True,random_state=SEED)}
    if model not in estimators: raise ValueError(f'Unknown model: {model}')
    clf=estimators[model]().set_params(**params)
    return Pipeline([('features',BookingFeatures(conservative,engineered,categorical_ids,drop_features)),('prep',prep),('clf',clf)])
