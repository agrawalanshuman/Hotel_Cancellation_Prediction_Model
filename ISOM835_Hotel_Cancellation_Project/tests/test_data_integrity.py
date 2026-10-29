import ast,json
from pathlib import Path
import nbformat,pytest
from src.config import ROOT,SEED,TRAIN_ROWS,TEST_ROWS
from src.data_loader import load_data,validate_data

def test_actual_competition_data():
    try:frames=load_data()
    except FileNotFoundError as e:pytest.skip(str(e))
    result=validate_data(*frames)
    assert (result['train_rows'],result['test_rows'])==(TRAIN_ROWS,TEST_ROWS)

def test_required_structure():
    required=['README.md','PROJECT_OVERVIEW.md','TEAM_COMPLIANCE.md','PROJECT_STATUS.md','REQUIREMENTS_AUDIT.md','CONTRIBUTING.md','requirements.txt','.gitignore','.env.example',
     'classwork/DELIVERABLE_CHECKLIST.md','classwork/04_Submission_Records/kaggle_submission_log.csv',
     'src/config.py','src/data_loader.py','src/preprocessing.py','src/features.py','src/leakage_audit.py','src/train.py','src/evaluate.py','src/compare_models.py','src/predict.py','src/validate_submission.py',
     'docs/PROJECT_WORKFLOW.md','docs/MODELING_DECISIONS.md','docs/BUSINESS_INTERPRETATION.md','docs/GOOGLE_COLAB_GUIDE.md','docs/FINAL_QUALITY_CHECK.md']
    assert all((ROOT/p).is_file() for p in required)

def test_notebooks_valid():
    paths=sorted((ROOT/'notebooks').glob('*.ipynb'));assert len(paths)==9
    for path in paths:nbformat.validate(nbformat.read(path,as_version=4))

def test_syntax_and_imports():
    import importlib
    for p in (ROOT/'src').glob('*.py'):
        ast.parse(p.read_text());importlib.import_module('src.'+p.stem)

def test_seed_and_names():
    assert SEED==835
    text=(ROOT/'README.md').read_text()
    for name in ['Anshuman Agrawal','Mark Belov','Phuc Hoang','Sarah Legault']:assert name in text
    assert 'AI assistance:' in text


def test_measured_primary_results_are_consistent():
    import numpy as np,pandas as pd
    from sklearn.metrics import roc_auc_score
    index=ROOT/'results/reports/run_index.json'
    if not index.exists():pytest.skip('No measured run index yet')
    try:train,_,_=load_data()
    except FileNotFoundError as e:pytest.skip(str(e))
    runs=json.loads(index.read_text())
    if any(not (ROOT/'results/reports'/rid/'oof.npy').exists() for rid in runs.values()):
        pytest.skip('Private OOF arrays are intentionally excluded from GitHub; restore locally for measured-result checks')
    assert len(runs)==10
    folds=pd.read_csv(ROOT/'experiments/cv_fold_scores.csv')
    for rid in runs.values():
        folder=ROOT/'results/reports'/rid
        metrics=json.loads((folder/'metrics.json').read_text())
        scores=folds[folds.experiment_id==rid].auc.to_numpy()
        assert len(scores)==5
        assert np.mean(scores)==pytest.approx(metrics['mean_auc'])
        oof=np.load(folder/'oof.npy')
        assert len(oof)==len(train) and np.isfinite(oof).all()
        assert ((0<=oof)&(oof<=1)).all()
        assert 0<roc_auc_score(train.is_canceled,oof)<1


def test_notebook_top_ai_disclosure():
    for p in (ROOT/'notebooks').glob('*.ipynb'):
        n=json.loads(p.read_text())
        assert n['cells'][0]['cell_type']=='markdown'
        assert 'AI assistance:' in ''.join(n['cells'][0]['source'])
