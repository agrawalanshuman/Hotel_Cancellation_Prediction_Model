import numpy as np,pandas as pd,pytest
from pypdf import PdfReader
from src.validate_submission import validate_submission
from src.config import ROOT,TEAM_NAME
from src.export_memo import SECTIONS
from src.data_loader import load_data

# Small validator fixtures only; never exported as predictions.
def fixture():
    test=pd.DataFrame({'id':[3,1,2]});sample=pd.DataFrame({'id':[1,2,3],'is_canceled':[0,0,0]})
    sub=pd.DataFrame({'id':[1,2,3],'is_canceled':[0.1,0.3,0.9]});return sub,test,sample

def test_valid_probability_contract():
    assert validate_submission(*fixture(),expected_rows=3)

@pytest.mark.parametrize('kind',['nan','inf','bounds','labels','duplicates','order','columns','length','ids'])
def test_invalid_submissions_rejected(kind):
    sub,test,sample=fixture()
    if kind=='nan':sub.loc[0,'is_canceled']=np.nan
    if kind=='inf':sub.loc[0,'is_canceled']=np.inf
    if kind=='bounds':sub.loc[0,'is_canceled']=1.1
    if kind=='labels':sub['is_canceled']=[0,1,1]
    if kind=='duplicates':sub.loc[0,'id']=2
    if kind=='order':sub=sub.iloc[::-1]
    if kind=='columns':sub=sub[['is_canceled','id']]
    if kind=='length':sub=sub.iloc[:2]
    if kind=='ids':sub.loc[0,'id']=9
    with pytest.raises(ValueError):validate_submission(sub,test,sample,expected_rows=3)

def test_actual_final_csv_if_generated():
    path=ROOT/'classwork/01_Competition_Submission'/f'{TEAM_NAME}_submission.csv'
    if not path.exists():pytest.skip('PENDING reviewed selection/export; no final CSV')
    _,test,sample=load_data();assert validate_submission(pd.read_csv(path),test,sample)

def test_actual_memo_pdf_if_generated():
    path=ROOT/'classwork/02_Modeling_Memo'/f'{TEAM_NAME}_memo.pdf'
    if not path.exists():pytest.skip('PENDING finalized reviewed memo; no PDF')
    pdf=PdfReader(path);assert len(pdf.pages)==1
    text=pdf.pages[0].extract_text()
    assert all(s in text for s in SECTIONS);assert 'AI assistance:' in text

def test_memo_source_sections_disclosure():
    text=(ROOT/'classwork/02_Modeling_Memo'/f'{TEAM_NAME}_memo.md').read_text()
    import re
    assert re.findall(r'^## (.+)$',text,re.M)==SECTIONS
    assert 'AI assistance:' in text


def test_generated_candidate_actual_csv():
    import json,hashlib
    report_path=ROOT/'results/reports/candidate_validation.json'
    if not report_path.exists():pytest.skip('No provisional candidate generated')
    report=json.loads(report_path.read_text())
    path=ROOT/'results/predictions'/f"{report['experiment_id']}_PROVISIONAL.csv"
    if not path.exists():pytest.skip('Private provisional CSV intentionally excluded from GitHub')
    _,test,sample=load_data()
    assert validate_submission(pd.read_csv(path),test,sample)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==report['sha256']

def test_generated_draft_memo_page_and_font():
    path=ROOT/'classwork/02_Modeling_Memo'/f'{TEAM_NAME}_memo_DRAFT.pdf'
    assert path.is_file()
    pdf=PdfReader(path);assert len(pdf.pages)==1
    text=pdf.pages[0].extract_text();assert 'DRAFT' in text
    assert all(s in text for s in SECTIONS) and 'AI assistance:' in text
    assert b'11 Tf' in pdf.pages[0].get_contents().get_data()
