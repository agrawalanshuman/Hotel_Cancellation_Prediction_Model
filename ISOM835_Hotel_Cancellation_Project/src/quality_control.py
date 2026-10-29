import json,hashlib,sys
from pathlib import Path
from pypdf import PdfReader
from .config import ROOT,TEAM_NAME
from .data_loader import load_data
from .validate_submission import validate_submission
import pandas as pd

def check():
    rows=[]
    def add(name,status,evidence):rows.append(dict(check=name,status=status,evidence=evidence))
    try:train,test,sample=load_data();add('competition_data','PASS','row/schema/target/IDs/known-leak checks passed')
    except Exception as e:add('competition_data','BLOCKED',str(e));test=sample=None
    add('team_eligibility','PASS','User confirmed instructor approval for four on October 29, 2026; document not independently reviewed')
    add('team_name','PASS' if TEAM_NAME else 'PENDING',TEAM_NAME)
    sub=ROOT/'classwork/01_Competition_Submission'/f'{TEAM_NAME}_submission.csv'
    if not sub.exists():add('final_csv','PENDING','Not exported; reviewed model selection required')
    elif test is not None:
        try:validate_submission(pd.read_csv(sub),test,sample);add('final_csv','PASS',hashlib.sha256(sub.read_bytes()).hexdigest())
        except Exception as e:add('final_csv','FAIL',str(e))
    pdf=ROOT/'classwork/02_Modeling_Memo'/f'{TEAM_NAME}_memo.pdf'
    if not pdf.exists():add('memo_pdf','PENDING','Final reviewed PDF absent')
    else:add('memo_pdf','PASS' if len(PdfReader(pdf).pages)==1 else 'FAIL','Page count only; content review separate')
    add('timing_approval','PENDING','Prediction moment and source timestamps require review')
    add('official_reference_review','BLOCKED','Non-starter course/Kaggle/Drive pages blocked by proxy')
    add('kaggle_canvas_uploads','PENDING','No externally confirmed uploads')
    dest=ROOT/'results/reports/quality_control.json';dest.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
    return rows
if __name__=='__main__':sys.exit(0 if all(r['status']=='PASS' for r in check()) else 1)
