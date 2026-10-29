import argparse,re
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from pypdf import PdfReader
from .config import ROOT,team_name
SECTIONS=['1. What You Tried','2. What Worked','3. What Leaked or Almost Did','4. Your Score','5. With Two More Weeks']

def export(source,reviewed=False):
    text=Path(source).read_text()
    if not reviewed or 'PENDING' in text: raise ValueError('Final export requires reviewed source with no unresolved placeholders')
    found=re.findall(r'^## (.+)$',text,re.M)
    if found!=SECTIONS: raise ValueError('Exactly five ordered memo sections required')
    if 'AI assistance:' not in text: raise ValueError('AI disclosure required')
    path=ROOT/'classwork/02_Modeling_Memo'/f'{team_name()}_memo.pdf'
    if path.exists(): raise FileExistsError(path)
    lines=[]
    for line in text.splitlines():lines.extend(simpleSplit(line.replace('## ','').replace('# ',''),'Helvetica',11,468) if line else [''])
    if len(lines)*13>648: raise ValueError('Text does not fit one page at 11pt and normal margins; edit source')
    c=canvas.Canvas(str(path),pagesize=(612,792));c.setFont('Helvetica',11);y=720
    for line in lines:c.drawString(72,y,line);y-=13
    c.save()
    if len(PdfReader(path).pages)!=1: raise ValueError('Expected exactly one page')
    return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--reviewed',action='store_true');a=p.parse_args();print(export(a.source,a.reviewed))
