from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from pypdf import PdfReader
from .config import ROOT,team_name

def render_draft():
    source=ROOT/'classwork/02_Modeling_Memo'/f'{team_name()}_memo.md'
    dest=source.with_name(f'{team_name()}_memo_DRAFT.pdf')
    lines=[]
    for line in source.read_text().splitlines():
        if not line:continue
        lines.extend(simpleSplit(line.replace('## ','').replace('# ',''),'Helvetica',11,468))
    if len(lines)*12>648:raise ValueError(f'Draft has {len(lines)} lines; shorten to fit one page at 11pt')
    c=canvas.Canvas(str(dest),pagesize=(612,792));c.setFont('Helvetica',11);y=720
    for line in lines:c.drawString(72,y,line);y-=12
    c.save();assert len(PdfReader(dest).pages)==1
    return dest
if __name__=='__main__':print(render_draft())
