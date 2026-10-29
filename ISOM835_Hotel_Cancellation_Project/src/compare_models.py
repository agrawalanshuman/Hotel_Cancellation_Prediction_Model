import json
import pandas as pd
import matplotlib.pyplot as plt
from .config import ROOT

def compare():
    log=pd.read_csv(ROOT/'experiments/experiment_log.csv')
    log=log[log.decision=='MEASURED_NOT_APPROVED']
    cols=['experiment_id','model_name','mean_cv_auc','std_cv_auc','leakage_review_status']
    out=log[cols].sort_values('mean_cv_auc',ascending=False)
    out.to_csv(ROOT/'experiments/model_comparison.csv',index=False)
    if len(out):
        fig,ax=plt.subplots(figsize=(9,5));ax.errorbar(out.experiment_id,out.mean_cv_auc,yerr=out.std_cv_auc,fmt='o');ax.set(title='Measured five-fold ROC-AUC (timing approval separate)',xlabel='Experiment',ylabel='ROC-AUC');ax.tick_params(axis='x',rotation=60);fig.tight_layout();fig.savefig(ROOT/'results/figures/model_comparison.png');plt.close(fig)
    return out
if __name__=='__main__': print(compare().to_string(index=False))
