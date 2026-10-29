import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import ROOT,TARGET
from .data_loader import load_data,validate_data,data_hashes

def explore():
    train,test,sample=load_data();report=ROOT/'results/reports';figs=ROOT/'results/figures'
    summary=validate_data(train,test,sample);summary['hashes']=data_hashes()
    (report/'data_integrity.json').write_text(json.dumps(summary,indent=2))
    train.dtypes.astype(str).rename('dtype').to_csv(report/'feature_types.csv')
    train.isna().mean().rename('missing_share').to_csv(report/'missingness.csv')
    train.describe(include='all').to_csv(report/'descriptive_statistics.csv')
    for col in train.select_dtypes(include='number'):
        if col in ('id',TARGET):continue
        fig,ax=plt.subplots();sns.histplot(train[col].dropna(),bins=40,ax=ax);ax.set(title=f'Distribution of {col}',xlabel=col.replace('_',' '),ylabel='Bookings');fig.tight_layout();fig.savefig(figs/f'distribution_{col}.png');plt.close(fig)
    puzzles={}
    for col in ['hotel','deposit_type','market_segment','distribution_channel','agent','company','country','total_of_special_requests','previous_cancellations','booking_changes','arrival_date_month','lead_time']:
        table=train.groupby(col,dropna=False)[TARGET].agg(['size','mean']).rename(columns={'size':'bookings','mean':'cancellation_rate'})
        table.to_csv(report/f'by_{col}.csv')
        show=table.sort_values('bookings',ascending=False).head(20)
        fig,ax=plt.subplots(figsize=(9,4));ax.bar(show.index.astype(str),show.cancellation_rate);ax.set(title=f'Observed cancellation rate by {col} (top 20 by count)',xlabel=col.replace('_',' '),ylabel='Cancellation fraction');ax.tick_params(axis='x',rotation=60);fig.tight_layout();fig.savefig(figs/f'rate_{col}.png');plt.close(fig)
        if col in ('deposit_type','total_of_special_requests'): puzzles[col]=table.reset_index().to_dict(orient='records')
    groups=pd.Series('elsewhere',index=train.index);groups[train.country=='PRT']='Portugal';groups[train.country.isna()]='missing'
    puzzles['country']=train.groupby(groups)[TARGET].agg(['size','mean']).reset_index().to_dict(orient='records')
    (report/'puzzles.json').write_text(json.dumps(puzzles,indent=2))
    (report/'EDA_INTERPRETATION.md').write_text('Plots show observed associations, not causal effects. High-cardinality charts show only top-count categories; full denominators are in CSVs. Numeric ID distributions are diagnostic, not quantities. Missing country is separated from elsewhere. No Deposit must not automatically be called refundable. Arrival month shows arrival seasonality, not necessarily booking seasonality. All booking-state fields need timing review. Puzzle numbers in the assignment are hypotheses; consult puzzles.json for measured values.')
    return summary,puzzles
if __name__=='__main__':print(explore())
