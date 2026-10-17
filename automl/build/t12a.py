import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
d=sm.datasets.co2.load_pandas().data["co2"].interpolate().resample("MS").mean()
print(len(d),d.index[0],d.index[-1])
y=d.values; n=len(y)
def make(y,L,diff):
    # target: y[t]-y[t-12] if diff else y[t]; features: lags 1..L of y (and of 12-differences), month
    rows=[];tg=[]
    for t in range(max(L,12),len(y)):
        lags=y[t-L:t][::-1]
        rows.append(np.r_[lags-(y[t-12] if diff else 0),t%12==np.arange(12)])
        tg.append(y[t]-(y[t-12] if diff else 0))
    return np.array(rows),np.array(tg)
H=36; train=y[:-H]; test=y[-H:]
# one-step-ahead walk-forward on last 36 months
def wf(model_fn,L,diff):
    errs=[]
    for k in range(H):
        t=n-H+k; hist=y[:t]
        X,T=make(hist,L,diff); m=model_fn().fit(X,T)
        lags=hist[-L:][::-1]; x=np.r_[lags-(hist[-12] if diff else 0),t%12==np.arange(12)]
        p=m.predict(x[None])[0]+(hist[-12] if diff else 0); errs.append(abs(p-y[t]))
    return np.mean(errs)
print("naive",np.mean(np.abs(y[-H:]-y[-H-1:-1])),"sazonal",np.mean(np.abs(y[-H:]-y[-H-12:-12])))
for nm,fn in (("ridge",lambda:Ridge(1.0)),("hgb",lambda:HistGradientBoostingRegressor(max_iter=100,random_state=0))):
    for diff in (False,True): print(nm,diff,wf(fn,12,diff))
