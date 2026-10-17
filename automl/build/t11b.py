import warnings; warnings.filterwarnings("ignore")
import numpy as np, time, json, itertools
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
X,y=load_digits(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=0)
Xa,Xv,ya,yv=train_test_split(Xtr,ytr,test_size=.3,stratify=ytr,random_state=0)
sc=StandardScaler().fit(Xa); Xa,Xv,Xt=sc.transform(Xa),sc.transform(Xv),sc.transform(Xte)
NEU=[8,16,32,64,128]; ACT=["relu","tanh"]
ARCHS=[(h,a) for L in (1,2,3) for h in itertools.product(NEU,repeat=L) for a in ACT]
def nparams(a):
    d=[64]+list(a[0])+[10]; return sum(i*o+o for i,o in zip(d[:-1],d[1:]))
def fit(a,it): return MLPClassifier(a[0],activation=a[1],max_iter=it,random_state=0).fit(Xa,ya)
t=time.time(); T=[]
for a in ARCHS:
    m10=fit(a,10); m40=fit(a,40)
    T.append(dict(arch=[list(a[0]),a[1]],params=nparams(a),v10=m10.score(Xv,yv),v40=m40.score(Xv,yv),t40=m40.score(Xt,yte)))
print(len(ARCHS),time.time()-t)
json.dump(T,open("/tmp/nas_table.json","w"))
v=np.array([r["v40"] for r in T]); print(v.max(),np.median(v),v.min())
i=v.argmax(); print(T[i])
from scipy.stats import spearmanr
print(spearmanr([r["v10"] for r in T],v))
