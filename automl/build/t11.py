import warnings; warnings.filterwarnings("ignore")
import numpy as np, time, json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
X,y=load_digits(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=0)
Xa,Xv,ya,yv=train_test_split(Xtr,ytr,test_size=.3,stratify=ytr,random_state=0)
sc=StandardScaler().fit(Xa); Xa,Xv=sc.transform(Xa),sc.transform(Xv)
NEU=[8,16,32,64,128]; ACT=["relu","tanh"]
def sample(r):
    L=r.integers(1,4); return (tuple(int(r.choice(NEU)) for _ in range(L)), str(r.choice(ACT)))
def nparams(a): 
    d=[64]+list(a[0])+[10]; return sum(i*o+o for i,o in zip(d[:-1],d[1:]))
def aval(a,it=40,s=0):
    m=MLPClassifier(a[0],activation=a[1],max_iter=it,random_state=s).fit(Xa,ya); return m.score(Xv,yv)
t=time.time(); r=np.random.default_rng(0)
for _ in range(5):
    a=sample(r); print(a,nparams(a),aval(a),time.time()-t)
