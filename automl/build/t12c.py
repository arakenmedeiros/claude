import warnings; warnings.filterwarnings("ignore")
import numpy as np, time
from scipy.ndimage import shift, rotate
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
X,y=load_digits(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.4,stratify=y,random_state=0)
Xtr,ytr=Xtr[:300],ytr[:300]   # poucos dados
Xa,Xv,ya,yv=train_test_split(Xtr,ytr,test_size=.3,stratify=ytr,random_state=0)
def distorce(Xs,r,dmax,rot,ruido):
    out=[]
    for x in Xs:
        im=x.reshape(8,8)
        im=shift(im,r.uniform(-dmax,dmax,2),order=1,mode="constant")
        if rot>0: im=rotate(im,r.uniform(-rot,rot),reshape=False,order=1,mode="constant")
        im=im+r.normal(0,ruido,im.shape); out.append(im.ravel())
    return np.array(out)
rt=np.random.default_rng(123)
Xte_d=distorce(Xte,rt,1.0,10,0.0); Xv_d=distorce(Xv,np.random.default_rng(5),1.0,10,0.0)
def mk(): return SVC(C=10,gamma="scale")
def treina(pol,seed=0,k=4):
    r=np.random.default_rng(seed); dmax,rot,ru=pol
    Xs=[Xa]; ys=[ya]
    if dmax>0 or rot>0 or ru>0:
        for _ in range(k): Xs.append(distorce(Xa,r,dmax,rot,ru)); ys.append(ya)
    return mk().fit(np.vstack(Xs),np.concatenate(ys))
t=time.time()
m0=treina((0,0,0)); print("sem aug: val limpo",m0.score(Xv,yv),"val distorcido",m0.score(Xv_d,yv),"teste limpo",m0.score(Xte,yte),"teste distorcido",m0.score(Xte_d,yte),time.time()-t)
r=np.random.default_rng(0); best=None
for i in range(15):
    pol=(round(r.uniform(0,2),2),round(r.uniform(0,25),1),round(r.uniform(0,2),2))
    m=treina(pol); sv=m.score(Xv_d,yv); print(pol,round(sv,3))
    if best is None or sv>best[0]: best=(sv,pol,m)
print(best[1],best[0],"teste limpo",best[2].score(Xte,yte),"teste dist",best[2].score(Xte_d,yte),time.time()-t)
