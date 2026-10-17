"""Experimentos do Cap. 9 (usados nas figuras; o código do capítulo é equivalente)."""
import numpy as np, json, os, collections, warnings; warnings.filterwarnings("ignore")
from scipy.stats import skew, entropy
from sklearn.datasets import make_classification, load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss
CACHE="/home/user/claude/automl/build/f9_cache.json"
ALG={"knn":lambda: make_pipeline(StandardScaler(),KNeighborsClassifier()),"svm":lambda: make_pipeline(StandardScaler(),SVC()),
     "logistica":lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),"floresta":lambda: RandomForestClassifier(60,random_state=0,n_jobs=-1)}
def metaatr(X,y):
    n,d=X.shape; c=len(np.unique(y)); p=np.bincount(y)/n
    sk=np.abs(skew(X,axis=0)); cor=np.abs(np.corrcoef(X.T)); np.fill_diagonal(cor,0)
    lm=cross_val_score(DecisionTreeClassifier(max_depth=1,random_state=0),X,y,cv=3).mean()
    return [float(np.log(n)),float(np.log(d)),float(n/d),float(c),float(entropy(p)/np.log(c)),float(np.nanmean(sk)),float(np.nanmean(cor)),float(lm)]
def avaliar(X,y):
    cv=StratifiedKFold(3,shuffle=True,random_state=0); return {k:float(cross_val_score(f(),X,y,cv=cv).mean()) for k,f in ALG.items()}
def gerar_conjunto(rng,i,base=0):
    n=int(rng.integers(150,500)); d=int(rng.integers(8,40)); inf=int(rng.integers(4,min(d-2,12)+1)); c=int(rng.integers(2,5))
    return make_classification(n_samples=n,n_features=d,n_informative=inf,n_redundant=int(rng.integers(0,(d-inf)//2+1)),n_classes=c,n_clusters_per_class=int(rng.integers(1,3)),class_sep=float(rng.uniform(0.4,2.0)),flip_y=float(rng.uniform(0,0.1)),random_state=base+i)
def experimento_meta():
    rng=np.random.default_rng(0); M=[];R=[]
    for i in range(150):
        X,y=gerar_conjunto(rng,i); M.append(metaatr(X,y)); R.append(avaliar(X,y))
    M=np.array(M); best=np.array([max(r,key=r.get) for r in R]); acc=[];reg=[]
    for i in range(len(M)):
        tr=np.arange(len(M))!=i; pred=RandomForestClassifier(200,random_state=0).fit(M[tr],best[tr]).predict(M[i:i+1])[0]
        acc.append(bool(pred==best[i])); reg.append(R[i][pred]-max(R[i].values()))
    maj=collections.Counter(best).most_common(1)[0][0]
    regmaj=[R[i][maj]-max(R[i].values()) for i in range(len(M))]
    clf=RandomForestClassifier(300,random_state=0).fit(M,best); reais={}
    for nome,l in [("iris",load_iris),("wine",load_wine),("cancer",load_breast_cancer),("digits",load_digits)]:
        X,y=l(return_X_y=True); r=avaliar(X,y); reais[nome]={"previsto":str(clf.predict(np.array(metaatr(X,y))[None])[0]),"notas":r}
    return {"contagem":dict(collections.Counter(best)),"acc_meta":float(np.mean(acc)),"acc_maj":float(collections.Counter(best).most_common(1)[0][1]/len(M)),"regret_meta":float(np.mean(reg)),"regret_maj":float(np.mean(regmaj)),"maj":maj,"reais":reais}
CS=[0.1,1,10,100,1000]; GREL=[1e-3,1e-2,1e-1,1,10]; CFGS=[(c,g) for c in CS for g in GREL]
def perf_svm(X,y):
    cv=StratifiedKFold(3,shuffle=True,random_state=0)
    return [float(cross_val_score(make_pipeline(StandardScaler(),SVC(C=c,gamma=g/X.shape[1])),X,y,cv=cv).mean()) for c,g in CFGS]
def experimento_portfolio():
    rng=np.random.default_rng(5); S=[]
    for i in range(40):
        X,y=gerar_conjunto(rng,i,base=1000); S.append(perf_svm(X,y))
    S=np.array(S); port=[]
    for _ in range(3):
        port.append(max([j for j in range(len(CFGS)) if j not in port],key=lambda j:np.mean(np.max(S[:,port+[j]],axis=1))))
    out={"portfolio":[list(CFGS[j]) for j in port],"alvos":{}}
    for nome,l in [("iris",load_iris),("wine",load_wine),("cancer",load_breast_cancer),("digits",load_digits)]:
        X,y=l(return_X_y=True); p=np.array(perf_svm(X,y)); r=np.random.default_rng(0)
        al={k:float(np.mean([np.max(p[r.choice(len(CFGS),k,replace=False)]) for _ in range(500)])) for k in (1,3,6)}
        out["alvos"][nome]={"portfolio":[float(p[port[0]]),float(p[port[:2]].max()),float(p[port].max())],"aleatoria":al,"melhor":float(p.max())}
    return out
def experimento_ensemble():
    X,y=make_classification(n_samples=2000,n_features=20,n_informative=8,n_redundant=4,n_classes=3,n_clusters_per_class=3,class_sep=0.9,flip_y=0.08,random_state=1)
    Xa,Xte,ya,yte=train_test_split(X,y,test_size=0.25,stratify=y,random_state=0); Xtr,Xva,ytr,yva=train_test_split(Xa,ya,test_size=0.33,stratify=ya,random_state=0)
    rng=np.random.default_rng(0); lib=[]
    for i in range(40):
        k=str(rng.choice(["svm","knn","lr","rf","hgb"]))
        if k=="svm": m=SVC(C=10**rng.uniform(-1,3),gamma=10**rng.uniform(-4,-0.5),probability=True,random_state=0)
        elif k=="knn": m=KNeighborsClassifier(int(rng.integers(1,25)),weights="distance")
        elif k=="lr": m=LogisticRegression(C=10**rng.uniform(-3,2),max_iter=2000)
        elif k=="rf": m=RandomForestClassifier(int(rng.integers(20,120)),max_depth=int(rng.integers(3,20)),random_state=0,n_jobs=-1)
        else: m=HistGradientBoostingClassifier(learning_rate=10**rng.uniform(-1.7,-0.3),max_iter=60,max_leaf_nodes=int(rng.integers(4,30)),random_state=0)
        m.fit(Xtr,ytr); lib.append((k,m.predict_proba(Xva),m.predict_proba(Xte)))
    va=[accuracy_score(yva,p.argmax(1)) for _,p,_ in lib]; te=[accuracy_score(yte,pt.argmax(1)) for *_,pt in lib]
    b=int(np.argmax(va)); sel=[]; cur=None
    for step in range(25):
        best=None
        for j,(k,pv,pt) in enumerate(lib):
            cand=pv if not sel else (cur*len(sel)+pv)/(len(sel)+1); ll=log_loss(yva,cand,labels=range(3))
            if best is None or ll<best[0]: best=(ll,j)
        cur=lib[best[1]][1] if not sel else (cur*len(sel)+lib[best[1]][1])/(len(sel)+1); sel.append(best[1])
    w=np.bincount(sel,minlength=len(lib)); ens=sum(w[j]*lib[j][2] for j in range(len(lib)))/w.sum()
    return {"melhor_val":float(va[b]),"melhor_teste":float(te[b]),"melhor_tipo":lib[b][0],"media_teste":float(np.mean(te)),"ens_teste":float(accuracy_score(yte,ens.argmax(1))),
            "ens_pesos":{f"{lib[j][0]}#{j}":int(w[j]) for j in range(len(lib)) if w[j]>0},"todos_teste":float(accuracy_score(yte,(sum(pt for *_,pt in lib)/len(lib)).argmax(1))),"tipos":[k for k,_,_ in lib],"teste_ind":[float(t) for t in te]}
def tudo():
    if os.path.exists(CACHE): return json.load(open(CACHE))
    d={"meta":experimento_meta(),"portfolio":experimento_portfolio(),"ensemble":experimento_ensemble()}; json.dump(d,open(CACHE,"w")); return d
if __name__=="__main__": print(json.dumps(tudo(),indent=1)[:1500])
