"""Mini-TPOT didático: algoritmo genético que evolui pipelines (escala, redução, modelo)."""
import numpy as np, copy
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def carregar():
    X,y=load_digits(return_X_y=True); return X/16.0,y
def sortear(rng):
    g={"escala":str(rng.choice(["nenhuma","padrao","minmax"])),"pca":int(rng.choice([0,0,10,20,30,40])),"modelo":str(rng.choice(["knn","svm","logistica","floresta"]))}
    g.update(sortear_modelo(rng,g["modelo"])); return g
def sortear_modelo(rng,m):
    if m=="knn": return {"k":int(rng.integers(1,30))}
    if m=="svm": return {"C":float(10**rng.uniform(-2,3)),"gamma":float(10**rng.uniform(-5,0))}
    if m=="logistica": return {"C":float(10**rng.uniform(-3,3))}
    return {"arvores":int(rng.integers(20,150)),"prof":int(rng.integers(3,20))}
def construir(g):
    passos=[]
    if g["escala"]=="padrao": passos.append(StandardScaler())
    if g["escala"]=="minmax": passos.append(MinMaxScaler())
    if g["pca"]>0: passos.append(PCA(g["pca"],random_state=0))
    m=g["modelo"]
    passos.append(KNeighborsClassifier(g["k"]) if m=="knn" else SVC(C=g["C"],gamma=g["gamma"]) if m=="svm" else LogisticRegression(C=g["C"],max_iter=300) if m=="logistica" else RandomForestClassifier(g["arvores"],max_depth=g["prof"],random_state=0,n_jobs=-1))
    return make_pipeline(*passos)
def aptidao(g,X,y,cache):
    chave=str(sorted(g.items()))
    if chave not in cache: cache[chave]=float(cross_val_score(construir(g),X,y,cv=StratifiedKFold(3,shuffle=True,random_state=0)).mean())
    return cache[chave]
def cruzar(a,b,rng):
    filho={}
    for k in ("escala","pca"): filho[k]=a[k] if rng.random()<0.5 else b[k]
    pai=a if rng.random()<0.5 else b; filho["modelo"]=pai["modelo"]
    filho.update({k:v for k,v in pai.items() if k not in ("escala","pca","modelo")}); return filho
def mutar(g,rng,p=0.3):
    g=copy.deepcopy(g)
    if rng.random()<p: g["escala"]=str(rng.choice(["nenhuma","padrao","minmax"]))
    if rng.random()<p: g["pca"]=int(rng.choice([0,0,10,20,30,40]))
    if rng.random()<p*0.7:
        g["modelo"]=str(rng.choice(["knn","svm","logistica","floresta"])); 
        for k in [k for k in g if k not in ("escala","pca","modelo")]: del g[k]
        g.update(sortear_modelo(rng,g["modelo"]))
    else:
        for k,v in list(g.items()):
            if k in ("escala","pca","modelo"): continue
            if rng.random()<0.6:
                if k in ("C","gamma"): g[k]=float(np.clip(v*10**rng.normal(0,0.4),1e-5 if k=="gamma" else 1e-3,1 if k=="gamma" else 1e3))
                elif k=="k": g[k]=int(np.clip(v+rng.integers(-3,4),1,40))
                elif k=="arvores": g[k]=int(np.clip(v+rng.integers(-20,21),10,200))
                elif k=="prof": g[k]=int(np.clip(v+rng.integers(-3,4),2,25))
    return g
def evoluir(X,y,semente,pop=16,geracoes=6,elite=2,cache=None):
    rng=np.random.default_rng(semente); cache={} if cache is None else cache
    populacao=[sortear(rng) for _ in range(pop)]; historico=[]; todos=[]
    for ger in range(geracoes+1):
        notas=np.array([aptidao(g,X,y,cache) for g in populacao]); todos+=list(zip(populacao,notas))
        historico.append((len(cache),float(notas.max()),float(notas.mean())))
        if ger==geracoes: break
        ordem=np.argsort(-notas); nova=[copy.deepcopy(populacao[i]) for i in ordem[:elite]]
        while len(nova)<pop:
            torneio=lambda: populacao[max(rng.choice(pop,3,replace=False),key=lambda i:notas[i])]
            nova.append(mutar(cruzar(torneio(),torneio(),rng),rng))
        populacao=nova
    return historico,todos,cache
def aleatoria(X,y,semente,n_aval):
    rng=np.random.default_rng(10_000+semente); cache={}; melhor=0; curva=[]
    for _ in range(n_aval):
        g=sortear(rng); melhor=max(melhor,aptidao(g,X,y,cache)); curva.append(melhor)
    return curva
