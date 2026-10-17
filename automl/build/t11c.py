import json, numpy as np
T=json.load(open("/tmp/nas_table.json"))
key=lambda a:(tuple(a[0]),a[1])
V={key(r["arch"]):r["v40"] for r in T}; TE={key(r["arch"]):r["t40"] for r in T}
ARCHS=list(V); NEU=[8,16,32,64,128]; ACT=["relu","tanh"]
def mutate(a,r):
    h,act=list(a[0]),a[1]; op=r.integers(4)
    if op==0: h[r.integers(len(h))]=int(r.choice(NEU))
    elif op==1 and len(h)<3: h.insert(r.integers(len(h)+1),int(r.choice(NEU)))
    elif op==2 and len(h)>1: h.pop(r.integers(len(h)))
    else: act=[x for x in ACT if x!=act][0]
    return (tuple(h),act)
def aleatoria(B,r):
    idx=r.choice(len(ARCHS),B,replace=False); return [ARCHS[i] for i in idx]
def evolucao(B,r,P=8,S=3):
    hist=[]; pop=[]; seen=set()
    while len(hist)<P:
        a=ARCHS[r.integers(len(ARCHS))]
        if a not in seen: seen.add(a); hist.append(a); pop.append(a)
    while len(hist)<B:
        amostra=[pop[i] for i in r.choice(len(pop),S,replace=False)]
        pai=max(amostra,key=V.get)
        for _ in range(50):
            f=mutate(pai,r)
            if f not in seen: break
        else: f=ARCHS[r.integers(len(ARCHS))]
        if f in seen: continue
        seen.add(f); hist.append(f); pop.append(f); pop.pop(0)
    return hist
B=30; R=300; best_all=max(V.values())
rank=lambda a: 1+sum(v>V[a] for v in V.values())
for nome,fn in (("aleatória",aleatoria),("evolução",evolucao)):
    res=[]; 
    for s in range(R):
        h=fn(B,np.random.default_rng(s)); b=max(h,key=V.get); res.append((V[b],rank(b),TE[b]))
    res=np.array(res); print(nome,"mediana val",np.median(res[:,0]),"posto mediano",np.median(res[:,1]),"top5%",np.mean(res[:,1]<=15),"teste",res[:,2].mean())
