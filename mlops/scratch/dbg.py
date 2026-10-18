import re
src=open('proj.py').read()
# executa até a definição de prod e depois depura
cut=src.index("ref_int = Xtr.std")
exec(compile(src[:cut],"p","exec"))
Xs,ys=prod[1]
c=make_pipeline(StandardScaler(), SVC(C=modelo_melhor.get_params()["svc__C"], gamma=modelo_melhor.get_params()["svc__gamma"])).fit(Xtr,ytr)
print("params",modelo_melhor.get_params()["svc__C"],modelo_melhor.get_params()["svc__gamma"])
print("cand sem peso, semana2 F1:", f1_score(ys,c.predict(Xs),average="macro"), "champ:", f1_score(ys,champion.predict(Xs),average="macro"))
X1,y1=prod[0]
Xt=np.vstack([Xtr,X1]); yt=np.r_[ytr,y1]; w=np.r_[np.ones(len(Xtr)),np.full(len(y1),3.0)]
c2=make_pipeline(StandardScaler(), SVC(C=modelo_melhor.get_params()["svc__C"], gamma=modelo_melhor.get_params()["svc__gamma"]))
c2.fit(Xt,yt,svc__sample_weight=w)
print("cand com peso:", f1_score(ys,c2.predict(Xs),average="macro"))
c3=make_pipeline(StandardScaler(), SVC(C=modelo_melhor.get_params()["svc__C"], gamma=modelo_melhor.get_params()["svc__gamma"])).fit(Xt,yt)
print("cand sem peso c/ semana1:", f1_score(ys,c3.predict(Xs),average="macro"))
print(Xtr.std(axis=0).min(), (Xtr.std(axis=0)<0.5).sum())
