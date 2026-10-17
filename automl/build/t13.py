import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
r=np.random.default_rng(0); n=6000
g=(r.random(n)<.3).astype(int)               # 1 = grupo minoritário
z=r.normal(0,1,n)                            # capacidade real de pagamento (latente, desconhecida na prática)
renda=np.exp(0.5*z+0.35*r.normal(size=n)-0.25*g)*3.0
emprego=np.clip(5+2*z+r.normal(0,2,n),0,None)
divida=np.clip(0.4-0.1*z+r.normal(0,.1,n),0,1)
cep=0.8*g+0.6*r.normal(size=n)               # proxy do grupo
hist=z+r.normal(0,1,n)-0.7*g*0               # score
y_real=(z>0.0).astype(int)                    # quem de fato pagaria
y_hist=((z-0.6*g+r.normal(0,.5,n))>0).astype(int)   # decisões históricas enviesadas
X=np.c_[renda,emprego,divida,cep,hist]; names=["renda","tempo_emprego","razao_divida","cep_score","score_historico"]
Xtr,Xte,ytr,yte,gtr,gte,ztr,zte,yrtr,yrte=train_test_split(X,y_hist,g,z,y_real,test_size=.3,random_state=0,stratify=y_hist)
print("taxa aprovação histórica por grupo:",ytr[gtr==0].mean(),ytr[gtr==1].mean())
print("taxa real de bons pagadores:",yrtr[gtr==0].mean(),yrtr[gtr==1].mean())
for nome,cols in (("com cep",[0,1,2,3,4]),("sem cep",[0,1,2,4])):
    m=HistGradientBoostingClassifier(random_state=0).fit(Xtr[:,cols],ytr); p=m.predict(Xte[:,cols])
    sel=[p[gte==k].mean() for k in (0,1)]
    tpr_real=[p[(gte==k)&(yrte==1)].mean() for k in (0,1)]
    print(nome,"acc",(p==yte).mean(),"sel",np.round(sel,3),"TPR real",np.round(tpr_real,3))
m=HistGradientBoostingClassifier(random_state=0).fit(Xtr,ytr)
pi=permutation_importance(m,Xte,yte,n_repeats=10,random_state=0); print(dict(zip(names,pi.importances_mean.round(3))))
import shap, lightgbm as lgb
m=lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,num_leaves=15,random_state=0,verbose=-1).fit(Xtr,ytr)
ex=shap.TreeExplainer(m); sv=ex.shap_values(Xte); sv=sv[1] if isinstance(sv,list) else sv
print(sv.shape, dict(zip(names,np.abs(sv).mean(0).round(3))))
p=m.predict_proba(Xte)[:,1]
# limiares por grupo para igualar taxa de seleção
t0=0.5; sel0=(p[gte==0]>=t0).mean()
t1=np.quantile(p[gte==1],1-sel0); pred=np.where(gte==0,p>=t0,p>=t1).astype(int)
print("t1",t1,"acc hist",(pred==yte).mean(),"acc real",(pred==yrte).mean(),"sel",[pred[gte==k].mean() for k in (0,1)],"TPR real",[pred[(gte==k)&(yrte==1)].mean() for k in (0,1)])
pred0=(p>=.5).astype(int); print("base acc hist",(pred0==yte).mean(),"acc real",(pred0==yrte).mean(),"TPR real",[pred0[(gte==k)&(yrte==1)].mean() for k in (0,1)])
print("sv for 1 example",sv[0].round(2),p[0])
