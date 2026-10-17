import warnings; warnings.filterwarnings("ignore")
import numpy as np, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split
r=np.random.default_rng(0)
POS=["ótimo","bom","excelente","rápido","perfeito","útil"]; NEG=["ruim","péssimo","lento","fraco","horrível","inútil"]
SUJ=["o produto","a entrega","o atendimento","o aplicativo","a embalagem"]; VERB=["é","foi","achei","parece"]
FIL=["no geral","sinceramente","para mim","nesta semana","de verdade","como esperado"]
def gera(n):
    docs=[];ys=[]
    for _ in range(n):
        pos=r.random()<.5; adj=r.choice(POS if pos else NEG); neg=r.random()<.4
        if neg: pos=not pos
        frase=f"{r.choice(SUJ)} {r.choice(VERB)} {'não ' if neg else ''}{adj}"
        if r.random()<.6: frase=r.choice(FIL)+" "+frase
        docs.append(frase); ys.append(int(pos))
    return docs,np.array(ys)
D,Y=gera(1200)
Dtr,Dte,ytr,yte=train_test_split(D,Y,test_size=.25,stratify=Y,random_state=0)
print(Dtr[:5],ytr[:5])
for ng in [(1,1),(1,2),(1,3)]:
    for nm,c in (("LR",LogisticRegression(max_iter=2000)),("SVM",LinearSVC()),("NB",MultinomialNB())):
        p=make_pipeline(TfidfVectorizer(ngram_range=ng,sublinear_tf=True),c); print(ng,nm,round(cross_val_score(p,Dtr,ytr,cv=5).mean(),3))
