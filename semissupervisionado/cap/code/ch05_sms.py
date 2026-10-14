import os, urllib.request
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def baixar(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino):
        urllib.request.urlretrieve(url, destino)
    return destino

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"      # espelho da SMS Spam Collection (UCI/Kaggle)
df = pd.read_csv(baixar(url, "dados/sms.tsv"), sep="\t", header=None, names=["rotulo", "texto"])
y = (df["rotulo"] == "spam").astype(int).values
print("mensagens:", len(df), "| spam: %.1f%%" % (100 * y.mean()))

t_tr, t_te, y_tr, y_te = train_test_split(df["texto"].values, y, test_size=0.3, stratify=y, random_state=0)
# Visão 1: palavras. Visão 2: sequências de caracteres (n-gramas de 2 a 4). Duas "formas de ler" o mesmo texto.
v1 = TfidfVectorizer(lowercase=True).fit(t_tr); v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit(t_tr)
A1, A2 = v1.transform(t_tr), v2.transform(t_tr); B1, B2 = v1.transform(t_te), v2.transform(t_te)

def co_treino(X1, X2, yp, rodadas=10, por_classe=(20, 3)):
    y = yp.copy()
    for _ in range(rodadas):
        com = y != -1
        h1 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X1[com], y[com]); h2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X2[com], y[com])
        sem = np.where(~com)[0]
        novos = {}
        for h, X in ((h1, X1), (h2, X2)):
            p = h.predict_proba(X[sem]); conf = p.max(1); cl = h.classes_[p.argmax(1)]
            for c in h.classes_:
                cand = np.where(cl == c)[0]
                for i in cand[np.argsort(-conf[cand])[:por_classe[c]]]: novos.setdefault(sem[i], c)   # classe 0 (legítima): 20 por rodada; classe 1 (spam, rara): 3
        for i, c in novos.items(): y[i] = c
    com = y != -1
    return [LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[com], y[com]) for X in (X1, X2)]

def f1_soma(h1, h2, X1, X2, y): return f1_score(y, (h1.predict_proba(X1) + h2.predict_proba(X2)).argmax(1))

res = {k: [] for k in ["palavras (só rotulados)", "caracteres (só rotulados)", "duas visões (só rotulados)", "co-training (2 visões)"]}
for s in range(10):
    r = np.random.RandomState(s)
    idx = np.r_[r.choice(np.where(y_tr == 0)[0], 20, replace=False), r.choice(np.where(y_tr == 1)[0], 20, replace=False)]    # 40 rótulos (20 por classe)
    yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
    g1 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(A1[idx], y_tr[idx]); g2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(A2[idx], y_tr[idx])
    res["palavras (só rotulados)"].append(f1_score(y_te, g1.predict(B1))); res["caracteres (só rotulados)"].append(f1_score(y_te, g2.predict(B2)))
    res["duas visões (só rotulados)"].append(f1_soma(g1, g2, B1, B2, y_te))
    h1, h2 = co_treino(A1, A2, yp); res["co-training (2 visões)"].append(f1_soma(h1, h2, B1, B2, y_te))
print("método                       | F1 da classe spam (média ± desvio, 10 sorteios, 40 rótulos)")
for k, v in res.items(): print(f"{k:28s} | {100 * np.mean(v):5.1f}% ± {100 * np.std(v):.1f}")
