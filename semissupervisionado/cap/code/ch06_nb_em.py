import os, urllib.request
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

def baixar(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino): urllib.request.urlretrieve(url, destino)
    return destino

df = pd.read_csv(baixar("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv", "dados/sms.tsv"), sep="\t", header=None, names=["rotulo", "texto"])
y = (df["rotulo"] == "spam").astype(int).values
t_tr, t_te, y_tr, y_te = train_test_split(df["texto"].values, y, test_size=0.3, stratify=y, random_state=0)
vec = CountVectorizer().fit(t_tr); A, B = vec.transform(t_tr), vec.transform(t_te)

def nb_em(Xl, yl, Xu, lam=1.0, iters=10):
    """Naive Bayes multinomial + EM (Nigam et al., 2000). Cada exemplo sem rótulo entra duas vezes, com peso lam x probabilidade de cada classe."""
    nb = MultinomialNB(alpha=1.0).fit(Xl, yl)
    for _ in range(iters):
        P = nb.predict_proba(Xu)                                                  # E-step: probabilidade de cada classe
        X_all = sp.vstack([Xl, Xu, Xu]); y_all = np.r_[yl, np.zeros(Xu.shape[0], int), np.ones(Xu.shape[0], int)]
        w = np.r_[np.ones(len(yl)), lam * P[:, 0], lam * P[:, 1]]
        nb = MultinomialNB(alpha=1.0).fit(X_all, y_all, sample_weight=w)          # M-step: reestima com pesos
    return nb

print("40 rótulos (20 por classe), F1 da classe spam, 10 sorteios")
prior = [1 - y_tr.mean(), y_tr.mean()]      # proporção real de spam (informação que, na prática, nem sempre se tem)
res = {"só rotulados": [], "só rotulados, prior real": []}; lams = [0.05, 0.2, 0.5, 1.0]
for lam in lams: res[f"NB + EM (peso {lam})"] = []
for s in range(10):
    r = np.random.RandomState(s)
    idx = np.r_[r.choice(np.where(y_tr == 0)[0], 20, replace=False), r.choice(np.where(y_tr == 1)[0], 20, replace=False)]
    resto = np.setdiff1d(np.arange(len(y_tr)), idx)
    res["só rotulados"].append(f1_score(y_te, MultinomialNB(alpha=1.0).fit(A[idx], y_tr[idx]).predict(B)))
    res["só rotulados, prior real"].append(f1_score(y_te, MultinomialNB(alpha=1.0, class_prior=prior).fit(A[idx], y_tr[idx]).predict(B)))
    for lam in lams:
        res[f"NB + EM (peso {lam})"].append(f1_score(y_te, nb_em(A[idx], y_tr[idx], A[resto], lam=lam).predict(B)))
for k, v in res.items(): print(f"  {k:25s}: {100 * np.mean(v):5.1f}% ± {100 * np.std(v):.1f}")
