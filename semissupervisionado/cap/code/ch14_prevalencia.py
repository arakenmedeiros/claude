import os, urllib.request
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.metrics import f1_score

def baixar(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino): urllib.request.urlretrieve(url, destino)
    return destino

df = pd.read_csv(baixar("https://raw.githubusercontent.com/justmarkham/DAT7/master/data/yelp.csv", "dados/yelp.csv"))
df = df[df["stars"].isin([1, 5])]; y = (df["stars"] == 5).astype(int).values; textos = df["text"].values
print(f"Avaliações do Yelp com 1 ou 5 estrelas: {len(df)} | positivas (5 estrelas): {100 * y.mean():.1f}%")
t_tr, t_te, y_tr, y_te = train_test_split(textos, y, test_size=0.3, stratify=y, random_state=0)
tf = TfidfVectorizer(min_df=2, stop_words="english", sublinear_tf=True).fit(t_tr); A, B = tf.transform(t_tr), tf.transform(t_te)
cv = CountVectorizer(min_df=2, stop_words="english").fit(t_tr); Ac, Bc = cv.transform(t_tr), cv.transform(t_te)
svd = TruncatedSVD(100, random_state=0).fit(A); nor = Normalizer()
Z, Zt = nor.fit_transform(svd.transform(A)), nor.transform(svd.transform(B))     # LSA aprendido com TODOS os textos de treino, sem usar rótulos

def nb_em(Xl, yl, Xu, lam=1.0, iters=10):
    nb = MultinomialNB(alpha=1.0).fit(Xl, yl)
    for _ in range(iters):
        P = nb.predict_proba(Xu); X_all = sp.vstack([Xl, Xu, Xu]); y_all = np.r_[yl, np.zeros(Xu.shape[0], int), np.ones(Xu.shape[0], int)]
        nb = MultinomialNB(alpha=1.0).fit(X_all, y_all, sample_weight=np.r_[np.ones(len(yl)), lam * P[:, 0], lam * P[:, 1]])
    return nb


print("40 rótulos sorteados ao acaso (proporção real de classes, mínimo de 3 de cada), F1 macro no teste, 10 sorteios")
nomes = ["TF-IDF + reg. logística (referência)", "LSA(100) + reg. logística", "LSA(100) + Label Spreading (knn, k = 10)", "Naive Bayes + EM"]
res = {n: [] for n in nomes}; pos = {n: [] for n in nomes}; negativos = []
for s in range(10):
    r = np.random.RandomState(s)
    while True:
        idx = r.choice(len(y_tr), 40, replace=False)
        if 3 <= y_tr[idx].sum() <= 37: break
    negativos.append(40 - y_tr[idx].sum()); resto = np.setdiff1d(np.arange(len(y_tr)), idx); yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
    def f(m, Xt, nome):
        p = m.predict(Xt); pos[nome].append(p.mean()); return f1_score(y_te, p, average="macro")
    res[nomes[0]].append(f(LogisticRegression(max_iter=3000).fit(A[idx], y_tr[idx]), B, nomes[0]))
    res[nomes[1]].append(f(LogisticRegression(max_iter=3000).fit(Z[idx], y_tr[idx]), Zt, nomes[1]))
    res[nomes[2]].append(f(LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=200).fit(Z, yp), Zt, nomes[2]))
    res[nomes[3]].append(f(nb_em(Ac[idx], y_tr[idx], Ac[resto]), Bc, nomes[3]))
print(f"rótulos negativos por sorteio (em média): {np.mean(negativos):.1f} de 40")
print(f"{'':45s}  F1 macro        | % previsto como positivo (real: {100 * y_te.mean():.1f}%)")
for k, v in res.items(): print(f"  {k:43s}: {100 * np.mean(v):5.1f}% ± {100 * np.std(v):.1f} | {100 * np.mean(pos[k]):.1f}%")
