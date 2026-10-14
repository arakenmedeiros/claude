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


print("40 rótulos (20 por classe), 10 sorteios: self-training com reg. logística, variando o limiar de confiança")
print("representação | limiar | pseudo-rótulos adicionados | precisão deles | F1 macro no teste")
for nome, (Xtr, Xte) in [("TF-IDF", (A, B)), ("LSA(100)", (Z, Zt))]:
    for lim in [None, 0.9, 0.8, 0.7, 0.6]:
        f1s, adds, precs = [], [], []
        for s in range(10):
            r = np.random.RandomState(s)
            idx = np.r_[r.choice(np.where(y_tr == 0)[0], 20, replace=False), r.choice(np.where(y_tr == 1)[0], 20, replace=False)]
            yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
            if lim is None: m = LogisticRegression(max_iter=3000).fit(Xtr[idx], y_tr[idx])
            else:
                m = SelfTrainingClassifier(LogisticRegression(max_iter=3000), threshold=lim).fit(Xtr, yp)
                novos = (m.transduction_ != -1) & (yp == -1); adds.append(novos.sum())
                if novos.sum(): precs.append(np.mean(m.transduction_[novos] == y_tr[novos]))
            f1s.append(f1_score(y_te, m.predict(Xte), average="macro"))
        lm = "sem self-training" if lim is None else f"{lim:.1f}".replace(".", ",")
        extra = f"{np.mean(adds):8.0f} | {100 * np.mean(precs):6.1f}%" if adds and precs else "       0 |     --"
        print(f"{nome:13s} | {lm:17s} | {extra} | {100 * np.mean(f1s):5.1f}% ± {100 * np.std(f1s):.1f}")
