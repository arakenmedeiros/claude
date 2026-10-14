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

print("Reg. logística sobre a LSA (aprendida com todos os textos de treino, sem rótulos): efeito da dimensão e do número de rótulos. F1 macro, 10 sorteios")
print("atributos        | 40 rótulos     | 200 rótulos    | 1000 rótulos")
svd = TruncatedSVD(600, random_state=0).fit(A); Fa, Fb = svd.transform(A), svd.transform(B)
opcoes = [("TF-IDF (todas as palavras)", A, B)] + [(f"LSA com {d} dimensões", Normalizer().fit_transform(Fa[:, :d]), Normalizer().fit_transform(Fb[:, :d])) for d in [10, 30, 100, 300, 600]]
for nome, Xtr, Xte in opcoes:
    linha = []
    for n in [20, 100, 500]:
        v = []
        for s in range(10):
            r = np.random.RandomState(s)
            idx = np.r_[r.choice(np.where(y_tr == 0)[0], n, replace=False), r.choice(np.where(y_tr == 1)[0], n, replace=False)]
            v.append(f1_score(y_te, LogisticRegression(max_iter=3000, C=10.0).fit(Xtr[idx], y_tr[idx]).predict(Xte), average="macro"))
        linha.append(f"{100 * np.mean(v):5.1f}% ± {100 * np.std(v):.1f}")
    print(f"{nome:26s} | " + " | ".join(linha))
