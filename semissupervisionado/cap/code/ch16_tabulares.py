import os, urllib.request
import numpy as np, pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading

def baixar(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino): urllib.request.urlretrieve(url, destino)
    return destino
def csv(nome, coluna_y): 
    d = pd.read_csv(baixar("https://raw.githubusercontent.com/jbrownlee/Datasets/master/" + nome, "dados/" + nome), header=None)
    return d.drop(columns=coluna_y).values.astype(float), pd.factorize(d[coluna_y])[0]

bases = {"Iris": load_iris(return_X_y=True), "Wine": load_wine(return_X_y=True), "Breast Cancer": load_breast_cancer(return_X_y=True), "Digits": load_digits(return_X_y=True),
         "Sonar": csv("sonar.csv", 60), "Ionosphere": csv("ionosphere.csv", 34), "Pima (diabetes)": csv("pima-indians-diabetes.data.csv", 8), "Glass": csv("glass.csv", 9)}
POR_CLASSE, SORTEIOS = 5, 20
print(f"{POR_CLASSE} rótulos por classe, {SORTEIOS} sorteios. Acurácia no teste (30%); ganho = método semissupervisionado menos a regressão logística só com os rotulados")
print("base            | exemplos | classes | reg. logística | self-training (limiar 0,9) | Label Spreading (knn, k = 7)")
for nome, (X, y) in bases.items():
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    esc = StandardScaler().fit(X_tr); X_tr, X_te = esc.transform(X_tr), esc.transform(X_te); classes = np.unique(y)
    a, b, c = [], [], []
    for s in range(SORTEIOS):
        r = np.random.RandomState(s); idx = np.concatenate([r.choice(np.where(y_tr == k)[0], POR_CLASSE, replace=False) for k in classes]); yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        a.append(LogisticRegression(max_iter=3000).fit(X_tr[idx], y_tr[idx]).score(X_te, y_te))
        b.append(SelfTrainingClassifier(LogisticRegression(max_iter=3000), threshold=0.9).fit(X_tr, yp).score(X_te, y_te))
        c.append(LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=200).fit(X_tr, yp).score(X_te, y_te))
    a, b, c = map(np.array, (a, b, c)); f = lambda d: f"{100 * d.mean():+5.1f} ± {100 * d.std():4.1f}"
    print(f"{nome:15s} | {len(y):8d} | {len(classes):7d} | {100 * a.mean():5.1f}% ± {100 * a.std():4.1f} | {100 * b.mean():5.1f}% ({f(b - a)}) | {100 * c.mean():5.1f}% ({f(c - a)})")
