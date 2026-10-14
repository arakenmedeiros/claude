import os, urllib.request
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

def baixar(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino):
        urllib.request.urlretrieve(url, destino)
    return destino

RAW = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
bases = {   # espelhos públicos de bases do repositório UCI
    "Pima Indians Diabetes": ("pima-indians-diabetes.data.csv", 20),
    "Sonar (minas x rochas)": ("sonar.csv", 20),
    "Ionosphere": ("ionosphere.csv", 20),
}
print("base                    | exemplos | atributos | rotulados | só rotulados | self-training | ganho (p.p.) | melhor/pior em 30")
for nome, (arq, n_rot) in bases.items():
    df = pd.read_csv(baixar(RAW + arq, "dados/" + arq), header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = pd.factorize(df.iloc[:, -1])[0]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    a, b = [], []
    for s in range(30):
        r = np.random.RandomState(s)
        idx = np.concatenate([r.choice(np.where(y_tr == c)[0], n_rot // 2, replace=False) for c in (0, 1)])
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        mod = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        a.append(mod().fit(X_tr[idx], y_tr[idx]).score(X_te, y_te))
        b.append(SelfTrainingClassifier(mod(), threshold=0.9).fit(X_tr, yp).score(X_te, y_te))
    d = np.array(b) - np.array(a)
    print(f"{nome:23s} | {len(df):8d} | {X.shape[1]:9d} | {n_rot:9d} | {100 * np.mean(a):11.1f}% | {100 * np.mean(b):12.1f}% | {100 * d.mean():+11.1f} | {(d > 0).sum()}/{(d < 0).sum()}")
