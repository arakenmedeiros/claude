import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
modelos = {
    "Regressão logística": lambda: LogisticRegression(max_iter=2000),
    "Naive Bayes gaussiano": lambda: GaussianNB(),
    "k vizinhos (k=5)": lambda: KNeighborsClassifier(5),
    "Floresta aleatória": lambda: RandomForestClassifier(100, random_state=0),
    "SVM (probability=True)": lambda: SVC(probability=True, random_state=0),
}
print("modelo base              | só rotulados | self-training | ganho (p.p.) | melhor/pior em 10")
for nome, fab in modelos.items():
    a, b = [], []
    for s in range(10):
        r = np.random.RandomState(s)
        idx = np.concatenate([r.choice(np.where(y_tr == d)[0], 5, replace=False) for d in range(10)])
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        a.append(fab().fit(X_tr[idx], y_tr[idx]).score(X_te, y_te))
        b.append(SelfTrainingClassifier(fab(), threshold=0.9).fit(X_tr, yp).score(X_te, y_te))
    d = np.array(b) - np.array(a)
    print(f"{nome:24s} | {100 * np.mean(a):11.1f}% | {100 * np.mean(b):12.1f}% | {100 * d.mean():+11.1f} | {(d > 0).sum()}/{(d < 0).sum()}")
