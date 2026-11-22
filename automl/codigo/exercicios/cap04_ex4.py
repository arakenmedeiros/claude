# Capítulo 4 — solução de referência (EX 4)
import numpy as np
from scipy.stats import loguniform
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, train_test_split

X0, y0 = load_wine(return_X_y=True)
espaco = {"svc__C": loguniform(1e-2, 1e3), "svc__gamma": loguniform(1e-5, 1)}
dif = []
for s in range(6):
    X, _, y, _ = train_test_split(X0, y0, train_size=60, stratify=y0, random_state=s)
    pipe = make_pipeline(StandardScaler(), SVC())
    interno = StratifiedKFold(5, shuffle=True, random_state=s)
    externo = StratifiedKFold(5, shuffle=True, random_state=50 + s)
    a = RandomizedSearchCV(pipe, espaco, n_iter=40, cv=interno, random_state=s).fit(X, y).best_score_
    b = cross_val_score(RandomizedSearchCV(pipe, espaco, n_iter=40, cv=interno, random_state=s), X, y, cv=externo).mean()
    dif.append(a - b)
    print(f"semente {s}: busca = {a:.3f} | aninhada = {b:.3f} | otimismo = {a - b:+.3f}")
print(f"Otimismo médio: {np.mean(dif):+.3f}")
