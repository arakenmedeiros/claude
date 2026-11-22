# Capítulo 3 — solução de referência (EX 3)
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X0, y = load_wine(return_X_y=True)
res = []
for semente in range(5):
    rng = np.random.default_rng(semente)
    X = np.hstack([X0, rng.normal(size=(len(y), 1000))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=60, stratify=y, random_state=semente)

    # (a) seleção FORA do pipeline (vazamento)
    sel = SelectKBest(f_classif, k=15).fit(Xtr, ytr)
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    val_a = cross_val_score(modelo, sel.transform(Xtr), ytr, cv=5).mean()
    teste_a = modelo.fit(sel.transform(Xtr), ytr).score(sel.transform(Xte), yte)

    # (b) seleção DENTRO do pipeline
    pipe = make_pipeline(SelectKBest(f_classif, k=15), StandardScaler(), LogisticRegression(max_iter=3000))
    val_b = cross_val_score(pipe, Xtr, ytr, cv=5).mean()
    teste_b = pipe.fit(Xtr, ytr).score(Xte, yte)
    res.append((val_a, teste_a, val_b, teste_b))

va, ta, vb, tb = np.mean(res, axis=0)
print(f"(a) fora do pipeline  : validação = {va:.3f} | teste final = {ta:.3f} | otimismo = {va - ta:+.3f}")
print(f"(b) dentro do pipeline: validação = {vb:.3f} | teste final = {tb:.3f} | otimismo = {vb - tb:+.3f}")
