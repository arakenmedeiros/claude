# Capítulo 8 — Engenharia e seleção automática de atributos
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# --- blocos anteriores deste capítulo (encadeados) ---
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

X, y = make_circles(400, noise=0.08, factor=0.45, random_state=0)
original = make_pipeline(LogisticRegression())
com_quadrados = make_pipeline(PolynomialFeatures(2, include_bias=False), LogisticRegression(max_iter=2000))

print("Atributos originais        : acurácia =", round(cross_val_score(original, X, y, cv=5).mean(), 3))
print("Com termos quadráticos     : acurácia =", round(cross_val_score(com_quadrados, X, y, cv=5).mean(), 3))
print("Atributos criados por PolynomialFeatures(2):", PolynomialFeatures(2, include_bias=False).fit(X).get_feature_names_out(["x1", "x2"]))
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

Xd, yd = load_diabetes(return_X_y=True)
kf = KFold(5, shuffle=True, random_state=0)
base = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
poly = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=10.0))

print("Atributos originais (10)        : R² =", round(cross_val_score(base, Xd, yd, cv=kf, scoring="r2").mean(), 3))
print("Com termos quadráticos (65)     : R² =", round(cross_val_score(poly, Xd, yd, cv=kf, scoring="r2").mean(), 3))
# --- fim dos blocos anteriores ---
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

Xw, yw = load_wine(return_X_y=True)
rng = np.random.default_rng(0)
Xr = np.hstack([Xw, rng.normal(size=(len(yw), 100))])           # 13 reais + 100 de ruído
cv = StratifiedKFold(5, shuffle=True, random_state=0)
logis = lambda: LogisticRegression(max_iter=3000)

metodos = {
    "sem seleção (113 atributos)":    make_pipeline(StandardScaler(), logis()),
    "filtro: teste F (k=10)":         make_pipeline(StandardScaler(), SelectKBest(f_classif, k=10), logis()),
    "filtro: informação mútua (k=10)": make_pipeline(StandardScaler(), SelectKBest(lambda a, b: mutual_info_classif(a, b, random_state=0), k=10), logis()),
    "wrapper: RFE (10)":              make_pipeline(StandardScaler(), RFE(logis(), n_features_to_select=10, step=0.2), logis()),
    "embutido: penalidade L1":        make_pipeline(StandardScaler(), SelectFromModel(LogisticRegression(penalty="l1", solver="saga", C=0.2, max_iter=5000)), logis()),
    "embutido: floresta (mediana)":   make_pipeline(SelectFromModel(RandomForestClassifier(100, random_state=0), threshold="median"), StandardScaler(), logis()),
}

print(f"{'método':34s} | acurácia | atributos | reais entre eles")
for nome, pipe in metodos.items():
    acc = cross_val_score(pipe, Xr, yw, cv=cv).mean()
    pipe.fit(Xr, yw)
    sel = pipe[0].get_support() if nome.startswith("embutido: floresta") else (pipe[1].get_support() if len(pipe) > 2 else np.ones(Xr.shape[1], bool))
    print(f"{nome:34s} | {acc:8.3f} | {sel.sum():9d} | {sel[:13].sum():2d} de 13")
