# Capítulo 8 — Engenharia e seleção automática de atributos
# Bloco 5. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
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
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

rng = np.random.default_rng(0)
n = 800
clientes = pd.DataFrame({"id": range(n), "idade": rng.integers(18, 75, n), "cidade": rng.choice(["A", "B", "C"], n)})
n_compras = rng.poisson(8, n) + 1
media = rng.uniform(30, 120, n)
variab = rng.uniform(0.1, 1.2, n)
linhas = []
for i in range(n):
    valores = media[i] * np.exp(rng.normal(0, variab[i], n_compras[i]))
    linhas += [(i, v) for v in valores]
compras = pd.DataFrame(linhas, columns=["id_cliente", "valor"])

# "DFS" manual: agregações das compras de cada cliente
agg = compras.groupby("id_cliente")["valor"].agg(media_valor="mean", max_valor="max", std_valor="std", n_compras="count").reset_index()
agg = agg.rename(columns={"id_cliente": "id"}).fillna(0)
dados = clientes.merge(agg, on="id")
dados["cv_valor"] = dados["std_valor"] / dados["media_valor"]           # atributo derivado (razão)

# O cancelamento depende da variabilidade relativa e do nº de compras (padrão que simulamos)
risco = 2.0 * dados["cv_valor"] - 0.02 * dados["n_compras"]
y = (rng.random(n) < 1 / (1 + np.exp(-3 * (risco - risco.median())))).astype(int)

so_clientes = pd.get_dummies(dados[["idade", "cidade"]]).astype(float)
com_agregados = pd.concat([so_clientes, dados[["media_valor", "max_valor", "std_valor", "n_compras", "cv_valor"]]], axis=1)

cv = StratifiedKFold(5, shuffle=True, random_state=0)
for nome, Xa in [("apenas a tabela de clientes", so_clientes), ("com atributos agregados das compras", com_agregados)]:
    auc = cross_val_score(GradientBoostingClassifier(random_state=0), Xa, y, cv=cv, scoring="roc_auc").mean()
    print(f"{nome:38s} AUC = {auc:.3f}")
# --- fim dos blocos anteriores ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer

class GeradorDeAtributos(BaseEstimator, TransformerMixin):
    """Acrescenta quadrados, logaritmos, produtos e razões das primeiras `m` colunas."""
    def __init__(self, m=8):
        self.m = m
    def fit(self, X, y=None):
        self.colunas_ = np.arange(min(self.m, np.asarray(X).shape[1]))
        return self
    def transform(self, X):
        X = np.asarray(X); c = X[:, self.colunas_]
        novos = [X, c ** 2, np.log1p(np.abs(c))]
        for i in range(c.shape[1]):
            for j in range(i + 1, c.shape[1]):
                novos.append((c[:, i] * c[:, j])[:, None])
                novos.append((c[:, i] / (np.abs(c[:, j]) + 1e-3))[:, None])
        return np.hstack(novos)

print("Atributos gerados a partir de 8 colunas:", GeradorDeAtributos(8).fit(Xw).transform(Xw).shape[1] - Xw.shape[1], "(mais as originais)")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
print(f"\n{'conjunto':8s} | {'modelo':22s} | {'original':8s} | com atributos gerados")
for nome, carregar in [("Wine", load_wine), ("Cancer", load_breast_cancer)]:
    Xc, yc = carregar(return_X_y=True)
    pares = [("regressão logística", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
              make_pipeline(GeradorDeAtributos(8), StandardScaler(), SelectKBest(f_classif, k=15), LogisticRegression(max_iter=3000))),
             ("gradient boosting", GradientBoostingClassifier(random_state=0),
              make_pipeline(GeradorDeAtributos(8), SelectKBest(f_classif, k=15), GradientBoostingClassifier(random_state=0)))]
    for modelo, base, com_fe in pares:
        a, b = cross_val_score(base, Xc, yc, cv=cv).mean(), cross_val_score(com_fe, Xc, yc, cv=cv).mean()
        print(f"{nome:8s} | {modelo:22s} | {a:8.3f} | {b:.3f}")
