# Capítulo 8 — solução de referência (EX 4)
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

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

g = compras.groupby("id_cliente")["valor"]
agg = g.agg(media_valor="mean", max_valor="max", std_valor="std", n_compras="count")
agg["min_valor"] = g.min(); agg["mediana_valor"] = g.median()
agg["acima_100"] = compras.assign(a=compras["valor"] > 100).groupby("id_cliente")["a"].sum()
agg = agg.fillna(0).reset_index().rename(columns={"id_cliente": "id"})
dados = clientes.merge(agg, on="id")
dados["cv_valor"] = dados["std_valor"] / dados["media_valor"]

risco = 2.0 * dados["cv_valor"] - 0.02 * dados["n_compras"]
y = (rng.random(n) < 1 / (1 + np.exp(-3 * (risco - risco.median())))).astype(int)

base = pd.get_dummies(dados[["idade", "cidade"]]).astype(float)
col_a = ["media_valor", "max_valor", "std_valor", "n_compras", "cv_valor"]
col_b = col_a + ["min_valor", "mediana_valor", "acima_100"]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for nome, cols in [("(a) agregados do capítulo", col_a), ("(b) mais mínimo, mediana e acima de 100", col_b)]:
    auc = cross_val_score(GradientBoostingClassifier(random_state=0), pd.concat([base, dados[cols]], axis=1), y, cv=cv, scoring="roc_auc").mean()
    print(f"{nome:42s} AUC = {auc:.3f}")
