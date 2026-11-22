# Capítulo 3 — Pipelines e pré-processamento
# Bloco 4. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

r = np.random.default_rng(0); n = 3000
x = r.normal(size=(n, 3)); cidade = r.integers(0, 600, n)                  # 600 cidades, SEM relação com o alvo
y = ((x[:, 0] + 0.5 * x[:, 1] + r.normal(0, 1.5, n)) > 0).astype(int)
df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "cidade": cidade})
dtr, dte, ytr, yte = train_test_split(df, y, test_size=0.3, random_state=0, stratify=y)
dtr, dva, ytr, yva = train_test_split(dtr, ytr, test_size=0.3, random_state=1, stratify=ytr)
NUM = ["x1", "x2", "x3"]
hgb = lambda: HistGradientBoostingClassifier(random_state=0)

def linha(nome, m, ftr, fva, fte):
    print(f"{nome:<36s} {m.score(ftr, ytr):7.3f} {m.score(fva, yva):10.3f} {m.score(fte, yte):7.3f}")

print(f"{'codificação da cidade':<36s} {'treino':>7s} {'validação':>10s} {'teste':>7s}")
m = hgb().fit(dtr[NUM], ytr); linha("(a) sem a coluna cidade", m, dtr[NUM], dva[NUM], dte[NUM])

ohe = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["cidade"])], remainder="passthrough")
m = make_pipeline(ohe, hgb()).fit(dtr, ytr); linha("(b) one-hot (%d colunas)" % ohe.transform(dtr[:5]).shape[1], m, dtr, dva, dte)

media = pd.Series(ytr).groupby(dtr["cidade"].values).mean(); geral = ytr.mean()
ingenuo = lambda d: d[NUM].assign(cidade_te=d["cidade"].map(media).fillna(geral).values)       # média do alvo no PRÓPRIO treino
m = hgb().fit(ingenuo(dtr), ytr); linha("(c) target encoding ingênuo", m, ingenuo(dtr), ingenuo(dva), ingenuo(dte))

te = ColumnTransformer([("c", TargetEncoder(random_state=0), ["cidade"])], remainder="passthrough")   # com cross-fitting
m = make_pipeline(te, hgb()).fit(dtr, ytr); linha("(d) TargetEncoder (cross-fitting)", m, dtr, dva, dte)
novas = dte.assign(cidade=9999)                                                                  # cidade nunca vista
print(f"\n(d) com TODAS as cidades do teste desconhecidas: acurácia no teste = {m.score(novas, yte):.3f}")
