# Capítulo 3 — solução de referência (EX 5)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

rng = np.random.default_rng(5); n = 2500
x = rng.normal(size=(n, 3)); y = ((x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 1.5, n)) > 0).astype(int)
df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "bairro": rng.integers(0, 400, n)})
dtr, dte, ytr, yte = train_test_split(df, y, test_size=0.3, random_state=0, stratify=y)
NUM = ["x1", "x2", "x3"]; hgb = lambda: HistGradientBoostingClassifier(random_state=0)
m = hgb().fit(dtr[NUM], ytr); print(f"(a) sem bairro       : treino = {m.score(dtr[NUM], ytr):.3f} | teste = {m.score(dte[NUM], yte):.3f}")
media = pd.Series(ytr).groupby(dtr["bairro"].values).mean()
ing = lambda d: d[NUM].assign(te=d["bairro"].map(media).fillna(ytr.mean()).values)
m = hgb().fit(ing(dtr), ytr); print(f"(b) TE ingênuo       : treino = {m.score(ing(dtr), ytr):.3f} | teste = {m.score(ing(dte), yte):.3f}")
te = ColumnTransformer([("b", TargetEncoder(random_state=0), ["bairro"])], remainder="passthrough")
m = make_pipeline(te, hgb()).fit(dtr, ytr); print(f"(c) TargetEncoder    : treino = {m.score(dtr, ytr):.3f} | teste = {m.score(dte, yte):.3f}")
