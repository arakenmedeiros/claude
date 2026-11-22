# Capítulo 3 — Pipelines e pré-processamento
# Bloco 8. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# --- blocos anteriores deste capítulo (encadeados) ---
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

rng = np.random.default_rng(0)
df = load_wine(as_frame=True).frame.copy()

# Coluna categórica simulada ("safra"), levemente ligada à classe do vinho
safras = np.array(["2018", "2019", "2020"])
df["safra"] = np.where(rng.random(len(df)) < 0.8, safras[df["target"].values % 3], rng.choice(safras, len(df)))

# 8% de valores ausentes, ao acaso, em quatro colunas
for coluna in ["alcohol", "malic_acid", "flavanoids", "proline"]:
    df.loc[rng.random(len(df)) < 0.08, coluna] = np.nan

y = df.pop("target")
X = df
print("Formato de X:", X.shape)
print("\nValores ausentes por coluna:")
print(X.isna().sum()[X.isna().sum() > 0].to_string())
print("\nColunas numéricas:", X.select_dtypes("number").shape[1], "| colunas de texto:", X.select_dtypes(exclude="number").shape[1])
from sklearn.linear_model import LogisticRegression

try:
    LogisticRegression().fit(X, y)
except ValueError as erro:
    print("O scikit-learn recusou os dados:")
    print(str(erro).splitlines()[0][:110])
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

colunas_num = X.select_dtypes(include="number").columns.tolist()
colunas_cat = ["safra"]

preparo = ColumnTransformer([
    ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                      ("escala", StandardScaler())]), colunas_num),
    ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), colunas_cat),
])

pipeline = Pipeline([("preparo", preparo), ("modelo", LogisticRegression(max_iter=2000))])

notas = cross_val_score(pipeline, X, y, cv=5)
print("Notas:", notas.round(3), "| média =", round(notas.mean(), 3))
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler

busca_pipe = Pipeline([
    ("preparo", ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer()), ("escala", StandardScaler())]), colunas_num),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), colunas_cat),
    ])),
    ("modelo", LogisticRegression(max_iter=3000)),
])

grade = {
    "preparo__num__imputar__strategy": ["mean", "median"],
    "preparo__num__escala": [StandardScaler(), MinMaxScaler(), RobustScaler()],
    "modelo__C": [0.1, 1, 10],
}
busca = GridSearchCV(busca_pipe, grade, cv=5, n_jobs=-1).fit(X, y)

print("Combinações testadas:", len(busca.cv_results_["params"]))
print("Melhor nota de validação:", round(busca.best_score_, 3))
for nome, valor in busca.best_params_.items():
    print(f"  {nome} = {valor}")
# --- fim dos blocos anteriores ---
import joblib

melhor_pipeline = busca.best_estimator_            # já treinado com todos os dados
joblib.dump(melhor_pipeline, "/tmp/pipeline_vinho.joblib")

recarregado = joblib.load("/tmp/pipeline_vinho.joblib")
posicoes = [0, 30, 70, 100, 140, 170]              # vinhos de classes diferentes, com ausentes e texto
novos = X.iloc[posicoes]
print("Classes reais            :", y.iloc[posicoes].tolist())
print("Previsões do original   :", melhor_pipeline.predict(novos).tolist())
print("Previsões do recarregado:", recarregado.predict(novos).tolist())
