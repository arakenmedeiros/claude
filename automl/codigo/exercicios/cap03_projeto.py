# Capítulo 3 — solução de referência (PROJ)
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

dados = load_breast_cancer(as_frame=True)
df, y = dados.data.copy(), dados.target
rng = np.random.default_rng(0)

# 1) Sujar os dados de propósito
df["tamanho"] = pd.qcut(df["mean radius"], 3, labels=["pequeno", "médio", "grande"]).astype(str)
df["textura"] = pd.qcut(df["mean texture"], 3, labels=["lisa", "média", "áspera"]).astype(str)
df = df.drop(columns=["mean radius", "mean texture"])
num = [c for c in df.columns if c not in ("tamanho", "textura")]
cat = ["tamanho", "textura"]
for c in num:
    df.loc[rng.random(len(df)) < 0.10, c] = np.nan
print("Formato:", df.shape, "| ausentes (numéricas):", round(df[num].isna().mean().mean(), 3))

# 2) Teste final
Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.25, stratify=y, random_state=0)

# 3) Pré-processamento por tipo de coluna
def preparo(estrategia="median"):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy=estrategia)), ("esc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

# 4) Comparar modelos
cv = StratifiedKFold(5, shuffle=True, random_state=0)
modelos = {"logística": LogisticRegression(max_iter=3000), "k-vizinhos": KNeighborsClassifier(),
           "floresta": RandomForestClassifier(200, random_state=0)}
res = {}
for nome, m in modelos.items():
    res[nome] = cross_val_score(Pipeline([("p", preparo()), ("m", m)]), Xtr, ytr, cv=cv).mean()
    print(f"{nome:11s} validação = {res[nome]:.3f}")
melhor = max(res, key=res.get); print("Escolhido:", melhor)

# 5) Busca DENTRO do pipeline (imputação + hiperparâmetro do modelo)
pipe = Pipeline([("p", preparo()), ("m", modelos[melhor])])
grade = {"p__num__imp__strategy": ["mean", "median"]}
if melhor == "logística": grade["m__C"] = [0.01, 0.1, 1, 10]
elif melhor == "k-vizinhos": grade["m__n_neighbors"] = [3, 5, 9, 15]
else: grade["m__max_depth"] = [4, 8, None]
busca = GridSearchCV(pipe, grade, cv=cv).fit(Xtr, ytr)
print("Melhores ajustes:", busca.best_params_, "| validação =", round(busca.best_score_, 3))

# 6) Teste final (uma vez), salvar, recarregar e prever com categoria nova
print("Teste final:", round(busca.score(Xte, yte), 3))
joblib.dump(busca.best_estimator_, "/tmp/pipeline_cap3.joblib")
rec = joblib.load("/tmp/pipeline_cap3.joblib")
novas = Xte.iloc[:3].copy(); novas.iloc[2, novas.columns.get_loc("tamanho")] = "gigante"
print("Previsões em linhas novas:", rec.predict(novas).tolist(), "(a 3ª tem categoria nunca vista)")
