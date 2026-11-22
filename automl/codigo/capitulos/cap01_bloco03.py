# Capítulo 1 — Bem-vindo ao AutoML
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

X, y = load_wine(return_X_y=True)

candidatos = {
    "Árvore de decisão": DecisionTreeClassifier(random_state=0),
    "k-vizinhos":        make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Regressão logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
}

for nome, modelo in candidatos.items():
    notas = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
    print(f"{nome:20s} acurácia média = {notas.mean():.3f}  (±{notas.std():.3f})")
