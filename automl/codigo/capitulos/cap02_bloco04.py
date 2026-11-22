# Capítulo 2 — Aprendizado de máquina em um capítulo
# Bloco 4. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
notas = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
print("Notas das 5 rodadas:", notas.round(3))
print(f"Média = {notas.mean():.3f} | desvio padrão = {notas.std():.3f}")
