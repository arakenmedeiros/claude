# Capítulo 3 — Pipelines e pré-processamento
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)   # versão completa (sem ausentes) só para este teste
sem = cross_val_score(KNeighborsClassifier(), X, y, cv=5).mean()
com = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier()), X, y, cv=5).mean()
print(f"k-vizinhos sem padronizar: {sem:.3f}")
print(f"k-vizinhos padronizado   : {com:.3f}")
