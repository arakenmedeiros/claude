# Capítulo 4 — Busca em grade e aleatória
# Bloco 5. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split

X0, y0 = load_breast_cancer(return_X_y=True)
grade = {"svc__C": np.logspace(-2, 3, 8), "svc__gamma": np.logspace(-5, 0, 8)}   # 64 combinações

comum, aninhada = [], []
for s in range(6):
    X, _, y, _ = train_test_split(X0, y0, train_size=80, stratify=y0, random_state=s)
    pipe = make_pipeline(StandardScaler(), SVC())
    interno = StratifiedKFold(5, shuffle=True, random_state=s)
    externo = StratifiedKFold(5, shuffle=True, random_state=100 + s)
    comum.append(GridSearchCV(pipe, grade, cv=interno, n_jobs=-1).fit(X, y).best_score_)
    aninhada.append(cross_val_score(GridSearchCV(pipe, grade, cv=interno, n_jobs=-1), X, y, cv=externo).mean())

print(f"Nota da busca (best_score_)  : {np.mean(comum):.3f}")
print(f"Nota aninhada (honesta)      : {np.mean(aninhada):.3f}")
print(f"Otimismo médio da busca      : {np.mean(comum) - np.mean(aninhada):+.3f}")
