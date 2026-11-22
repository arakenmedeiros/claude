# Capítulo 3 — Pipelines e pré-processamento
# Bloco 6. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2000))     # atributos sem nenhuma relação com y
y = rng.integers(0, 2, 100)          # alvo sorteado

# ERRADO: seleciona atributos com TODOS os dados, depois valida
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
errado = cross_val_score(LogisticRegression(), X_sel, y, cv=5).mean()

# CERTO: a seleção está DENTRO do pipeline (refeita em cada divisão)
certo_pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
certo = cross_val_score(certo_pipe, X, y, cv=5).mean()

print(f"Seleção fora do pipeline (errado): acurácia = {errado:.3f}")
print(f"Seleção dentro do pipeline (certo): acurácia = {certo:.3f}")
