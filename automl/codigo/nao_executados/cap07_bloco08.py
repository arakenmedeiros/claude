# Capítulo 7 — Métodos evolutivos e programação genética
# Bloco 8. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# ATENÇÃO: este trecho NÃO foi executado na elaboração do livro. Fixe versões e confira a documentação.
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

X, y = load_digits(return_X_y=True)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

tpot = TPOTClassifier(generations=5,        # nº de gerações
                      population_size=20,   # nº de pipelines por geração
                      cv=5,                 # validação cruzada
                      random_state=0,
                      n_jobs=-1,
                      verbosity=2)
tpot.fit(X_treino, y_treino)                # evolui os pipelines (pode levar de minutos a horas)
print("Acurácia no teste:", tpot.score(X_teste, y_teste))
tpot.export("melhor_pipeline.py")           # grava o código Python do melhor pipeline
