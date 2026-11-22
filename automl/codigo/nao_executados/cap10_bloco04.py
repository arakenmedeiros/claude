# Capítulo 10 — Ferramentas de AutoML na prática
# Bloco 4. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# ATENÇÃO: este trecho NÃO foi executado na elaboração do livro. Fixe versões e confira a documentação.
import autosklearn.classification
from sklearn.metrics import accuracy_score

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=300,      # orçamento total (segundos)
    per_run_time_limit=30,            # limite por modelo
    seed=0,
)
automl.fit(Xtr, ytr)
print(automl.leaderboard())           # tabela dos modelos do ensemble
print(accuracy_score(yte, automl.predict(Xte)))
