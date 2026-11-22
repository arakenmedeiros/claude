# Capítulo 10 — Ferramentas de AutoML na prática
# Bloco 7. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# ATENÇÃO: este trecho NÃO foi executado na elaboração do livro. Fixe versões e confira a documentação.
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label="y").fit(df_treino, time_limit=300)
print(predictor.leaderboard(df_teste))
predicoes = predictor.predict(df_teste)
