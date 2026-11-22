# Capítulo 10 — Ferramentas de AutoML na prática
# Bloco 6. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# ATENÇÃO: este trecho NÃO foi executado na elaboração do livro. Fixe versões e confira a documentação.
import h2o
from h2o.automl import H2OAutoML

h2o.init()
treino = h2o.H2OFrame(df_treino)       # df_treino: DataFrame com a coluna-alvo "y"
treino["y"] = treino["y"].asfactor()
aml = H2OAutoML(max_runtime_secs=300, seed=1)
aml.train(y="y", training_frame=treino)
print(aml.leaderboard)                 # ranking dos modelos
