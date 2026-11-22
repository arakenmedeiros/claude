# Capítulo 10 — Ferramentas de AutoML na prática
# Bloco 5. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# ATENÇÃO: este trecho NÃO foi executado na elaboração do livro. Fixe versões e confira a documentação.
# TPOT série 0.x (histórico; por exemplo, 0.12.2). Não vale para a versão 1.0 em diante.
from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=5, population_size=20, cv=3,
                      random_state=0, verbosity=2)
tpot.fit(Xtr, ytr)
print(tpot.score(Xte, yte))
tpot.export("melhor_pipeline.py")     # gera o código do pipeline encontrado
