# Capítulo 13 — Explicabilidade, ética e limitações do AutoML
# Bloco 2. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
from sklearn.model_selection import train_test_split

r = np.random.default_rng(0); n = 6000
g = (r.random(n) < 0.3).astype(int)                          # 1 = grupo minoritário
z = r.normal(0, 1, n)                                        # capacidade real de pagamento (escondida)
renda = np.exp(0.5 * z + 0.35 * r.normal(size=n) - 0.25 * g) * 3.0
emprego = np.clip(5 + 2 * z + r.normal(0, 2, n), 0, None)
divida = np.clip(0.4 - 0.1 * z + r.normal(0, 0.1, n), 0, 1)
cep = 0.8 * g + 0.6 * r.normal(size=n)                       # variável proxy do grupo
score = z + r.normal(0, 1, n)
y_real = (z > 0).astype(int)                                 # quem de fato pagaria (só existe na simulação)
y_hist = ((z - 0.6 * g + r.normal(0, 0.5, n)) > 0).astype(int)   # decisões históricas, enviesadas contra o grupo 1
NOMES = ["renda", "tempo_emprego", "razao_divida", "cep_score", "score_historico"]
X = np.c_[renda, emprego, divida, cep, score]
Xa, Xte, ya, yte, ga, gte, yra, yrte = train_test_split(X, y_hist, g, y_real, test_size=0.3, random_state=0, stratify=y_hist)   # teste: 30%
Xtr, Xva, ytr, yva, gtr, gva, yrtr, yrva = train_test_split(Xa, ya, ga, yra, test_size=0.25, random_state=1, stratify=ya)       # treino e validação
print("Treino: %d | validação (calibração): %d | teste: %d" % (len(ytr), len(yva), len(yte)))

print("Taxa de aprovação NOS DADOS HISTÓRICOS  : grupo 0 = %.2f | grupo 1 = %.2f" % (ytr[gtr == 0].mean(), ytr[gtr == 1].mean()))
print("Taxa de bons pagadores (verdade, só aqui): grupo 0 = %.2f | grupo 1 = %.2f" % (yrtr[gtr == 0].mean(), yrtr[gtr == 1].mean()))
