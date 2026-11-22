# Capítulo 2 — Aprendizado de máquina em um capítulo
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

# 1) Separa o TESTE (cofre): 20% dos dados
X_resto, X_teste, y_resto, y_teste = train_test_split(X, y, test_size=0.20, stratify=y, random_state=0)
# 2) Do que sobrou, separa a VALIDAÇÃO: 25% do resto (= 20% do total)
X_treino, X_valid, y_treino, y_valid = train_test_split(X_resto, y_resto, test_size=0.25, stratify=y_resto, random_state=0)

print("treino:", len(X_treino), "| validação:", len(X_valid), "| teste:", len(X_teste))
