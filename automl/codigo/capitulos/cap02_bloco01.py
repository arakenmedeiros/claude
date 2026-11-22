# Capítulo 2 — Aprendizado de máquina em um capítulo
# Bloco 1. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

dados = load_breast_cancer()
X, y = dados.data, dados.target

# Hiperparâmetro: escolhido por nós ANTES do treino
arvore = DecisionTreeClassifier(max_depth=3, random_state=0)
arvore.fit(X, y)

# Parâmetros: APRENDIDOS pelo modelo durante o treino
raiz = arvore.tree_.feature[0]
print("Hiperparâmetro max_depth =", arvore.get_params()["max_depth"])
print("Nº de nós aprendidos     =", arvore.tree_.node_count)
print("Pergunta da raiz         =", dados.feature_names[raiz], "<=", round(arvore.tree_.threshold[0], 2))
