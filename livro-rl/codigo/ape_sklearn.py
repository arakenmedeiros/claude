"""
Livro: Aprendizado por Reforço — Apêndice E — Introdução ao scikit-learn
Autor do livro: Araken de Medeiros Santos

Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).
Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.
"""
import warnings; warnings.filterwarnings('ignore')

# ======================================================================
# Bloco 1
# ======================================================================
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, (200, 1))
y = 3 * X[:, 0] + 2 + rng.normal(0, 1, 200)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

reg = LinearRegression().fit(X_tr, y_tr)
print("coef =", reg.coef_.round(2), "intercepto =", round(reg.intercept_, 2), "R² teste =", round(reg.score(X_te, y_te), 3))
print("previsão para x = 4:", reg.predict([[4.0]]).round(2))

# ======================================================================
# Bloco 2
# ======================================================================
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
import pandas as pd

Xnum = np.array([[1.0, 200.0], [2.0, 400.0], [3.0, 600.0]])
print(StandardScaler().fit_transform(Xnum).round(2))        # média 0, desvio 1

cores = pd.DataFrame({"cor": ["azul", "verde", "azul"]})
print(OneHotEncoder().fit_transform(cores).toarray())        # uma coluna por categoria

dados = np.array([[0.1], [0.4], [0.5], [0.9], [1.5], [2.0]])
kb = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile").fit(dados)
print(kb.transform(dados).ravel(), [b.round(2) for b in kb.bin_edges_])   # faixas de mesma frequência

# ======================================================================
# Bloco 3
# ======================================================================
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

modelo = make_pipeline(StandardScaler(), RBFSampler(gamma=0.5, n_components=50, random_state=0), Ridge(alpha=1.0))
modelo.fit(X_tr, y_tr)
print("R² do pipeline no teste:", round(modelo.score(X_te, y_te), 3))

# ======================================================================
# Bloco 4
# ======================================================================
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor

arv = DecisionTreeClassifier(max_depth=2, random_state=0).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
print(export_text(arv, feature_names=["x"]))

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(rng.normal(0, 1, (100, 2)) + np.repeat([[0, 0], [5, 5]], 50, axis=0))
print("centros:", km.cluster_centers_.round(1).tolist())

rede = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=800, random_state=0).fit(X_tr, y_tr)
print("MLP R² teste:", round(rede.score(X_te, y_te), 3))

sgd = SGDRegressor(learning_rate="constant", eta0=0.01, random_state=0)
for _ in range(200): sgd.partial_fit(X_tr / 10, y_tr)                   # aprendizado online
print("SGD coef:", sgd.coef_.round(2))

# ======================================================================
# Bloco 5
# ======================================================================
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from sklearn.utils import resample
from sklearn.model_selection import GroupKFold, ParameterGrid

print("MSE:", round(mean_squared_error(y_te, reg.predict(X_te)), 3))

# intervalo de confiança por bootstrap
amostra = rng.normal(100, 20, 300)
boot = [resample(amostra, random_state=b).mean() for b in range(500)]
print("média =", amostra.mean().round(1), "IC 95% =", np.percentile(boot, [2.5, 97.5]).round(1))

# validação por grupos (todas as linhas de um grupo ficam juntas)
grupos = np.repeat(np.arange(10), 20)
treino, teste = next(GroupKFold(n_splits=5).split(X, y, groups=grupos))
print("treino:", len(treino), "teste:", len(teste))

# todas as combinações de hiperparâmetros
for p in ParameterGrid({"alpha": [0.01, 0.1], "eps": [0.05, 0.2]}): print(p)

