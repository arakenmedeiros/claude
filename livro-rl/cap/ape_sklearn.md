# Apêndice E — Introdução ao scikit-learn
O **scikit-learn** é a biblioteca de aprendizado de máquina clássico em Python (PEDREGOSA et al., 2011). Ele **não** tem algoritmos de AR, mas fornece as peças que usamos o tempo todo: modelos de regressão e classificação, pré-processamento, agrupamento, métricas e validação.

## E.1 A API: fit, predict, transform

Quase todo objeto do scikit-learn segue o mesmo padrão:

| Método | O que faz |
|---|---|
| `fit(X, y)` | **Aprende** com os dados (ajusta os parâmetros) |
| `predict(X)` | **Prevê** a saída para novos exemplos |
| `transform(X)` | **Transforma** os dados (pré-processamento) |
| `fit_transform(X)` | `fit` e `transform` de uma vez |
| `score(X, y)` | Mede o desempenho (acurácia, R²...) |
| `partial_fit(X, y)` | Aprende **aos poucos**, a cada lote (aprendizado online) |

`X` é uma matriz com **uma linha por exemplo** e **uma coluna por característica**. `y` é o alvo.

```python
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
```

## E.2 Pré-processamento

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
import pandas as pd

Xnum = np.array([[1.0, 200.0], [2.0, 400.0], [3.0, 600.0]])
print(StandardScaler().fit_transform(Xnum).round(2))        # média 0, desvio 1

cores = pd.DataFrame({"cor": ["azul", "verde", "azul"]})
print(OneHotEncoder().fit_transform(cores).toarray())        # uma coluna por categoria

dados = np.array([[0.1], [0.4], [0.5], [0.9], [1.5], [2.0]])
kb = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile").fit(dados)
print(kb.transform(dados).ravel(), [b.round(2) for b in kb.bin_edges_])   # faixas de mesma frequência
```

:::dica Padronizar é quase sempre uma boa ideia
Redes neurais, RBF, K-Means e modelos lineares regularizados funcionam melhor quando as características têm escalas parecidas. Ajuste o `StandardScaler` **só nos dados de treino** e aplique no teste.
:::

## E.3 Pipelines

Um **pipeline** encadeia etapas (pré-processamento + modelo), evitando vazamento de dados e simplificando o código.

```python
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

modelo = make_pipeline(StandardScaler(), RBFSampler(gamma=0.5, n_components=50, random_state=0), Ridge(alpha=1.0))
modelo.fit(X_tr, y_tr)
print("R² do pipeline no teste:", round(modelo.score(X_te, y_te), 3))
```

## E.4 Modelos que usamos no livro

| Modelo | Uso no livro |
|---|---|
| `DecisionTreeClassifier` | Imitar episódios de elite (Cap. 1) e interpretar políticas (Cap. 5) |
| `LogisticRegression` | Probabilidade de cogumelo venenoso (Cap. 2) |
| `KMeans` | Criar estados de clientes (Cap. 3) |
| `SGDRegressor` / `SGDClassifier` | Aprendizado online com `partial_fit` |
| `ExtraTreesRegressor` | Aproximar funções q no Fitted Q-Iteration (Cap. 7) |
| `RBFSampler` + `Ridge` | Aproximação com características RBF |
| `MLPRegressor` / `MLPClassifier` | Redes neurais simples |
| `HistGradientBoostingRegressor` | Aproximar valores de clientes (Cap. 7) |
| `PoissonRegressor` | Estimar demanda (Cap. 6) |

```python
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
```

## E.5 Avaliação e incerteza

```python
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
```

## E.6 O que o scikit-learn não faz

- **Não** treina redes neurais profundas com GPU (use PyTorch, Apêndice G).
- **Não** tem ambientes nem algoritmos de AR (use Gymnasium e Stable-Baselines3, Apêndices F e H).
- **Não** tem validação cruzada para séries temporais de AR "pronta": cuide para que dados de treino e teste não se misturem.
