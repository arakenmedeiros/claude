# Apêndice C — Guia rápido de algoritmos e códigos
## C.1 Mapa de decisão

| Se você quer… | Considere | Capítulo |
|---|---|---|
| Treinar sem juntar dados horizontais | FedAvg | 3 |
| Lidar com dados muito diferentes entre clientes | SCAFFOLD, FedProx, FedNova, otimizadores no servidor | 4 |
| Reduzir o tráfego de subida | Quantização, top-k com memória de erro, mais épocas locais | 5 |
| Atender bem a cada cliente | Ajuste fino, Ditto, partição corpo e cabeça | 6 |
| Atender grupos com conceitos diferentes | IFCA, agrupamento de clientes | 6 |
| Entender o que pode vazar | Inversão de gradientes, inferência de pertencimento | 7 |
| Garantia formal de privacidade | Privacidade diferencial (corte e ruído) | 8 |
| Esconder atualizações individuais | Agregação segura, criptografia homomórfica, SMPC | 8 |
| Resistir a clientes maliciosos | Mediana, média aparada, Multi-Krum, corte de norma | 9 |
| Combinar colunas de diferentes donos | Aprendizado federado vertical | 10 |
| Poucos recursos no cliente, rede neural | Split learning | 10 |
| Não ter servidor | Aprendizado descentralizado (gossip) | 10 |
| Agrupar ou reduzir dimensões sem rótulos | K-means e PCA federados | 11 |
| Especializar modelos grandes | Adaptadores de baixo posto (LoRA) | 11 |

## C.2 Arquivos de código do livro

| Arquivo | Capítulo | O que contém |
|---|---|---|
| `fedlib.py` | 3 | Dados dos dígitos, partições (IID, shards, Dirichlet), treino local, FedAvg |
| `scaffold.py` | 4 | Algoritmo SCAFFOLD |
| `compressao.py` | 5 | Quantização, top-k, memória de erro e FedAvg com compressão |
| `personalizacao.py` | 6 | Teste local, Ditto, mudança de conceito e IFCA |
| `privacidade.py` | 8 | Contabilidade de Rényi e DP-FedAvg |
| `robustez.py` | 9 | Agregadores robustos, ataques e FedAvg com maliciosos |
| `arquiteturas.py` | 10 | VFL, split learning e consenso descentralizado |
| `fronteiras.py` | 11 | K-means e PCA federados e adaptadores de baixo posto |
| `projeto.py` | 12 | Motor do projeto final |

Os códigos completos de cada arquivo aparecem nos respectivos capítulos. A seguir, a base de todos eles.

## C.3 Código completo da biblioteca `fedlib.py`

```python
"""fedlib.py - biblioteca didática do livro 'Aprendizado Federado'.
Simula clientes e servidor em um único computador, usando NumPy e scikit-learn."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def carregar_digits(seed=0):
    """Dígitos manuscritos da UCI (via scikit-learn), padronizados, com coluna de viés."""
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    esc = StandardScaler().fit(Xtr)
    mais_um = lambda A: np.hstack([esc.transform(A), np.ones((len(A), 1))])
    return mais_um(Xtr), mais_um(Xte), ytr, yte


def particao_iid(y, k, seed=0):
    """Embaralha tudo e reparte em k clientes de tamanho parecido."""
    idx = np.random.default_rng(seed).permutation(len(y))
    return [np.array(p) for p in np.array_split(idx, k)]


def particao_shards(y, k, por_cliente=2, seed=0):
    """Ordena por classe, corta em k*por_cliente pedaços e sorteia pedaços para cada cliente."""
    rng = np.random.default_rng(seed)
    pedacos = np.array_split(np.argsort(y, kind="stable"), k * por_cliente)
    ordem = rng.permutation(len(pedacos))
    return [np.concatenate([pedacos[ordem[i * por_cliente + j]] for j in range(por_cliente)]) for i in range(k)]


def particao_dirichlet(y, k, alfa=0.5, seed=0):
    """Cada classe é repartida entre os clientes com proporções sorteadas de uma Dirichlet(alfa)."""
    rng = np.random.default_rng(seed)
    partes = [[] for _ in range(k)]
    for c in np.unique(y):
        ic = rng.permutation(np.where(y == c)[0])
        p = rng.dirichlet([alfa] * k)
        cortes = (np.cumsum(p) * len(ic)).astype(int)[:-1]
        for i, pedaco in enumerate(np.split(ic, cortes)):
            partes[i] += pedaco.tolist()
    return [np.array(p, dtype=int) for p in partes]


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def gradiente(W, X, y, mu=0.0, W_ref=None):
    """Gradiente da entropia cruzada (regressão logística multiclasse) + termo proximal opcional."""
    P = softmax(X @ W)
    P[np.arange(len(y)), y] -= 1.0
    g = X.T @ P / len(y)
    if mu > 0 and W_ref is not None:
        g = g + mu * (W - W_ref)
    return g


def treino_local(W, X, y, epocas=1, eta=0.1, lote=32, mu=0.0, seed=0):
    """Treino local do cliente: mini-batch SGD a partir do modelo global W."""
    rng = np.random.default_rng(seed)
    W0, W = W, W.copy()
    for _ in range(epocas):
        ordem = rng.permutation(len(y))
        for i in range(0, len(y), lote):
            b = ordem[i:i + lote]
            W -= eta * gradiente(W, X[b], y[b], mu, W0)
    return W


def acuracia(W, X, y):
    return float(np.mean(np.argmax(X @ W, axis=1) == y))


def perda(W, X, y):
    P = softmax(X @ W)
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))


def fedavg(partes, X, y, Xte, yte, rodadas=30, epocas=1, eta=0.1, lote=32,
           frac=1.0, mu=0.0, seed=0, W0=None, classes=10):
    """FedAvg: em cada rodada, clientes sorteados treinam localmente e o servidor faz a média ponderada."""
    rng = np.random.default_rng(seed)
    W = np.zeros((X.shape[1], classes)) if W0 is None else W0.copy()
    historico = []
    for r in range(rodadas):
        k = max(1, int(round(frac * len(partes))))
        escolhidos = rng.choice(len(partes), k, replace=False)
        locais, tamanhos = [], []
        for c in escolhidos:
            if len(partes[c]) == 0:
                continue
            locais.append(treino_local(W, X[partes[c]], y[partes[c]], epocas, eta, lote, mu, seed * 1000 + r * 50 + int(c)))
            tamanhos.append(len(partes[c]))
        p = np.array(tamanhos) / sum(tamanhos)
        W = sum(pi * Wi for pi, Wi in zip(p, locais))
        historico.append(acuracia(W, Xte, yte))
    return W, historico
```

## C.4 Modelo de código para um experimento federado

```python
import numpy as np
from fedlib import *

# 1) Dados e partição entre clientes
Xtr, Xte, ytr, yte = carregar_digits()
partes = particao_dirichlet(ytr, k=20, alfa=0.3, seed=0)      # clientes heterogêneos

# 2) Treino federado, com várias sementes
resultados = []
for semente in range(3):
    partes = particao_dirichlet(ytr, 20, 0.3, seed=semente)
    W, hist = fedavg(partes, Xtr, ytr, Xte, yte, rodadas=30, epocas=5, eta=0.1, frac=0.5, seed=semente)
    resultados.append(hist[-1])

# 3) Relato honesto: média e desvio, e não uma única execução
print(f"acurácia final: {np.mean(resultados):.3f} ± {np.std(resultados):.3f}")

# 4) Custo de comunicação
n_param = Xtr.shape[1] * 10
print("bytes por rodada:", 2 * 4 * n_param * int(0.5 * 20))
```

## C.5 Lista de verificação final

:::memorize Antes de apresentar seus resultados
- Os dados foram divididos entre clientes de forma realista (IID e não-IID)?
- Os resultados usam **várias sementes**, com média e desvio?
- A avaliação inclui **por cliente** (média, pior e percentil 10)?
- O **custo de comunicação** foi medido?
- O **modelo de ameaça** está explícito? Há ataque e defesa testados?
- As **limitações** estão escritas (dados simulados, modelo simples, ataque simples)?
- Nenhuma afirmação de privacidade sem **garantia formal** (ε, δ) ou sem ressalvas?
- A parte jurídica e ética foi consultada (LGPD, GDPR, RIPD)?
:::
