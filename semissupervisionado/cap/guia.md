# Apêndice C — Guia rápido de classes, convenções e protocolo
## C.1 Convenções do scikit-learn para dados sem rótulo

| Item | Como funciona |
|---|---|
| Rótulo desconhecido | Valor -1 no vetor y. As classes verdadeiras devem ser 0, 1, 2… |
| X | Contém todos os exemplos, rotulados e sem rótulo |
| fit(X, y) | Aprende com X inteiro e o y com -1 nos exemplos sem rótulo |
| predict(X_novo) | Prevê as classes de exemplos novos (visão indutiva) |
| transduction_ | Rótulo atribuído a cada exemplo do treino (visão transdutiva) |
| score(X_teste, y_teste) | Acurácia em um conjunto de teste separado |

## C.2 Classes semissupervisionadas e parâmetros principais

| Classe | Módulo | Parâmetros-chave (padrão da versão 1.9) | Capítulo |
|---|---|---|---|
| SelfTrainingClassifier | sklearn.semi_supervised | estimator, threshold=0,75, criterion="threshold", k_best=10, max_iter=10 | 4 |
| LabelPropagation | sklearn.semi_supervised | kernel="rbf", gamma=20, n_neighbors=7, max_iter=1000 | 8 |
| LabelSpreading | sklearn.semi_supervised | kernel="rbf", gamma=20, n_neighbors=7, alpha=0,2, max_iter=30 | 8 |

Atributos úteis do SelfTrainingClassifier: `labeled_iter_` (em qual iteração cada exemplo ganhou rótulo; 0 = rótulo original; -1 = nunca rotulado), `n_iter_` (iterações realizadas) e `transduction_`.

O `estimator` do SelfTrainingClassifier precisa oferecer `predict_proba`. Um `SVC` só oferece se for criado com `probability=True`.

## C.3 Receita: comparar com o modelo de referência

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading

def esconder_rotulos(y, n_rot, semente):
    rng = np.random.RandomState(semente)
    classes = np.unique(y)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], n_rot // len(classes), replace=False) for c in classes])
    y_parcial = np.full(len(y), -1); y_parcial[idx] = y[idx]
    return y_parcial, idx

def comparar(X, y, n_rot=20, sementes=20):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    dif = []
    for s in range(sementes):
        yp, idx = esconder_rotulos(y_tr, n_rot, s)
        ref = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr[idx], y_tr[idx])
        ss = make_pipeline(StandardScaler(), LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2)).fit(X_tr, yp)
        dif.append(ss.score(X_te, y_te) - ref.score(X_te, y_te))
    dif = np.array(dif)
    return dif.mean(), int((dif > 0).sum()), int((dif < 0).sum())
```

## C.4 Lista de verificação de uma avaliação honesta

| Pergunta | Se a resposta for "não" |
|---|---|
| O teste foi separado antes de tudo e ficou fora do treino? | Risco de vazamento |
| Há um modelo de referência supervisionado bem ajustado? | O ganho pode ser exagerado |
| Os parâmetros foram escolhidos sem olhar o teste? | Resultado otimista |
| Houve vários sorteios, com média e desvio padrão? | Conclusão frágil |
| A comparação foi feita no mesmo sorteio? | Diferenças podem ser ruído |
| Os dados sem rótulo vêm da mesma população dos rotulados? | Risco de descasamento de classes |
| Os casos em que o método não ajudou foram relatados? | Viés de publicação |

## C.5 Escolhendo as primeiras ações

| Se… | Considere | Capítulo |
|---|---|---|
| Você precisa de um ponto de partida simples com qualquer classificador | Self-training | 4 |
| Os dados formam grupos ou curvas bem definidos | Label Spreading | 8 |
| Você não sabe se vale a pena | Diagnóstico de viabilidade (projeto do Capítulo 3) | 3 |
