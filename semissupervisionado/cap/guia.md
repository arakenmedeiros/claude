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

## C.3 Os módulos de PyTorch do livro (Capítulos 9 a 13 e 18)

Salve estes arquivos na mesma pasta dos seus scripts. Todos foram usados nos experimentos do livro.

| Arquivo | O que oferece | Capítulos |
|---|---|---|
| `utilidades.py` | `carregar_mnist()`, `carregar_fashion()`, `sortear_rotulados`, a `CNN`, `aumentar`, `aumento_fraco`, `aumento_forte`, `treinar_supervisionado` | 9 a 18 |
| `ssl_profundo.py` | `treinar_ssl(metodo, ...)` com os métodos `'sup'`, `'pi'`, `'mt'`, `'uda'`, `'pseudo'`, `'fixmatch'` e `'mixmatch'` | 10, 11, 15, 18 |
| `contrastivo.py` | `ntxent`, `pre_treinar` (SimCLR e rotação), `sonda_linear`, `ajuste_fino`, `carregar_ou_treinar` | 12, 18 |
| `generativos.py` | `VAE`, `treinar_vae`, `treinar_m2`, `treinar_sgan`, `treinar_mlp` | 13 |

Parâmetros mais úteis de `treinar_ssl`: `passos` (orçamento de treino), `peso` (peso da perda não supervisionada), `limiar` (confiança do FixMatch e da UDA), `ema` (fator da média móvel), `forte` e `fraco` (funções de aumento), `yu` (rótulos verdadeiros dos não rotulados, **só para diagnóstico**) e `relatar` (passos em que se mede o teste).

## C.4 Receita: comparar com o modelo de referência

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

## C.5 Lista de verificação de uma avaliação honesta

| Pergunta | Se a resposta for "não" |
|---|---|
| O teste foi separado antes de tudo e ficou fora do treino? | Risco de vazamento |
| Há um modelo de referência supervisionado bem ajustado? | O ganho pode ser exagerado |
| Os parâmetros foram escolhidos sem olhar o teste? | Resultado otimista |
| Houve vários sorteios, com média e desvio padrão? | Conclusão frágil |
| A comparação foi feita no mesmo sorteio? | Diferenças podem ser ruído |
| Os dados sem rótulo vêm da mesma população dos rotulados? | Risco de descasamento de classes |
| Os casos em que o método não ajudou foram relatados? | Viés de publicação |
| O ganho tem intervalo de confiança que exclui o zero, em sorteios pareados? | O ganho pode ser ruído |
| As classes são desbalanceadas, e a métrica reflete isso? | A acurácia engana |
| Em séries, a divisão respeitou o tempo? | Vazamento por janelas sobrepostas |
| O desempenho foi medido por subgrupo (classe, grupo)? | A média pode esconder falhas |

## C.6 Escolhendo as primeiras ações

| Se… | Considere | Capítulo |
|---|---|---|
| Você precisa de um ponto de partida simples com qualquer classificador | Self-training | 4 |
| Os dados formam grupos ou curvas bem definidos | Label Spreading | 8 |
| Você não sabe se vale a pena | Diagnóstico de viabilidade (projeto do Capítulo 3) | 3 |
| Os dados sem rótulo são imagens e há bons aumentos | UDA ou FixMatch, contra uma CNN com aumento como referência | 10, 11 |
| Há muito dado sem rótulo e tempo de computação | Pré-treino auto-supervisionado (SimCLR) e sonda linear | 12 |
| Você tem texto | Representação aprendida com o texto sem rótulo (LSA ou modelo de linguagem) + modelo linear | 14 |
| Você tem tabelas ou séries | Referência supervisionada forte; teste com cautela; divisão no tempo nas séries | 16 |
| Você precisa comparar métodos | Sorteios pareados, intervalo de confiança, validação nos rotulados, métricas por classe | 17 |
| Você vai entregar um projeto | O roteiro de dez passos e o modelo de relatório | 17, 18 |
