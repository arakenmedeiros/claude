# Apêndice C — Guia rápido
## C.1 Qual método de busca usar?

Use esta tabela como ponto de partida. Ela resume o que aprendemos; o ideal é sempre **comparar com busca aleatória**.

| Situação | Opção sugerida | Capítulo |
|---|---|---|
| Poucos hiperparâmetros (até 2) e treino barato | Busca em grade | 4 |
| Vários hiperparâmetros, orçamento limitado | Busca aleatória (linha de base forte) | 4 |
| Treino caro, poucas tentativas possíveis | Otimização bayesiana (TPE, SMAC, Optuna) | 5 |
| Treino com muitas épocas ou muitos dados | Successive Halving, Hyperband ou BOHB (multifidelidade) | 6 |
| Espaço grande, discreto, com estrutura de pipeline | Evolução (algoritmos genéticos, TPOT) | 7 |
| Muitos atributos, poucos exemplos | Seleção de atributos dentro do pipeline | 8 |
| Já se resolveram problemas parecidos | Meta-aprendizado e warm start | 9 |
| Máxima precisão e há tempo e memória | Ensembles (média, stacking, seleção de Caruana) | 9 |
| Quer um modelo pronto rapidamente | Ferramenta de AutoML (FLAML, Auto-sklearn, AutoGluon, H2O) | 10 |
| Rede neural com arquitetura a definir | NAS, com espaço pequeno e linha de base aleatória | 11 |
| Dados em série temporal | Validação temporal, atrasos, diferenciação | 12 |

## C.2 Receitas de código

Os trechos abaixo reúnem os padrões usados no livro. Eles vêm dos capítulos e das soluções, mas **as interfaces das bibliotecas mudam**: confira a documentação da sua versão.

### Pipeline sem vazamento

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
print(cross_val_score(pipe, X, y, cv=5).mean())      # o scaler é ajustado só no treino de cada parte
```

### Busca aleatória

```python
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

busca = RandomizedSearchCV(SVC(), {"C": loguniform(1e-2, 1e3), "gamma": loguniform(1e-4, 1e0)},
                           n_iter=30, cv=5, random_state=0).fit(Xtr, ytr)
print(busca.best_params_, busca.best_score_)
```

### Validação aninhada (estimar o ganho da busca)

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
interna = GridSearchCV(SVC(), {"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1]}, cv=3)
print(cross_val_score(interna, X, y, cv=5).mean())    # a busca roda dentro de cada parte externa
```

### Successive Halving no scikit-learn

```python
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
h = HalvingRandomSearchCV(modelo, distribuicoes, factor=3, random_state=0).fit(Xtr, ytr)
```

### Otimização bayesiana (TPE) com Optuna

```python
import optuna
def objetivo(trial):
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    return cross_val_score(SVC(C=C, gamma="scale"), Xtr, ytr, cv=3).mean()
estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
estudo.optimize(objetivo, n_trials=30)
estudo.enqueue_trial({"C": 1.0})                      # warm start: configuração conhecida na fila (antes do optimize)
```

### FLAML

```python
from flaml import AutoML
a = AutoML()
a.fit(Xtr, ytr, task="classification", time_budget=60, metric="accuracy", eval_method="cv", n_splits=3, seed=0)
print(a.best_estimator, a.best_config, a.best_loss)
```

### Ensemble por votação suave

```python
from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([("lr", lr), ("rf", rf), ("svm", svm)], voting="soft").fit(Xtr, ytr)
```

### Séries temporais: validação sem olhar o futuro

```python
from sklearn.model_selection import TimeSeriesSplit
for tr, va in TimeSeriesSplit(5, test_size=12).split(X):
    ...   # treina em X[tr] (passado) e valida em X[va] (futuro)
```

### Explicação e auditoria

```python
from sklearn.inspection import permutation_importance
pi = permutation_importance(modelo, Xte, yte, n_repeats=10, random_state=0)
# por grupo: taxa de seleção = pred[grupo == k].mean(); TPR = pred[(grupo == k) & (y_verdade == 1)].mean()
```

### Salvar e recarregar o pipeline

```python
import joblib
joblib.dump(pipe, "modelo.joblib")
pipe = joblib.load("modelo.joblib")                   # nunca carregue arquivos de origem desconhecida
```

## C.3 Fórmulas e métricas em palavras

| Conceito | Em palavras |
|---|---|
| Acurácia | acertos ÷ total de casos |
| Taxa de verdadeiros positivos (TPR) | entre os casos positivos reais, a proporção que o modelo acertou |
| MAE | média dos erros em valor absoluto |
| Erro padrão de uma proporção | raiz de p(1 − p) ÷ n; indica a incerteza da acurácia |
| Fronteira de Pareto | soluções sem outra melhor em todos os objetivos ao mesmo tempo |
| Correlação de Spearman | concordância entre as ordens de duas listas (1 = mesma ordem) |
| Posto médio | média das posições de um método em várias bases (1 = melhor) |
| PSI | soma, por faixa, de (proporção nova − proporção antiga) × log(proporção nova ÷ antiga) |
| Taxa de seleção | proporção de aprovados em um grupo |

## C.4 Lista de verificação de um projeto

**Antes de buscar**
- O problema e a métrica refletem o custo real dos erros?
- O **teste final** foi separado antes de qualquer decisão?
- Há risco de **vazamento** (pré-processamento fora da validação, dados repetidos, futuro no passado)?
- Existe uma **linha de base** simples?

**Durante a busca**
- O espaço é razoável, e o orçamento, definido?
- A validação respeita a estrutura dos dados (tempo, grupos)?
- Use **validação aninhada** (ou um teste intocado) para medir o ganho da busca.
- Registre semente, versões e orçamento.

**Depois da busca**
- Compare no teste uma única vez, com **incerteza** (erro padrão, várias divisões).
- O ensemble ou o AutoML ganharam da linha de base **além do ruído**?
- Explique o modelo (global e local) e meça o desempenho **por grupo**.
- Meça a **latência** e o custo; compare com o orçamento.

**Em produção**
- Salve o pipeline inteiro e a ficha do modelo.
- Monitore dados (KS, PSI) e desempenho (com rótulos recentes).
- Retreine com **campeão e desafiante**, com margem.
- Mantenha pessoas no circuito e canal de contestação.

## C.5 As dez armadilhas mais comuns

1. **Escolher pelo teste** (o teste vira validação).
2. **Vazamento** por pré-processar antes de dividir.
3. **Efeito do vencedor:** nota do melhor candidato é otimista.
4. **Comparar sem linha de base** ou com orçamentos desiguais.
5. **Tirar conclusões de uma única divisão** ou de poucos exemplos de teste.
6. **Embaralhar uma série temporal** na validação.
7. **Confiar em explicações como justificativas.**
8. **Achar que remover o atributo sensível elimina o viés** (proxies e rótulos enviesados).
9. **Ignorar o custo** (tempo, energia, latência).
10. **Esquecer o monitoramento** depois da implantação.

## C.6 Mapa dos capítulos

| Parte | Capítulos | Ideia central |
|---|---|---|
| I — Fundamentos | 1 a 3 | O problema CASH, a avaliação honesta e os pipelines sem vazamento |
| II — Otimização de Hiperparâmetros | 4 a 7 | Grade e aleatória; bayesiana; multifidelidade; evolução |
| III — Automação do Pipeline | 8 a 10 | Atributos; meta-aprendizado e ensembles; ferramentas |
| IV — Redes Neurais e Novos Domínios | 11 e 12 | NAS; séries temporais, texto e imagens |
| V — Responsabilidade e Prática | 13 e 14 | Explicabilidade, ética e limites; MLOps e projeto integrador |
