# Apêndice I — Correspondência entre o pacote do livro, o scikit-learn e o SciPy

:::atencao Nota de verificação
As classes e parâmetros do scikit-learn citados foram conferidos na versão **1.9.1** (assinaturas e solvers disponíveis); o comportamento numérico não foi comparado, algoritmo a algoritmo, com o do pacote `rnbook`.
:::

| Pacote `rnbook` | Papel | Equivalente ou análogo em Python |
|---|---|---|
| `MLP([n0, h, m], ativacoes)` | rede de múltiplas camadas | `MLPClassifier` / `MLPRegressor` (`hidden_layer_sizes=(h,)`, `activation`) |
| `gradiente_descendente(taxa, momento)` | gradiente descendente com momento (lote) | `solver='sgd'` (`learning_rate_init`, `momentum`; minilotes por padrão) |
| `modernos.sgd_momento(nesterov=True)` | SGD com momento de Nesterov (minilote) | `solver='sgd'`, `nesterovs_momentum=True` |
| `modernos.adam` (`weight_decay=0`) | Adam (minilote) | `solver='adam'` (`beta_1`, `beta_2`, `epsilon`); `alpha` é penalidade L2 acoplada |
| `modernos.adam` (`weight_decay>0`) | AdamW (decaimento desacoplado) | sem equivalente no scikit-learn |
| `modernos.adagrad`, `modernos.rmsprop` | taxas por acumulação | sem equivalente no scikit-learn |
| `perdas.ce_logits`, `perdas.bce_logits` | perdas a partir de logits | `MLPClassifier` usa a perda logarítmica sobre probabilidades recortadas |
| `gd_estocastico` | gradiente por amostra ou minilote | `solver='sgd'` com `batch_size` |
| `quickprop` | Quickprop | sem equivalente no scikit-learn |
| `rprop(variante=...)` | RPROP e três variantes | sem equivalente no scikit-learn |
| `levenberg_marquardt` | Levenberg–Marquardt | `scipy.optimize.least_squares(method='lm')` (com a mesma função de resíduos e Jacobiana) |
| `delta_delta`, `delta_bar_delta`, `delta_bar_delta_estendido` | taxas por peso | sem equivalente; `learning_rate='adaptive'` adapta uma taxa **global** |
| `paciencia` (com `val=`) | parada antecipada pela perda de validação | `early_stopping=True`, `n_iter_no_change`, `validation_fraction` (monitora a pontuação de validação: acurácia no `MLPClassifier`, $R^2$ no `MLPRegressor`; só `sgd` e `adam`) |
| `perda` (com fator escolhido) | erro quadrático | `MLPRegressor(loss='squared_error')`: `loss_` $=E_{\text{sse}}/(Nm)$ mais a penalidade $\alpha$; ver o item 3 abaixo para outras perdas |
| `grad_numerico` | verificação do gradiente | `scipy.optimize.approx_fprime` |
| (fora do `rnbook`) | divisão e pré-processamento | `train_test_split`, `StandardScaler`, `MinMaxScaler` (ajustados **só no treino**) |
| (fora do `rnbook`) | métricas | `accuracy_score`, `r2_score`, `mean_squared_error` |

## I.1 Solvers do scikit-learn e o que os distingue

- **`lbfgs`**: quase-Newton de memória limitada, em lote; é o mais indicado para bases pequenas e é a referência externa usada nos Capítulos 6 e 7.
- **`sgd`**: gradiente descendente em minilotes, com `momentum` (e `nesterovs_momentum`) e três políticas de taxa: `'constant'`, `'invscaling'` e `'adaptive'`.
- **`adam`**: taxas individuais por normalização com estimativas de momentos do gradiente, em minilotes; **não** é avaliado neste livro.

## I.2 Diferenças práticas importantes

1. **Convenção de dados.** Amostras nas **linhas** (`X` com forma $(N,n_0)$); `coefs_[l]` tem forma $(n_{l-1},n_l)$, a transposta de $W^{(l)}$.
2. **Pré-processamento e divisão.** O scikit-learn não padroniza nem divide os dados por conta própria (a divisão só ocorre, para parada antecipada, com `early_stopping=True`): faça ambos explicitamente e sem vazamento.
3. **Perda.** A equivalência $\texttt{loss\_}=E_{\text{sse}}/(Nm)$ (mais a penalidade) vale **somente** para `MLPRegressor` com `loss='squared_error'` e $E_{\text{sse}}=\tfrac12\sum r^2$ (verificado no Capítulo 4). O `MLPClassifier` usa a **perda logarítmica** (entropia cruzada média, com soma sobre as classes), e o `MLPRegressor(loss='poisson')` usa **metade da deviance de Poisson**, com saída exponencial. Converta antes de comparar valores de erro.
4. **Lote e minilote.** Os algoritmos dos Capítulos 5 a 8 são de **lote inteiro**; `sgd` e `adam` usam minilotes por padrão (`batch_size='auto'`).
5. **Resultados.** Mesmo com hiperparâmetros parecidos, implementações diferentes não dão resultados idênticos (inicialização, critérios de parada, números de iterações).
