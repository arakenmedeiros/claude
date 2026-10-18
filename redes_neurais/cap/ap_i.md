# Apêndice I — Correspondência entre o pacote do livro, o scikit-learn e o SciPy

:::atencao Nota de verificação
As classes e parâmetros do scikit-learn citados foram conferidos na versão **1.9.1** (assinaturas e solvers disponíveis); o comportamento numérico não foi comparado, algoritmo a algoritmo, com o do pacote `rnbook`.
:::

| Pacote `rnbook` | Papel | Equivalente ou análogo em Python |
|---|---|---|
| `MLP([n0, h, m], ativacoes)` | rede de múltiplas camadas | `MLPClassifier` / `MLPRegressor` (`hidden_layer_sizes=(h,)`, `activation`) |
| `gradiente_descendente(taxa, momento)` | gradiente descendente com momento (lote) | `solver='sgd'` (`learning_rate_init`, `momentum`; minilotes por padrão) |
| `gd_estocastico` | gradiente por amostra ou minilote | `solver='sgd'` com `batch_size` |
| `quickprop` | Quickprop | sem equivalente no scikit-learn |
| `rprop(variante=...)` | RPROP e três variantes | sem equivalente no scikit-learn |
| `levenberg_marquardt` | Levenberg–Marquardt | `scipy.optimize.least_squares(method='lm')` (com a mesma função de resíduos e Jacobiana) |
| `delta_delta`, `delta_bar_delta`, `delta_bar_delta_estendido` | taxas por peso | sem equivalente; `learning_rate='adaptive'` adapta uma taxa **global** |
| `paciencia` (com `val=`) | parada antecipada | `early_stopping=True`, `n_iter_no_change`, `validation_fraction` |
| `perda` (com fator escolhido) | erro | `loss_` = $E_{\text{sse}}/(Nm)$ mais a penalidade $\alpha$ |
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
3. **Perda.** `loss_` equivale a $E_{\text{sse}}/(Nm)$ (mais penalidade) neste livro (verificado no Capítulo 4); converta antes de comparar valores de erro.
4. **Lote e minilote.** Os algoritmos dos Capítulos 5 a 8 são de **lote inteiro**; `sgd` e `adam` usam minilotes por padrão (`batch_size='auto'`).
5. **Resultados.** Mesmo com hiperparâmetros parecidos, implementações diferentes não dão resultados idênticos (inicialização, critérios de parada, números de iterações).
