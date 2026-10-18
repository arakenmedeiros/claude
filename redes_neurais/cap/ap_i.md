# Apêndice I — Correspondência MATLAB ↔ Python

:::atencao Nota de verificação
A tabela abaixo resume correspondências **conceituais**. Os valores padrão e o comportamento exato das funções do MATLAB **não foram verificados** diretamente na documentação para esta edição; confira-os na documentação da sua versão (Deep Learning Toolbox).
:::

| MATLAB | Função | Equivalente neste livro / Python |
|---|---|---|
| `feedforwardnet(h)` | cria uma rede de alimentação direta (padrão: uma camada oculta de 10 neurônios e `trainlm`) | `MLP([n0, h, m], [...])` do `rnbook`; `MLPRegressor(hidden_layer_sizes=(h,))` |
| `train(net, X, T)` | treina a rede (no MATLAB, as amostras ficam nas **colunas**) | `rnbook.otimizadores.*` ou `.fit(X, y)` do scikit-learn (amostras nas **linhas**) |
| `trainlm` | Levenberg–Marquardt; parâmetros `mu`, `mu_dec`, `mu_inc`, `mu_max` (valores usuais: $10^{-3}$, $0{,}1$, $10$, $10^{10}$) | `levenberg_marquardt`; `scipy.optimize.least_squares(method="lm")` |
| `trainrp` | RPROP; `delt_inc`, `delt_dec`, `delta0`, `deltamax` (valores usuais: $1{,}2$; $0{,}5$; $0{,}07$; $50$) | `rprop(variante=...)` |
| `traingdx` | gradiente descendente com momento e taxa global adaptativa | `gradiente_descendente(taxa, momento)`; não há equivalente direto de DBD |
| `mse`, `sse` | funções de desempenho | `perda` da `MLP` com $c=1/(Nm)$ ou $c=\tfrac12$ |
| `dividerand` | divisão aleatória treino/validação/teste | `sklearn.model_selection.train_test_split` (duas chamadas) |
| `mapminmax`, `mapstd` | normalização das entradas | `MinMaxScaler`, `StandardScaler` (ajustados **só no treino**) |

## I.1 Diferenças práticas importantes

1. **Orientação dos dados.** No MATLAB, cada **coluna** é uma amostra; no Python (e neste livro), cada **linha**.
2. **Pré-processamento e divisão.** O MATLAB aplica, por padrão, normalização e divisão aleatória dentro do `train`; em Python, ambas devem ser feitas explicitamente (e sem vazamento).
3. **Perda.** O desempenho padrão do MATLAB é o erro quadrático médio; o `loss_` do scikit-learn para regressão é $E_{\text{sse}}/(N\cdot m)$, com o fator $\tfrac12$ do erro quadrático (verificado no Capítulo 4).
4. **Resultados.** Mesmo com os mesmos hiperparâmetros, as duas plataformas **não** dão resultados idênticos (inicialização, pré-processamento, critérios de parada).
