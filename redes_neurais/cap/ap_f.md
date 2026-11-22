# Apêndice F — Guia dos códigos

Todos os caminhos são relativos à pasta `codigo/`.

| Capítulo | Script | O que produz |
|---|---|---|
| 1 | `cap01/exp_primeira_rede.py`, `cap01/figs_cap01.py` | primeira rede treinada ("Olá, rede") e figuras conceituais |
| 2 | `cap02/cap02.py` | derivada, cadeia, gradiente, gradiente descendente, Newton, secantes, lógica e mínimos quadrados |
| 3 | `cap03/cap03.py` | experimentos e figuras do perceptron |
| 4 | `cap04/cap04a.py` a `cap04d.py` | exemplo 2–2–1, verificação de gradientes, escalar × vetorizado, XOR, aproximação universal, minilotes, correspondência com o scikit-learn, vazamento e figuras |
| 5 | `cap05/cap05.py` | experimentos e figuras do Quickprop |
| 6 | `cap06/cap06.py`, `cap06/cap06_1d.py` | RPROP e variantes, parada antecipada; rastreio do passo em uma variável |
| 7 | `cap07/cap07.py` | Newton, Gauss–Newton e Levenberg–Marquardt; comparação com o SciPy; custo |
| 8 | `cap08/cap08.py` | delta-delta, DBD e EDBD; quadráticas; rede 1–8–1; sensibilidade |
| 9 | `cap09/cap09.py` | otimizadores de minilote, AdamW, autodiferenciação em grafos, perdas estáveis, produtos Hessiana–vetor, LM com CG e diagnóstico |

## Pacote `rnbook`

| Arquivo | Conteúdo |
|---|---|
| `ativacoes.py` | sigmoide, tanh, ReLU, identidade, softmax e sigmoide de Fahlman (com derivada $+0{,}1$) |
| `rede.py` | classe `MLP(camadas, ativacoes, semente, init)`; métodos `forward`, `prever`, `residuos`, `perda`, `grad`, `jacobiana` |
| `backprop_escalar.py` | backpropagation de uma rede com um neurônio por camada (para estudo passo a passo) |
| `gradcheck.py` | gradiente numérico por diferença central e erro relativo |
| `perceptron.py` | `PerceptronDidatico` |
| `otimizadores.py` | `gradiente_descendente`, `gd_estocastico`, `quickprop`, `rprop` (quatro variantes), `levenberg_marquardt`, `levenberg_marquardt_nielsen`, `gauss_newton`, `delta_delta`, `delta_bar_delta`, `delta_bar_delta_estendido` |
| `modernos.py` | `sgd_momento` (clássico e Nesterov), `adagrad`, `rmsprop`, `adam` (com AdamW), `hvp_dif_finitas`, `jvp_dif_finitas`, `gradiente_conjugado`, `lm_sem_matriz`, `diagnostico` |
| `perdas.py` | `logsumexp`, softmax estável, entropia cruzada e perda binária a partir de logits (e versões ingênuas, para comparação) |
| `autodiff.py` | diferenciação automática reversa didática (`No`) e normalização de camada |
| `estilo.py`, `desenho.py` | estilo das figuras e desenho de diagramas de redes |

## Onde obter os arquivos

O código-fonte completo do pacote `rnbook` está impresso no **Apêndice L**. Os scripts de cada capítulo, os testes, o `requirements.txt`, os notebooks e o registro de pendências (`PENDENCIAS.md`) acompanham o livro como material complementar, na pasta `codigo/` (e `notebooks/`) do repositório que o distribui.

## Testes

`testes_rnbook.py` verifica: o gradiente analítico contra diferenças finitas (erro relativo $\approx10^{-10}$), a igualdade entre as implementações escalar e vetorizada, a Jacobiana, o comportamento dos otimizadores em problemas conhecidos e a concordância do LM com o `scipy.optimize.least_squares`.

:::atencao Implementações didáticas
Os algoritmos de `otimizadores.py` são **implementações didáticas do livro**: priorizam clareza e fidelidade às fórmulas do texto, não desempenho nem a reprodução exata dos artigos originais.
:::
