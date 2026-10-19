# Apêndice G — Símbolos e notação

Este apêndice reúne a notação usada nos capítulos. Convenções gerais: letras minúsculas em negrito ($\mathbf{x}$) são vetores; maiúsculas em negrito ($\mathbf{W}$) são matrizes; letras sem negrito são números (escalares). O logaritmo $\ln$ é o natural, e $\log_2$ aparece apenas quando indicado. Alguns símbolos têm **mais de um sentido**, porque a literatura usa as mesmas letras para coisas diferentes; a última seção lista esses casos, e cada capítulo define o símbolo no ponto em que o usa.

## Convenções gerais (Capítulos 1 e 2)

| Símbolo | Significado | Onde |
| --- | --- | --- |
| $w_t$ | token na posição $t$ de uma sequência | Cap. 1 |
| $P(w_t \mid w_1,\ldots,w_{t-1})$ | probabilidade do token $w_t$ dados os anteriores | Caps. 1, 5 |
| $n$, $N$ | comprimento de sequência, número de exemplos ou de documentos (conforme o contexto) | Caps. 1 a 17 |
| $V$ | tamanho do vocabulário | Caps. 2, 3, 4 |
| $d$ | dimensão dos vetores (embeddings, modelo) | Caps. 2, 3, 4 |
| $\mathbf{a}\cdot\mathbf{b}$ | produto escalar | Cap. 2 |
| $\lVert\mathbf{a}\rVert$ | norma (comprimento) do vetor | Cap. 2 |
| $\cos(\mathbf{a},\mathbf{b})$ | similaridade de cosseno | Caps. 2, 3, 10 |
| $\mathbf{W}\in\mathbb{R}^{m\times n}$ | matriz real com $m$ linhas e $n$ colunas | Cap. 2 |
| $\mathbf{x}^{\top}$ | transposta | Caps. 2, 4 |
| $\mathbf{z}$ | vetor de logits | Cap. 2 |
| $p_i$ | probabilidade da classe ou do token $i$ | Cap. 2 |
| $\mathrm{softmax}(\mathbf{z})_i$ | $e^{z_i}/\sum_j e^{z_j}$ | Cap. 2 |
| $I(p)$ | surpresa (informação), $-\ln p$ | Cap. 2 |
| $H(\mathbf{p})$ | entropia | Cap. 2 |
| $H(\mathbf{p},\mathbf{q})$ | entropia cruzada | Cap. 2 |
| $D_{\mathrm{KL}}(\mathbf{p}\Vert\mathbf{q})$ | divergência de Kullback-Leibler | Caps. 2, 7 |
| $\mathcal{L}$ | função de perda | Caps. 2, 5, 7, 12, 13 |
| $\mathrm{PPL}$ | perplexidade, $e^{\mathcal{L}}$ média | Caps. 1, 2, 16 |
| $\theta$ | parâmetros do modelo | Caps. 2, 5, 7 |
| $\nabla_\theta\mathcal{L}$ | gradiente da perda | Cap. 2 |
| $\eta$ | taxa de aprendizado | Caps. 2, 5 |
| $\partial/\partial x$ | derivada parcial | Cap. 2 |

## Capítulo 3: texto e embeddings

| Símbolo | Significado | Onde |
| --- | --- | --- |
| $\mathrm{tf}(t,d)$ | frequência do termo $t$ no documento $d$ | Caps. 3, 10 |
| $\mathrm{df}(t)$ | número de documentos que contêm o termo $t$ | Caps. 3, 10 |
| $\mathrm{tfidf}(t,d)$ | $\mathrm{tf}\cdot(\ln(N/\mathrm{df})+1)$ | Cap. 3 |
| $\mathbf{E}\in\mathbb{R}^{V\times d}$ | tabela de embeddings; $\mathbf{e}_i$ é a linha $i$ | Cap. 3 |
| $\mathrm{PMI}(a,b)$, $\mathrm{PPMI}$ | informação mútua pontual e sua versão truncada em zero | Cap. 3 |

## Capítulo 4: atenção e Transformer

| Símbolo | Significado | Onde |
| --- | --- | --- |
| $\mathbf{X}$ | matriz de entrada, uma linha por posição | Cap. 4 |
| $\mathbf{Q},\mathbf{K},\mathbf{V}$ | consultas, chaves e valores, $\mathbf{X}\mathbf{W}_Q$, $\mathbf{X}\mathbf{W}_K$, $\mathbf{X}\mathbf{W}_V$ | Cap. 4 |
| $d_k$ | dimensão das chaves e consultas | Cap. 4 |
| $\mathbf{W}_O$ | projeção de saída da atenção multicabeça | Cap. 4 |
| $h$ | número de cabeças de atenção | Cap. 4 |
| $\mathrm{cab}_m$ | saída da cabeça $m$ | Cap. 4 |
| $\mathrm{PE}(p,i)$ | codificação posicional senoidal na posição $p$ | Cap. 4 |
| $F(\mathbf{x})$ | sub-bloco (atenção ou FFN) em uma conexão residual | Cap. 4 |
| $\mathrm{LN}(\mathbf{x})$ | normalização de camada | Cap. 4 |
| $\boldsymbol{\gamma},\boldsymbol{\beta}$ | escala e deslocamento aprendidos da normalização | Cap. 4 |
| $\mu,\sigma^2,\varepsilon$ | média, variância e constante de estabilidade | Cap. 4 |
| $\mathrm{FFN}$ | rede alimentada adiante, $\mathbf{W}_2\sigma(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2$ | Cap. 4 |
| $N_{\text{parâm}}$ | número de parâmetros, aproximadamente $12Nd^2+Vd$ (aqui $N$ é o número de blocos) | Cap. 4 |

## Capítulos 5 a 7: treinamento, geração e pós-treinamento

| Símbolo | Significado | Onde |
| --- | --- | --- |
| $p_\theta$ | distribuição do modelo com parâmetros $\theta$ | Caps. 5, 7 |
| $\eta_{\max},\eta_{\min},w,S$ | taxa máxima e mínima, passos de aquecimento e total na programação de taxa | Cap. 5 |
| $J(A,B)$ | similaridade de Jaccard | Cap. 5 |
| $C\approx 6ND$ | custo de treinamento em FLOPs; $N$ parâmetros, $D$ tokens | Cap. 5 |
| $T$ | temperatura em $p_i(T)=e^{z_i/T}/\sum_j e^{z_j/T}$; também número de tokens do contexto (Cap. 4: $\mathbf{X}\in\mathbb{R}^{T\times d_{\text{modelo}}}$; Cap. 6: fórmula do cache) | Caps. 4, 6, 15 |
| $L_{\text{entrada}},L_{\text{saída}},L_{\text{janela}}$ | tokens de entrada, de saída e janela de contexto | Cap. 6 |
| $t_{\text{prefill}},t_{\text{token}}$ | tempo de preenchimento e tempo por token gerado | Cap. 6 |
| $n_{\text{kv}},d_{\text{cab}}$ | número de cabeças de chaves e valores; dimensão de cada cabeça | Cap. 6 |
| $\mathcal{L}_{\text{SFT}}$, $m_t$ | perda do ajuste supervisionado e máscara (1 na resposta, 0 na instrução) | Cap. 7 |
| $x,y_w,y_l$ | instrução, resposta preferida e resposta rejeitada | Cap. 7 |
| $r(x,y)$ | recompensa | Cap. 7 |
| $\sigma(\cdot)$ | função sigmoide (no Cap. 4, também uma ativação genérica) | Caps. 4, 7 |
| $\pi_\theta,\pi_{\text{ref}}$ | política treinada e política de referência | Cap. 7 |
| $\beta$ | coeficiente da penalidade de KL no RLHF e no DPO | Cap. 7 |
| $Z(x)$ | constante de normalização da política ótima | Cap. 7 |
| $\mathcal{L}_{\text{RM}},\mathcal{L}_{\text{DPO}}$ | perdas do modelo de recompensa e do DPO | Cap. 7 |

## Capítulos 8 a 11: prompts, aplicações, busca e RAG

| Símbolo | Significado | Onde |
| --- | --- | --- |
| $\mathrm{BM25}(q,d)$ | pontuação lexical da consulta $q$ para o documento $d$ | Cap. 10 |
| $f_{t,d}$ | contagem do termo $t$ em $d$ | Cap. 10 |
| $k_1,b$ | parâmetros do BM25 (saturação e normalização por tamanho) | Cap. 10 |
| $\lvert d\rvert,\overline{\lvert d\rvert}$ | tamanho do documento e tamanho médio | Cap. 10 |
| $\mathrm{rel}_i$ | relevância do item na posição $i$ | Cap. 10 |
| $\mathrm{DCG@}k,\mathrm{nDCG@}k$ | ganho acumulado descontado e sua versão normalizada | Cap. 10 |
| $r_m(d)$, $\mathrm{pos}_L(d)$ | posição do documento $d$ no ranking $m$ (ou na lista $L$) | Caps. 10, 11 |
| $k$ (no RRF), $c$ | constante de suavização da fusão por posição (valor típico 60 nos exemplos) | Caps. 10, 11 |
| $k$ (em recall@$k$, top-$k$) | número de itens considerados | Caps. 6, 10, 11 |
| $M=N\,d\,b$ | memória de um índice: $N$ vetores, dimensão $d$, $b$ bytes por número | Cap. 10 |
| $\mathrm{cob}(q,s)$ | cobertura ponderada da pergunta pela frase | Cap. 11 |
| $R(\cdot)$, $\mathrm{idf}(r)$ | conjunto de radicais e raridade do radical | Cap. 11 |
| $\lambda$ | limiar de abstenção do gerador simulado | Cap. 11 |
| $T_{\text{ent}},T_{\text{sai}},p_{\text{ent}},p_{\text{sai}}$ | tokens e preços (por milhão) de entrada e de saída | Caps. 11, 15 |
| $C$ | custo de uma consulta | Cap. 11 |

## Capítulos 12 e 13: ajuste fino e multimodalidade

| Símbolo | Significado | Onde |
| --- | --- | --- |
| $\mathbf{W}$, $\Delta\mathbf{W}$ | peso congelado e atualização de baixo posto | Cap. 12 |
| $\mathbf{A},\mathbf{B}$ | fatores do LoRA, $\mathbf{B}\in\mathbb{R}^{d\times r}$, $\mathbf{A}\in\mathbb{R}^{r\times k}$ | Cap. 12 |
| $r$ | posto do LoRA | Cap. 12 |
| $\alpha$ | fator de escala do LoRA, aplicado como $\alpha/r$ | Cap. 12 |
| $r(d+k)/(dk)$ | fração de parâmetros treináveis de uma camada | Cap. 12 |
| $H,L,p$ | altura, largura da imagem e lado do patch | Cap. 13 |
| $N=(H/p)(L/p)$ | número de patches | Cap. 13 |
| $\mathbf{z}_k,\mathbf{pos}_k$ | embedding do patch $k$ e sua posição | Cap. 13 |
| $\mathbf{u}_i,\mathbf{t}_j$ | embeddings normalizados de imagem e de texto | Cap. 13 |
| $S_{ij}$ | similaridade dividida pela temperatura | Cap. 13 |
| $\tau$ | temperatura da perda contrastiva | Cap. 13 |
| $\mathbf{x}_t$, $\mathbf{x}_0$ | imagem com ruído no passo $t$ e imagem original | Cap. 13 |
| $\alpha_t,\bar\alpha_t$ | coeficiente de preservação do sinal no passo $t$ e seu produto acumulado | Cap. 13 |
| $\boldsymbol{\varepsilon}$ | ruído gaussiano | Cap. 13 |

## Capítulos 14 a 17: agentes, implantação, avaliação e segurança

| Símbolo | Significado | Onde |
| --- | --- | --- |
| $t_{\text{mem}},t_{\text{calc}},t_{\text{passo}}$ | tempos de memória, de cálculo e do passo de decodificação | Cap. 15 |
| $N,\beta,b,T,m_{\text{kv}}$ | parâmetros, bytes por parâmetro, sequências simultâneas (lote), tokens de contexto, bytes de cache por token | Cap. 15 |
| $B_W,P$ | largura de banda de memória e capacidade de cálculo do hardware (hipotéticas) | Cap. 15 |
| $q,s,q_{\max}$ | valor quantizado, escala e maior inteiro representável, $2^{b-1}-1$ | Cap. 15 |
| $\mathcal{N}$ | distribuição normal | Cap. 13 |
| $\tau$ (no cache semântico) | limiar de cosseno para reutilizar uma resposta | Cap. 15 |
| $\gamma$ (decodificação especulativa) | número de tokens propostos pelo rascunhador por rodada | Cap. 15 |
| $\alpha$ (decodificação especulativa) | taxa de aceitação dos tokens do rascunhador | Cap. 15 |
| $c$ (decodificação especulativa) | custo relativo do rascunhador | Cap. 15 |
| $G(\alpha,\gamma,c)$ | ganho de velocidade esperado | Cap. 15 |
| $\lambda,\mu,\rho,c,a$ | chegada, atendimento, utilização, servidores e carga oferecida em filas M/M/c | Cap. 15 |
| $C(c,a)$, $W_q$, $W$ | probabilidade de espera, espera na fila e tempo total | Cap. 15 |
| $\mathrm{PSI}$ | índice de estabilidade populacional, $\sum(a_i-e_i)\ln(a_i/e_i)$ | Cap. 15 |
| $e_i,a_i$ | proporções esperada (referência) e atual da categoria $i$ no PSI | Cap. 15 |
| $z$ | quantil da normal (1,96 para 95%) | Caps. 16, 17 |
| $s,e$ (juiz) | sensibilidade e especificidade do juiz automático, estimadas em amostra rotulada | Cap. 16 |
| $e$ (no tamanho de amostra) | margem de erro desejada | Cap. 16 |
| $\delta,\psi$ | diferença a detectar e proporção de casos discordantes | Cap. 16 |
| $\alpha$ e $1-\beta$ (testes) | nível de significância e poder | Cap. 16 |
| $P,R,F$ | precisão, revocação e F1 | Cap. 16 |
| $\kappa$, $p_o$, $p_e$ | kappa de Cohen, concordância observada e esperada | Cap. 16 |
| $\alpha$ (Krippendorff) | $1-D_o/D_e$ | Cap. 16 |
| $D_o,D_e$ | discordância observada e esperada | Cap. 16 |
| $b,c$ (McNemar) | casos em que só A acerta e casos em que só B acerta | Cap. 16 |
| $\chi^2$ | estatística do teste de McNemar com correção de continuidade | Cap. 16 |
| $\hat p$ | proporção observada | Caps. 16, 17 |
| $\Delta_{\mathrm{DP}},\Delta_{\mathrm{IO}}$ | diferença de paridade demográfica e de igualdade de oportunidades | Cap. 17 |
| $\hat Y,Y,G$ | decisão prevista, rótulo verdadeiro e atributo de grupo | Cap. 17 |
| $k$ (k-anonimato) | tamanho mínimo de um grupo de quase identificadores | Cap. 17 |

## Símbolos com mais de um sentido

Os símbolos abaixo aparecem com significados diferentes. Em cada capítulo o contexto e a definição local resolvem a ambiguidade; ao cruzar capítulos, confira esta tabela.

| Símbolo | Sentidos (capítulo) |
| --- | --- |
| $\alpha$ | fator de escala do LoRA (12); coeficiente de preservação do sinal na difusão (13); taxa de aceitação na decodificação especulativa (15); nível de significância (16); alfa de Krippendorff (16) |
| $\beta$ | deslocamento da normalização de camada (4); coeficiente da penalidade de KL (7); bytes por parâmetro (15); $1-\beta$ é o poder estatístico (16) |
| $\gamma$ | escala da normalização de camada (4); tokens propostos por rodada na decodificação especulativa (15) |
| $\tau$ | temperatura da perda contrastiva (13); limiar de cosseno do cache semântico (15); limiar de confiança do roteador entre modelos (15) |
| $T$ | temperatura de amostragem e de destilação (6, 15); número de tokens do contexto (4, 6, 15); total de passos de difusão (13) |
| $\lambda$ | limiar de abstenção (11); taxa de chegada em filas (15) |
| $\mu$ | média na normalização de camada (4); taxa de atendimento em filas (15) |
| $\sigma$ | sigmoide ou ativação (4, 7); desvio-padrão ($\sigma^2$ na normalização, 4) |
| $b$ | parâmetro do BM25 (10); bytes por número (6, 10); tamanho do lote e bits (15); casos discordantes em McNemar (16) |
| $c$ | constante do RRF (11); servidores em filas (15); custo relativo do rascunhador (15); casos discordantes em McNemar (16) |
| $k$ | número de itens (top-$k$, recall@$k$); constante do RRF (10); $k_1$ do BM25 (10); coluna de $\mathbf{A}$ (12); k-anonimato (17) |
| $r$ | posto do LoRA (12); recompensa (7); posição no ranking (10); radical (11) |
| $N$ | número de camadas (4, em $12Nd^2+Vd$); número de parâmetros (5, 6, 15); número de documentos (3, 10); número de patches (13) |
| $d$ | dimensão do embedding ou do modelo (2, 3, 10; no Cap. 4 também escrita $d_{\text{modelo}}$); dimensão de saída da camada (12); documento (10) |
| $P$ | probabilidade (1, 2); precisão (16); capacidade de cálculo (15) |
| $R$ | revocação (16); conjunto de radicais (11) |
| $\varepsilon$ | constante de estabilidade (4); ruído gaussiano da difusão (13) |
| $e$ | número de Euler (2); margem de erro no tamanho de amostra (16); especificidade do juiz (16); proporção esperada $e_i$ (15) |
| $\Delta$ | atualização de pesos $\Delta\mathbf{W}$ (12); diferença entre grupos (17) |
| $\mathcal{L}$ | perda; as variantes são indicadas por índice ($\mathcal{L}_{\text{SFT}}$, $\mathcal{L}_{\text{DPO}}$ etc.) |
