# Apêndice C — Símbolos, notação e dimensões

## C.1 Convenções gerais

Amostras nas **linhas**: $X\in\mathbb{R}^{N\times n_0}$, alvos $T\in\mathbb{R}^{N\times m}$. Letras minúsculas em negrito são vetores-coluna; $\theta\in\mathbb{R}^P$ reúne **todos** os pesos e vieses da rede.

## C.2 Tabela de símbolos

| Símbolo | Significado | Dimensão |
|---|---|---|
| $N$ | número de amostras | escalar |
| $L$ | número de camadas com pesos | escalar |
| $n_l$ | neurônios da camada $l$ ($n_0$: entradas; $n_L=m$: saídas) | escalar |
| $W^{(l)}$ | pesos da camada $l$ | $n_l\times n_{l-1}$ |
| $\mathbf{b}^{(l)}$ | vieses da camada $l$ | $n_l$ |
| $Z^{(l)}$ | pré-ativações da camada $l$, $Z^{(l)}=A^{(l-1)}W^{(l)T}+\mathbf{1}\mathbf{b}^{(l)T}$ | $N\times n_l$ |
| $A^{(l)}$ | ativações, $A^{(l)}=f_l(Z^{(l)})$, com $A^{(0)}=X$ | $N\times n_l$ |
| $\theta$ | vetor único de parâmetros | $P$ |
| $P$ | número de parâmetros, $P=\sum_l n_l(n_{l-1}+1)$ | escalar |
| $r$ | resíduos, $r=y-t$ (achatados) | $Nm$ |
| $E$ | erro, $E=c\sum r^2$; $c=\tfrac12$ ("sse") ou $c=1/(Nm)$ ("mse") | escalar |
| $g=\nabla E$ | gradiente, $g=J^Tr$ (quando $c=\tfrac12$) | $P$ |
| $J$ | Jacobiana dos resíduos, $J=\partial r/\partial\theta$ | $Nm\times P$ |
| $H$ | Hessiana de $E$ | $P\times P$ |
| $\delta^{(l)}$ | sensibilidade, $\delta^{(l)}=\partial E/\partial Z^{(l)}$ | $N\times n_l$ |
| $\eta,\ \alpha$ | taxa de aprendizagem (global; por peso) | escalar; $P$ |
| $\beta,\ \mu$ | coeficiente de momento; parâmetro de amortecimento de LM | escalar |
| $\Delta_i$ | passo individual do RPROP | $P$ |
| $\kappa,\phi,\theta_{\text{DBD}}$ | aumento aditivo, redução multiplicativa e memória do DBD | escalares |

:::atencao Dois usos de $\theta$
Neste livro, $\theta$ (sem índice) é o vetor de **parâmetros**. No Capítulo 8, a constante de memória do Delta-Bar-Delta aparece como $\theta$ nas fórmulas do texto e como `teta` no código; o contexto distingue os dois usos.
:::

## C.3 Verificação de dimensões

Uma regra prática: sempre confira que cada produto de matrizes tem dimensões internas iguais. Exemplo da rede 1–8–1 com $N=100$: $X$ é $100\times1$; $W^{(1)}$ é $8\times1$; $Z^{(1)}=XW^{(1)T}+\mathbf{1}\mathbf{b}^{(1)T}$ é $100\times8$; $W^{(2)}$ é $1\times8$; $Z^{(2)}$ é $100\times1$; $P=8\cdot2+1\cdot9=25$ parâmetros; $J$ é $100\times25$.
