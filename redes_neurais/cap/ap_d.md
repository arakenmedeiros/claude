# Apêndice D — Revisão matemática

Este apêndice é um resumo de consulta; as demonstrações completas estão no Capítulo 2.

## D.1 Derivadas e regra da cadeia

Para $y=f(g(x))$: $y'=f'(g(x))\,g'(x)$. Para funções de várias variáveis, $\partial y/\partial x_j=\sum_k(\partial y/\partial u_k)(\partial u_k/\partial x_j)$. O backpropagation é a aplicação sistemática dessa regra, de trás para frente.

## D.2 Gradiente, Jacobiana e Hessiana

- **Gradiente** de $E:\mathbb{R}^P\to\mathbb{R}$: o vetor de derivadas parciais $\nabla E=(\partial E/\partial\theta_j)_j$; aponta na direção de maior crescimento.
- **Jacobiana** de $r:\mathbb{R}^P\to\mathbb{R}^{Nm}$: a matriz $J_{ij}=\partial r_i/\partial\theta_j$.
- **Hessiana**: $H_{jk}=\partial^2E/\partial\theta_j\partial\theta_k$, simétrica se $E$ é de classe $C^2$.
- Para $E=\tfrac12\|r\|^2$: $\nabla E=J^Tr$ e $H=J^TJ+\sum_ir_i\nabla^2r_i$.

## D.3 Série de Taylor

$E(\theta+\mathbf{d})=E(\theta)+g^T\mathbf{d}+\tfrac12\mathbf{d}^TH\mathbf{d}+O(\|\mathbf{d}\|^3)$. Condições de mínimo local: $g=0$ e $H\succeq0$ (necessárias); $g=0$ e $H\succ0$ (suficientes).

## D.4 Álgebra linear essencial

- $A$ simétrica é **definida positiva** se $\mathbf{v}^TA\mathbf{v}>0$ para todo $\mathbf{v}\neq0$; equivale a autovalores positivos.
- $J^TJ$ é semidefinida positiva, pois $\mathbf{v}^TJ^TJ\mathbf{v}=\|J\mathbf{v}\|^2\ge0$; $J^TJ+\mu I$ é definida positiva se $\mu>0$.
- **Número de condição** de uma matriz simétrica definida positiva: $\kappa(A)=\lambda_{\max}/\lambda_{\min}$. Para $\tfrac12\mathbf{x}^TA\mathbf{x}$, o gradiente descendente com passo fixo converge em um número de iterações proporcional a $\kappa(A)$.
- **Mínimos quadrados**: $\min\|A\mathbf{x}-\mathbf{b}\|^2$ tem como solução as equações normais $A^TA\mathbf{x}=A^T\mathbf{b}$; em código, resolva o sistema (por exemplo, `np.linalg.lstsq`), nunca inverta $A^TA$.

## D.5 Quadráticas e passo ótimo

Para $f(\mathbf{x})=\tfrac12\sum_i\lambda_ix_i^2$ e passo $\eta$, cada coordenada é multiplicada por $1-\eta\lambda_i$ a cada iteração: converge se $\eta<2/\lambda_{\max}$; o melhor passo único é $\eta^*=2/(\lambda_{\min}+\lambda_{\max})$, com fator de contração $(\kappa-1)/(\kappa+1)$.

## D.6 Probabilidade e estatística mínimas

- **Mediana** e **intervalo** são usados para resumir execuções com sementes diferentes; a média é sensível a execuções divergentes.
- **Variância do ruído**: se $y=f(x)+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma^2)$, o menor erro quadrático médio alcançável é $\sigma^2$ (e $\tfrac12\sigma^2$ com o fator $\tfrac12$).
- **Vazamento de dados**: qualquer informação do conjunto de validação ou teste que influencie o ajuste (escalonamento, seleção de atributos, escolha de hiperparâmetros) invalida a estimativa de desempenho.
