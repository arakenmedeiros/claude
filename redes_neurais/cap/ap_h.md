# Apêndice H — Quadro comparativo dos algoritmos

| Algoritmo | Ordem | O que adapta | Usa a magnitude do gradiente? | Hiperparâmetros principais | Custo por iteração | Observações |
|---|---|---|---|---|---|---|
| Gradiente descendente | 1ª | nada (taxa global) | sim | $\eta$ (e $\beta$) | $O(P)$ após o gradiente | simples; sensível à taxa e ao mal-condicionamento |
| Quickprop | 1ª (modelo quadrático por peso) | passo por peso via secante | sim | $\varepsilon$, $\mu$, decaimento | $O(P)$ | pode ser instável; regras de segurança necessárias |
| RPROP (quatro variantes) | 1ª | passo por peso | **não** (só o sinal) | $\eta^+$, $\eta^-$, $\Delta_0$, $\Delta_{\min}$, $\Delta_{\max}$ | $O(P)$ | robusto a escala; só em lote |
| Levenberg–Marquardt | 2ª aproximada ($J^TJ$) | $\mu$ (amortecimento) | sim | $\mu_0$, fatores de $\mu$ | $O(NmP^2+P^3)$; memória $O(NmP+P^2)$ | só para soma de quadrados; ótimo em redes pequenas |
| Delta-delta | 1ª | taxa por peso | sim | $\alpha_0$, $\gamma$ | $O(P)$ | frágil; dependente da escala |
| Delta-Bar-Delta | 1ª | taxa por peso | sim | $\alpha_0$, $\kappa$, $\phi$, $\theta$ | $O(P)$ | quatro hiperparâmetros; só em lote |
| EDBD (versão didática) | 1ª | taxa e momento por peso | sim | $\ge9$ | $O(P)$ | mais parâmetros; limites $\alpha_{\max}$, $\mu_{\max}$ |

:::limite Como usar este quadro
O quadro resume propriedades **conceituais**. Ele não classifica os algoritmos por desempenho: os experimentos do livro mostram que o resultado depende do problema, da arquitetura, da inicialização e do ajuste de hiperparâmetros, e que os métodos ficam, com frequência, empatados dentro da variação entre sementes.
:::

## H.1 Quando considerar cada método (orientação, não regra)

- **Rede pequena, regressão, poucos dados:** o Levenberg–Marquardt costuma ser muito eficiente (Capítulo 7); use validação e cuidado com o sobreajuste.
- **Muitos pesos ou muitas amostras:** métodos de primeira ordem; memória e tempo do LM tornam-no inviável (por exemplo, $J$ de $10^{10}$ elementos).
- **Insensibilidade à escala do gradiente:** RPROP.
- **Curvaturas muito diferentes entre pesos, treinamento em lote:** DBD/EDBD podem ajudar, ao custo de mais hiperparâmetros.
- **Minilotes ruidosos:** os métodos de lote deste livro não são projetados para esse regime.
