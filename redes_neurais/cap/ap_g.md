# Apêndice G — Glossário

**Ativação.** Valor de saída de um neurônio, $a=f(z)$, em que $f$ é a função de ativação.

**Backpropagation (retropropagação).** Algoritmo que calcula o gradiente do erro em relação a todos os pesos aplicando a regra da cadeia da saída para a entrada.

**Batelada (lote).** Uso de todas as amostras para calcular cada gradiente. Contraste com **minilote** e **padrão a padrão**.

**Condição (número de).** Razão entre o maior e o menor autovalor da Hessiana (em quadráticas); mede a dificuldade da otimização.

**Delta ($\delta$).** Sensibilidade do erro à pré-ativação, $\delta=\partial E/\partial z$.

**Delta-Bar-Delta (DBD).** Regra de taxas individuais com aumento aditivo e redução multiplicativa, guiada pelo sinal de $\bar g\,g$.

**Delta-delta.** Regra de adaptação da taxa pela derivada do erro em relação à taxa: $\Delta\alpha=\gamma\,g(t)g(t-1)$.

**Diferenças finitas.** Aproximação de derivadas por $(f(x+h)-f(x-h))/2h$; usada para verificar gradientes.

**Época.** Uma passagem completa por todas as amostras de treino.

**Gauss–Newton.** Método para mínimos quadrados que aproxima a Hessiana por $J^TJ$.

**Gradiente.** Vetor das derivadas parciais do erro em relação aos parâmetros.

**Hessiana.** Matriz das derivadas parciais de segunda ordem.

**Hiperparâmetro.** Parâmetro do algoritmo escolhido pelo usuário (taxa, momento, $\mu$, $\kappa$…), em oposição aos pesos, aprendidos.

**Jacobiana.** Matriz das derivadas dos resíduos em relação aos parâmetros.

**Levenberg–Marquardt (LM).** Método que resolve $(J^TJ+\mu I)\mathbf{d}=-g$, interpolando entre Gauss–Newton e gradiente descendente.

**LMBP.** Levenberg–Marquardt aplicado ao treinamento por backpropagation (Hagan; Menhaj, 1994).

**Momento.** Termo que soma uma fração do passo anterior ao passo atual.

**Overfitting (sobreajuste).** Ajuste excessivo ao treino, com piora em dados novos.

**Parada antecipada.** Interromper o treino quando o erro de validação deixa de melhorar.

**Perceptron.** Neurônio de limiar com regra de aprendizagem por correção de erros (Rosenblatt, 1958).

**Quickprop.** Método que aproxima a derivada do erro por uma parábola obtida por secante, por peso.

**Resíduo.** Diferença $r=y-t$ entre saída e alvo.

**RPROP.** Método de passos individuais que usa apenas o sinal do gradiente (RPROP+, RPROP−, iRPROP+, iRPROP−).

**Taxa de aprendizagem.** Fator que multiplica o gradiente no passo do gradiente descendente.

**Vazamento de dados.** Uso de informação de validação ou teste durante o treino ou o pré-processamento.
