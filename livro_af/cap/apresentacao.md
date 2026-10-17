# Apresentação
## Para quem é este livro

Este livro foi escrito para um **público misto**: estudantes de graduação e pós-graduação, profissionais de dados e curiosos que querem entender como as máquinas aprendem **em equipe, sem que ninguém precise entregar seus dados**.

Você **não precisa** de base avançada em matemática ou programação. Cada conceito técnico essencial é explicado no momento em que aparece. Quando uma fórmula surgir, ela virá com uma explicação "em palavras" e, quase sempre, com um exemplo numérico pequeno.

Ao mesmo tempo, o livro avança. Os capítulos começam no nível básico, passam pelo intermediário e chegam a temas avançados, como privacidade diferencial, agregação segura, ataques de inversão de gradientes, aprendizado federado vertical e modelos de fundação federados.

## Por que este tema importa

Dados pessoais, de saúde e financeiros são valiosos e sensíveis. Ao mesmo tempo, os modelos de inteligência artificial precisam de muitos dados para funcionar bem. O **aprendizado federado** propõe uma saída: **levar o modelo até os dados**, em vez de levar os dados até o modelo.

Este livro explica essa ideia do começo ao fim: como funciona, onde falha, como se proteger e como praticar.

## Como o livro está organizado

O livro tem **cinco partes** e **doze capítulos**.

- **Parte I — Fundamentos:** o que é aprendizado federado, as bases de treino de modelos e o algoritmo FedAvg.
- **Parte II — Desafios do treino federado:** dados diferentes entre clientes (não-IID), economia de comunicação e personalização.
- **Parte III — Privacidade e segurança:** ameaças à privacidade, privacidade diferencial, agregação segura e robustez contra participantes mal-intencionados.
- **Parte IV — Arquiteturas, aplicações e fronteiras:** aprendizado federado vertical, descentralizado e *split learning*; aprendizado não supervisionado federado, modelos de fundação, ferramentas, aplicações, regulação e ética.
- **Parte V — Projeto final:** um estudo completo que integra todas as técnicas.

Cada capítulo segue o mesmo roteiro:

1. **Objetivos de aprendizagem**, no início.
2. **Conteúdo** com analogias, figuras, exemplos e códigos em Python.
3. **Resumo** em cartões de memorização.
4. **30 questões de múltipla escolha** e **20 questões discursivas**.
5. **Exercícios práticos em Python** com a biblioteca scikit-learn, em ordem crescente de dificuldade.
6. **Projeto integrador do capítulo**, com contexto, tarefas, entregáveis e critérios de avaliação.
7. Um quadro **"Para ir além"**, com conjuntos de dados públicos da UCI e do Kaggle para praticar.

O **gabarito de todas as questões** está no Apêndice A. O **glossário** está no Apêndice B e o **guia rápido**, com o código da biblioteca `fedlib.py`, no Apêndice C. As **soluções dos exercícios práticos** estão no Apêndice D, e a **solução de referência dos projetos integradores**, no Apêndice E. Tente resolver sozinho antes de consultá-las.

## Como estudar com este livro

- **Estude em blocos curtos.** Sessões de 25 a 40 minutos, com pausas, ajudam a manter a atenção.
- **Pare e explique.** Após cada seção, tente explicar a ideia, em voz alta, como se falasse com um amigo.
- **Rode os códigos.** Altere um parâmetro por vez e observe o que muda.
- **Responda antes de olhar o gabarito.** Tentar lembrar fixa mais do que reler.
- **Volte aos cartões de memorização** dias depois. Revisões espaçadas ajudam a memória de longo prazo.

## Caixas e símbolos do livro

:::dica Dica
Sugestões práticas para facilitar seu trabalho.
:::

:::atencao Atenção
Cuidados e erros comuns. Leia com calma.
:::

:::cotidiano No seu dia a dia
Exemplos do cotidiano que aproximam o conceito da sua realidade.
:::

:::saibamais Saiba mais
Curiosidades e aprofundamentos para quem quer ir além.
:::

:::exemplo Exemplo ou exercício guiado
Pequenas atividades para você experimentar.
:::

:::memorize Memorize
O essencial do assunto, em poucas linhas.
:::

## Sobre os códigos

Todos os códigos foram escritos em **Python 3** e testados com **NumPy**, **scikit-learn** e **matplotlib**. Para a maior parte do livro, basta instalar:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Os experimentos usam conjuntos de dados da **UCI Machine Learning Repository** (Dua; Graff, 2019) que já vêm no scikit-learn (dígitos manuscritos, câncer de mama de Wisconsin, vinhos, íris). Os quadros "Para ir além" sugerem outras bases da UCI e do Kaggle, que **não foram usadas nem testadas** nos códigos do livro: confira descrição, licença e colunas antes de utilizá-las.

A biblioteca didática **`fedlib.py`**, apresentada no Capítulo 3, simula clientes e servidor em um só computador. Salve-a na pasta dos seus exercícios.

:::atencao Atenção
As simulações do livro são **didáticas**. Os números exatos podem variar com versões diferentes das bibliotecas. Em um sistema real, há também rede, autenticação, falhas e requisitos legais que a simulação não mostra.
:::

## Agradecimentos e uso

Bons estudos! Se você encontrar erros ou tiver sugestões, registre-as: elas ajudam a melhorar o material nas próximas edições.

*Araken de Medeiros Santos*
