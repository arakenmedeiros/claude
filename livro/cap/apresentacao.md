# Apresentação
## Para quem é este livro

Este livro foi escrito para um **público misto**: estudantes de graduação e pós-graduação, profissionais de dados e curiosos que querem entender como as máquinas descobrem padrões **sem receber respostas prontas**.

Você **não precisa** de base prévia em matemática ou programação. Cada conceito técnico essencial é explicado no momento em que aparece. Quando uma fórmula surgir, ela virá com uma explicação "em palavras" e, quase sempre, com um exemplo numérico pequeno.

Ao mesmo tempo, o livro avança. Os capítulos começam no nível básico, passam pelo intermediário e chegam a temas avançados, como aprendizado de variedades, autoencoders variacionais, redes adversariais e aprendizado contrastivo.

## Como o livro está organizado

O livro tem **cinco partes** e **doze capítulos**.

- **Parte I — Fundamentos:** o que é aprendizado não supervisionado; dados, distâncias e pré-processamento.
- **Parte II — Agrupamento:** K-means, hierárquico, densidade (DBSCAN, OPTICS, HDBSCAN), modelos de mistura, agrupamento espectral e avaliação.
- **Parte III — Redução de dimensionalidade:** PCA, ICA, NMF, MDS, Isomap, t-SNE e UMAP.
- **Parte IV — Padrões e anomalias:** regras de associação e detecção de anomalias.
- **Parte V — Aprendizado profundo:** autoencoders, VAEs, GANs, difusão e aprendizado auto-supervisionado.

Cada capítulo segue o mesmo roteiro:

1. **Objetivos de aprendizagem**, no início.
2. **Conteúdo** com analogias, figuras, exemplos e códigos em Python.
3. **Resumo** em cartões de memorização.
4. **30 questões de múltipla escolha** e **20 questões discursivas**.

O **gabarito de todas as questões** está no Apêndice A.

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

Todos os trechos de código em Python deste livro foram executados durante a elaboração do material, com as versões de bibliotecas abaixo. Resultados numéricos podem variar um pouco em versões diferentes.

| Componente | Versão utilizada |
|---|---|
| Python | 3.13 |
| NumPy | 2.5 |
| SciPy | 1.18 |
| pandas | 3.0 |
| scikit-learn | 1.9 |
| matplotlib | 3.11 |
| mlxtend | 0.25 |
| umap-learn | 0.5 |
| PyTorch | 2.14 |

Os exemplos usam **conjuntos de dados embutidos no scikit-learn** (como Iris, Wine e Dígitos) ou **dados sintéticos** gerados no próprio código, com sementes fixas. Assim, você não precisa baixar arquivos externos. Os dados de clientes, de pedidos e de coordenadas geográficas dos exemplos são simulados e servem apenas para fins didáticos.

## Leituras recomendadas

Para aprofundar os estudos, alguns livros de referência são:

- Hastie, Tibshirani e Friedman (2009), *The Elements of Statistical Learning*, que dedica um capítulo ao aprendizado não supervisionado.
- Bishop (2006), *Pattern Recognition and Machine Learning*, com tratamento probabilístico de misturas gaussianas e do algoritmo EM.
- Murphy (2012), *Machine Learning: A Probabilistic Perspective*, uma visão ampla e rigorosa.
- Goodfellow, Bengio e Courville (2016), *Deep Learning*, para a Parte V.
- Aggarwal (2015), *Data Mining: The Textbook*, que cobre agrupamento, regras de associação e anomalias.
- Jain, Murty e Flynn (1999) e Jain (2010), duas revisões clássicas sobre agrupamento.

As referências completas estão ao final do livro. Priorizamos artigos publicados em periódicos de alto impacto, trabalhos apresentados em conferências de referência da área e livros reconhecidos.

Bons estudos!
