# Apresentação
## Para quem é este livro

Este livro nasceu de uma pergunta que os estudantes fazem em toda primeira aula sobre redes neurais: **como, exatamente, uma rede aprende?** A resposta curta é "ajustando pesos para diminuir um erro". A resposta completa, que leva do perceptron ao Levenberg–Marquardt, ocupa estas páginas.

Foi escrito para um **público misto**: estudantes de graduação e pós-graduação, profissionais de dados e curiosos. **Não se supõe conhecimento prévio** de cálculo, álgebra linear ou programação: o Capítulo 2 reúne, com demonstrações, tudo o que é necessário (derivadas parciais, gradiente, Jacobiana, Hessiana, mínimos quadrados, Python e scikit-learn). Quem já domina esse material pode percorrê-lo como revisão e seguir adiante.

## O que você vai encontrar

O livro tem **duas partes e oito capítulos**.

- **Parte 1 — Introdução e fundamentos matemáticos e computacionais.** O Capítulo 1 situa o problema (o que é uma rede, o que significa "aprender", uma primeira rede funcionando). O Capítulo 2 reúne a matemática e as ferramentas computacionais.
- **Parte 2 — Perceptron e treinamento de redes de múltiplas camadas.** O Capítulo 3 estuda o perceptron e suas limitações. O Capítulo 4 deduz o **backpropagation** e o perceptron de múltiplas camadas. Os Capítulos 5 a 8 apresentam algoritmos que aceleram ou estabilizam o treinamento: **Quickprop** (5), **RPROP e suas variantes** (6), **Levenberg–Marquardt** (7) e **Delta-Bar-Delta e sua versão estendida** (8).

Cada capítulo traz situação inicial, objetivos, pré-requisitos, demonstrações passo a passo, exemplos resolvidos, figuras, algoritmos em pseudocódigo, implementações em Python, erros frequentes, síntese, **30 questões de múltipla escolha** e **20 discursivas** (240 e 160 no livro todo), exercícios práticos com scikit-learn e um projeto. O livro termina com um **projeto integrador** e apêndices (gabaritos comentados, símbolos e dimensões, revisão matemática, instalação, guia dos códigos, glossário, quadro comparativo e correspondência MATLAB↔Python).

## Como estudar

1. **Leia com lápis.** As demonstrações foram escritas para serem acompanhadas, linha a linha. Se uma passagem parecer curta demais, escreva você mesmo o passo intermediário.
2. **Rode os códigos.** Todos os experimentos reportados foram executados com as versões e sementes indicadas no Apêndice E; reproduza-os e altere os parâmetros.
3. **Resolva os exercícios antes de olhar o gabarito.** O Apêndice A explica por que cada alternativa correta é correta.
4. **Desconfie das conclusões.** Em quase todos os capítulos há uma seção sobre o que um experimento **não** permite afirmar. Compare métodos com partições separadas de treino, validação e teste e com várias sementes.

## Convenções

As amostras ficam nas **linhas** das matrizes; $W^{(l)}$ tem dimensão $n_l\times n_{l-1}$; todos os parâmetros da rede podem ser reunidos em um único vetor $\theta$; o resíduo é $r=y-t$ e o erro é $E=c\sum r^2$, com $c=\tfrac12$ (soma) ou $c=1/(Nm)$ (média). Os blocos coloridos indicam a função de cada trecho: situação inicial, pré-requisitos, demonstração, exemplo resolvido, atenção, erro frequente, limite do método e síntese.

As implementações próprias são identificadas como **"implementação didática do livro"**: foram escritas para ensinar e testadas contra verificações numéricas (diferenças finitas, comparação com o SciPy), mas **não** pretendem substituir bibliotecas profissionais.

## Limitações e honestidade intelectual

- **Fontes.** Os artigos originais de Fahlman (1988), Jacobs (1988), Riedmiller e Braun (1993), Igel e Hüsken (2000, 2003) e Minai e Williams (1990) **não foram lidos diretamente** na preparação do texto: as formulações seguem a literatura secundária e a notação do livro, com ressalvas onde há dúvida. Detalhes de funções do MATLAB devem ser conferidos na documentação da sua versão.
- **Originalidade.** O livro é um texto independente. O livro de referência indicado na encomenda do projeto **não foi consultado**.
- **Dados.** Os experimentos usam bases embutidas no scikit-learn (originárias do repositório UCI) e dados sintéticos. Os códigos que baixam bases externas (UCI, Kaggle) são sugeridos como extensão e **não foram executados** aqui.
- **Resultados numéricos.** Foram obtidos com NumPy 2.5.3, SciPy 1.18.1 e scikit-learn 1.9.1, em um único computador e, em geral, com poucas sementes. **Valem para o problema e a configuração descritos; não são afirmações universais** sobre os algoritmos.
- **Registro de pendências.** O arquivo `PENDENCIAS.md` que acompanha o material lista o que foi e o que não foi verificado.

Bom estudo!

*Araken de Medeiros Santos*
