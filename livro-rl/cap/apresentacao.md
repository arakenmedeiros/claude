# Apresentação
## Para quem é este livro

Este livro foi escrito para um **público misto**: estudantes de graduação e pós-graduação, profissionais de tecnologia e dados, e curiosos que querem entender como uma máquina **aprende a decidir** por tentativa e erro.

Você **não precisa** de conhecimento prévio em Python, probabilidade, álgebra linear ou aprendizado de máquina. Cada conceito essencial é explicado no momento em que aparece. Quando surge uma fórmula, ela vem acompanhada de uma explicação "em palavras" e, quase sempre, de um exemplo numérico pequeno. Os apêndices oferecem uma introdução a Python e à matemática necessária.

Ao mesmo tempo, o livro avança. Os capítulos partem do nível básico, passam pelo intermediário e chegam a temas avançados, como gradientes de política, algoritmos modernos (PPO, SAC), aprendizado offline, sistemas multiagente e aprendizado a partir de feedback humano.

## Como o livro está organizado

O livro tem **cinco partes** e **treze capítulos**, mais um projeto final.

- **Parte I — Fundamentos:** o que é aprendizado por reforço, bandits multi-armados, Processos de Decisão de Markov e programação dinâmica.
- **Parte II — Métodos Tabulares:** Monte Carlo e diferença temporal (SARSA e Q-learning).
- **Parte III — Aproximação de Funções e Aprendizado por Reforço Profundo:** aproximação de funções, DQN, gradientes de política, atores-críticos, PPO, DDPG, TD3 e SAC.
- **Parte IV — Tópicos Avançados:** aprendizado baseado em modelo, offline, por imitação, multiagente e por preferências humanas (RLHF).
- **Parte V — Projeto Integrador:** um projeto completo que une todos os conceitos.

## Como cada capítulo funciona

Todos os capítulos seguem o mesmo roteiro:

1. **Objetivos de aprendizagem**, para você saber aonde vai chegar.
2. **Conceitos explicados passo a passo**, com figuras e exemplos do cotidiano.
3. **Mão na massa:** exercícios práticos em Python, usando scikit-learn, Gymnasium e bases públicas (UCI e Kaggle).
4. **Projeto do capítulo:** uma tarefa maior para consolidar.
5. **Resumo**, em caixas de memorização.
6. **30 questões de múltipla escolha e 20 discursivas.** O gabarito está no Apêndice A.

## Como ler as caixas coloridas

:::dica Dica
Sugestões práticas para estudar e programar.
:::

:::atencao Atenção
Armadilhas e erros comuns. Vale a pena ler com calma.
:::

:::cotidiano No seu dia a dia
Analogias com situações que você conhece.
:::

:::saibamais Saiba mais
Aprofundamentos que podem ser lidos numa segunda passada.
:::

:::exemplo Exemplo
Contas feitas passo a passo.
:::

:::memorize Memorize
Os pontos que você deve levar do capítulo.
:::

## Sobre os códigos

Todos os códigos estão em **Python 3**. Eles foram escritos para serem executados em ordem, como células de um *notebook* (Jupyter ou Google Colab; veja o Apêndice C). Em cada capítulo os blocos formam uma sequência: o que foi definido num bloco pode ser usado nos seguintes.

Algumas observações importantes:

- O **scikit-learn** não possui algoritmos de aprendizado por reforço. Por isso, os ambientes vêm do **Gymnasium**, os algoritmos são escritos por nós (ou usamos o **Stable-Baselines3**) e o scikit-learn entra em tudo o que cerca o aprendizado: aproximação de funções, pré-processamento, avaliação e análise de dados.
- As bases de dados vêm de repositórios públicos, como o **UCI Machine Learning Repository** e o **Kaggle**. O Apêndice J explica como obtê-las.
- **Os resultados numéricos variam** de uma execução para outra, pois o aprendizado por reforço envolve sorteios. Compare tendências, não casas decimais.

:::atencao Nota sobre a verificação dos códigos
Os blocos de código foram executados de ponta a ponta com Python 3.13, NumPy 2, pandas 3, scikit-learn 1.9, Gymnasium 1.4, PyTorch 2.14 e Stable-Baselines3 2.9. Os resultados numéricos exibidos no texto e as figuras de curvas de aprendizado vêm dessas execuções com os ambientes do Gymnasium. Já os trechos que baixam bases do **UCI** (`ucimlrepo`) e do **Kaggle** foram validados quanto à lógica com dados sintéticos de mesma estrutura, pois o ambiente de verificação não tinha acesso a esses repositórios. Por isso, **não citamos números obtidos com as bases reais**: ao executá-las, você verá os seus próprios resultados, que podem diferir dos de nossos testes. Nomes de colunas e identificadores podem mudar com o tempo (Apêndice J).
:::

## Apêndices

- **Apêndice A:** gabarito de todas as questões (múltipla escolha e discursivas).
- **Apêndice B:** glossário de termos.
- **Apêndices C a H:** introdução a Python, NumPy/pandas/Matplotlib, scikit-learn, Gymnasium, PyTorch e Stable-Baselines3.
- **Apêndice I:** matemática essencial, sem medo.
- **Apêndice J:** bases de dados do UCI e do Kaggle.

## Sugestão de percursos

- **Leitura completa (um semestre):** Capítulos 1 a 13 em ordem, com os projetos.
- **Trilha rápida para a prática:** Capítulos 1, 2, 6, 8 e 10, com os apêndices C a H.
- **Trilha para quem vem de aprendizado de máquina:** leia os Capítulos 1 a 3 rapidamente e aprofunde-se a partir do Capítulo 4.

Bons estudos!

*Araken de Medeiros Santos*
