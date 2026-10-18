# Apresentação
## Para quem é este livro

Este livro foi escrito para um **público misto**: estudantes de graduação e pós-graduação, profissionais de dados e de software que querem colocar modelos de aprendizado de máquina em produção, e curiosos que desejam entender o que acontece **depois** que um modelo é treinado.

Você **não precisa** de experiência prévia em produção. É útil conhecer o básico de Python e a ideia de treinar um modelo, mas cada conceito técnico essencial é explicado no momento em que aparece. Quando uma fórmula surgir, ela virá com uma explicação "em palavras" e, quase sempre, com um exemplo numérico pequeno.

Ao mesmo tempo, o livro avança. Os capítulos começam no nível básico (por que modelos precisam de operação), passam pelo intermediário (pipelines, rastreamento, testes, serviço) e chegam a temas avançados (detecção de deriva, treinamento contínuo, governança, justiça, privacidade, armazém de atributos, orquestração e LLMOps).

## Como o livro está organizado

O livro tem **cinco partes** e **onze capítulos**.

- **Parte I — Fundamentos de MLOps:** por que modelos não chegam (ou não ficam) em produção; ciclo de vida, dívida técnica e maturidade.
- **Parte II — Dados e Experimentos:** qualidade, validação e versionamento de dados; pipelines reprodutíveis; rastreamento de experimentos e registro de modelos; avaliação rigorosa e otimização de hiperparâmetros.
- **Parte III — Entrega e Operação:** empacotamento e serviço; testes e CI/CD; monitoramento, deriva e retreinamento.
- **Parte IV — Confiança e Escala:** governança, explicabilidade, justiça, privacidade, armazém de atributos, orquestração e LLMOps.
- **Parte V — Projeto Final:** um sistema completo, do dado à produção, com monitoramento e retreino.

Cada capítulo segue o mesmo roteiro:

1. **Objetivos de aprendizagem**, no início.
2. **Conteúdo** com analogias, figuras, exemplos e códigos em Python.
3. **Resumo** em cartões de memorização.
4. **30 questões de múltipla escolha** e **20 questões discursivas**.
5. **Exercícios práticos em Python** com a biblioteca scikit-learn (e as ferramentas de MLOps do capítulo), em ordem crescente de dificuldade.
6. **Projeto integrador do capítulo**, com contexto, tarefas, entregáveis e critérios de avaliação.
7. Um quadro **"Para ir além"**, com conjuntos de dados públicos da UCI e do Kaggle para praticar.

O **gabarito de todas as questões** está no Apêndice A. As **soluções dos exercícios práticos** estão no Apêndice D, e a **solução de referência dos projetos integradores**, no Apêndice E. Tente resolver sozinho antes de consultá-las.

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

Todos os trechos de código **em Python** deste livro foram executados durante a elaboração do material, com as versões de bibliotecas abaixo. Resultados numéricos podem variar um pouco em versões diferentes.

| Componente | Versão utilizada |
|---|---|
| Python | 3.13 |
| NumPy | 2.5 |
| SciPy | 1.18 |
| pandas | 3.0 |
| scikit-learn | 1.9 |
| matplotlib | 3.11 |
| joblib | 1.6 |
| MLflow | 3.17 |
| FastAPI | 0.143 |
| Pydantic | 2.14 |
| pytest | 9.1 |

**Atenção:** os trechos que **não** são Python (comandos de terminal, arquivos `Dockerfile` e fluxos de CI em YAML, comandos do DVC) são **ilustrativos** e **não foram executados** na elaboração do livro, pois o ambiente de escrita não dispunha de Docker, DVC nem de um serviço de CI. Teste-os no seu ambiente e consulte a documentação oficial de cada ferramenta, porque versões e opções mudam com o tempo.

Os exemplos e projetos usam **conjuntos de dados embutidos no scikit-learn** ou **dados simulados** gerados no próprio código, com sementes fixas. Assim, você não precisa baixar arquivos externos e consegue reproduzir os resultados.

Os conjuntos embutidos vêm do repositório **UCI Machine Learning**: *Iris*, *Wine*, *Breast Cancer Wisconsin (Diagnostic)* e *Optical Recognition of Handwritten Digits* (a parte de teste, com 1.797 imagens). Os dados de clientes, de pedidos e de crédito usados em alguns exemplos e projetos são **simulados** e servem apenas para fins didáticos; cada projeto informa a origem dos seus dados. A **produção** e a **deriva** dos projetos também são simuladas.

### Bases públicas da UCI e do Kaggle

Ao final de cada capítulo, o quadro **"Para ir além"** sugere conjuntos de dados públicos do repositório **UCI Machine Learning** e do **Kaggle** para você repetir os exercícios com dados reais. Esses conjuntos **não foram usados nem testados nos códigos deste livro**: confira nos sites a descrição, a licença e as colunas antes de usá-los. Em geral, o fluxo é este:

- **UCI:** baixe o arquivo no site do repositório ou use o pacote `ucimlrepo`.
- **Kaggle:** baixe o arquivo no site (é preciso criar uma conta) ou use a API do Kaggle com um token pessoal.

Depois, carregue os dados com `pandas`, inspecione-os (formato, tipos, ausentes) e aplique o roteiro do capítulo.

## Leituras recomendadas

Para aprofundar os estudos, alguns livros e artigos de referência são:

- Huyen (2022), *Designing Machine Learning Systems*, visão prática e abrangente de sistemas de ML em produção.
- Sculley et al. (2015), "Hidden technical debt in machine learning systems", o artigo clássico sobre dívida técnica em ML.
- Kreuzberger, Kühl e Hirschl (2023), "Machine learning operations (MLOps): overview, definition, and architecture", uma revisão acadêmica do tema.
- Breck et al. (2017), "The ML test score", uma rubrica de prontidão para produção.
- Forsgren, Humble e Kim (2018), *Accelerate*, sobre medidas de desempenho de entrega de software.
- Beyer et al. (2016), *Site Reliability Engineering*, sobre confiabilidade de sistemas.
- Kohavi, Tang e Xu (2020), *Trustworthy Online Controlled Experiments*, sobre testes A/B.
- Barocas, Hardt e Narayanan (2023), *Fairness and Machine Learning*, sobre justiça em ML.

As referências completas estão ao final do livro. Priorizamos artigos publicados em periódicos de alto impacto, trabalhos apresentados em conferências de referência da área e livros reconhecidos.

Bons estudos!
