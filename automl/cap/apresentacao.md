# Apresentação
## Para quem é este livro

Este livro foi escrito para um **público misto**: estudantes de graduação e pós-graduação, profissionais de dados e pessoas curiosas que querem entender como as máquinas conseguem **escolher e ajustar, sozinhas, modelos de aprendizado de máquina**.

Você **não precisa** de conhecimento prévio. O Capítulo 2 faz uma revisão do essencial de aprendizado de máquina, e cada termo técnico é explicado no momento em que aparece. Quando surgir uma fórmula, ela virá com uma tradução "em palavras" e, quase sempre, com um exemplo numérico pequeno.

Ao mesmo tempo, o livro avança. Os capítulos partem do nível básico, passam pelo intermediário e chegam a temas avançados: otimização bayesiana, métodos de multifidelidade, programação genética, meta-aprendizado, busca de arquiteturas neurais, explicabilidade, ética e MLOps.

## O que é AutoML, em uma frase

**AutoML** (aprendizado de máquina automatizado) é o conjunto de técnicas que **automatiza as escolhas** feitas na construção de um modelo: preparo dos dados, atributos, algoritmo, hiperparâmetros e combinação de modelos. O livro mostra **como** essas escolhas são automatizadas, e não apenas como apertar o botão de uma ferramenta.

## Como o livro está organizado

O livro tem **cinco partes** e **catorze capítulos**.

- **Parte I — Fundamentos** (Caps. 1 a 3): o que é AutoML e o problema CASH; o essencial de aprendizado de máquina; pipelines e pré-processamento sem vazamento de dados.
- **Parte II — Otimização de hiperparâmetros** (Caps. 4 a 7): busca em grade e aleatória; otimização bayesiana; Successive Halving, Hyperband e BOHB; métodos evolutivos e TPOT.
- **Parte III — Automação do pipeline** (Caps. 8 a 10): engenharia e seleção automática de atributos; meta-aprendizado, *warm start* e ensembles; ferramentas de AutoML na prática.
- **Parte IV — AutoML avançado** (Caps. 11 e 12): busca de arquiteturas neurais (NAS); AutoML para séries temporais, texto e imagens.
- **Parte V — AutoML responsável e em produção** (Caps. 13 e 14): explicabilidade, ética, vieses e limitações; MLOps e um projeto integrador final.

Cada capítulo segue o mesmo roteiro:

1. **Objetivos de aprendizagem**, no início.
2. **Conteúdo** com analogias, figuras, exemplos e códigos em Python.
3. **Resumo** em cartões de memorização.
4. **30 questões de múltipla escolha** e **20 questões discursivas**, com gabarito no Apêndice A.
5. **Exercícios práticos em Python** (scikit-learn), com soluções e saídas no Apêndice D.
6. **Projeto integrador do capítulo**, com solução de referência e critérios de avaliação no Apêndice E.

O Capítulo 14 termina com o **projeto integrador final**, que reúne o que foi aprendido em um estudo completo.

## Ícones e caixas

Algumas caixas coloridas aparecem ao longo do texto.

- **No seu dia a dia:** liga o conceito a uma situação cotidiana.
- **Exemplo:** um caso numérico ou prático.
- **Dica:** um atalho ou boa prática.
- **Atenção:** um cuidado para evitar erros comuns.
- **Saiba mais:** uma indicação para ir além.
- **Memorize:** a ideia que vale a pena guardar.

## Ambiente de programação

Os códigos usam **Python** e, sobretudo, a biblioteca **scikit-learn**, acompanhada de NumPy, pandas, SciPy e Matplotlib. Para os capítulos de AutoML, usamos também **Optuna** e **FLAML** e, na explicabilidade, **SHAP**. Todos os exemplos executados foram rodados com estas versões:

| Biblioteca | Versão |
|---|---|
| Python | 3.13 |
| scikit-learn | 1.9.1 |
| NumPy / pandas / SciPy | 2.5.3 / 3.0.5 / 1.18.1 |
| Matplotlib | 3.11.2 |
| Optuna / FLAML | 5.0.0 / 2.7.0 |
| SHAP / XGBoost / LightGBM | 0.54.1 / 3.4.1 / 4.7.0 |

Diferenças pequenas nos números são normais com outras versões. Você pode executar tudo no seu computador ou no **Google Colab**, gratuito e no navegador.

## Um aviso importante sobre o que foi testado

Este livro procura ser transparente sobre o que foi executado durante a sua elaboração:

- **Códigos executados.** Todos os códigos do texto e das soluções que **mostram uma saída** foram executados de fato, e as saídas impressas são as obtidas na execução. Os experimentos usam sementes fixas para que você consiga reproduzi-los.
- **Códigos não executados.** Algumas ferramentas de AutoML (TPOT, Auto-sklearn, AutoGluon, H2O AutoML e PyCaret) **não puderam ser instaladas** no ambiente em que o livro foi preparado. Os trechos de código que as usam aparecem **sem saída** e são acompanhados de um aviso. Como as interfaces dessas bibliotecas mudam entre versões, consulte a documentação atual antes de usá-los.
- **Dados.** Os exercícios executados usam conjuntos de dados que acompanham o scikit-learn: Iris, Wine, Breast Cancer Wisconsin e Digits, originários do repositório **UCI Machine Learning**, e Diabetes, originário de um estudo estatístico clássico. Também há dados simulados, sempre que indicado. As bases do UCI e do Kaggle que aparecem na seção "Para ir além" **não foram usadas nem testadas** nos códigos do livro: confira descrição, licença e colunas antes de usá-las.
- **Referências.** Os artigos, livros e eventos citados foram conferidos em buscas na internet durante a elaboração. Quando uma informação (como o intervalo de páginas) não pôde ser confirmada, ela foi **omitida** da referência, e não preenchida de memória. Mesmo assim, confira o DOI e as páginas antes de citar o livro em um trabalho acadêmico.
- **Resultados honestos.** Em vários experimentos, o resultado foi **menos favorável ao método sofisticado** do que se poderia esperar, ou foi um empate com a busca aleatória. Isso foi mantido e comentado de propósito: saber quando uma técnica **não** ajuda é parte da formação.

## Como estudar com este livro

1. **Leia o texto com o código ao lado.** Execute cada bloco e altere os valores para ver o que muda.
2. **Tente os exercícios antes de olhar as soluções.** O aprendizado vem do esforço de resolver.
3. **Faça os projetos.** Eles são o melhor treino para situações reais.
4. **Use o glossário** (Apêndice B) e o **guia rápido** (Apêndice C) como consulta.
5. **Não confie cegamente em nenhum resultado**, nem nos do livro: repita com outras sementes, compare com uma linha de base e desconfie de notas boas demais.

Boa leitura, e bons experimentos!

*Araken de Medeiros Santos*
