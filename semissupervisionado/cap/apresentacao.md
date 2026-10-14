# Apresentação
## Para quem é este livro

Este livro foi escrito para um **público misto**: estudantes de graduação e pós-graduação, profissionais de dados e curiosos que querem entender como as máquinas aprendem com **poucos exemplos rotulados** e **muitos exemplos sem rótulo**.

Você **não precisa** de base prévia em matemática ou programação. Cada conceito técnico essencial é explicado no momento em que aparece. Quando uma fórmula surgir, ela virá com uma explicação em palavras e, quase sempre, com um exemplo numérico pequeno.

Ao mesmo tempo, o livro avança. Os capítulos começam no nível básico, passam pelo intermediário e chegam a temas avançados, como regularização por consistência, FixMatch, aprendizado contrastivo e modelos generativos profundos.

## Por que este tema?

Rotular dados custa caro. Um radiologista precisa de tempo para laudar uma tomografia. Uma pessoa precisa ouvir uma hora de áudio para transcrevê-la. Já os dados sem rótulo se acumulam todos os dias, quase de graça.

O **aprendizado semissupervisionado** nasce dessa assimetria. Ele busca usar **os dois tipos de dado ao mesmo tempo**. Quando funciona, o ganho pode ser grande. Mas há um aviso importante, que atravessa todo o livro: **dados sem rótulo não são mágica**. Eles ajudam quando a estrutura deles tem relação com os rótulos, e podem prejudicar quando não tem. Por isso este livro insiste em **avaliar com honestidade**.

## Como o livro está organizado

O plano completo do livro tem **cinco partes** e **dezessete capítulos**, mais um projeto integrador.

- **Parte I — Fundamentos:** rótulos escassos, ferramentas essenciais (Python, dados, probabilidade, avaliação) e as suposições que explicam quando dados sem rótulo ajudam. *(Capítulos 1 a 3.)*
- **Parte II — Métodos clássicos:** autotreinamento, co-treinamento, modelos generativos, separação por baixa densidade (S3VM) e métodos baseados em grafos. *(Capítulos 4 a 8.)*
- **Parte III — Métodos modernos com aprendizado profundo:** redes neurais, regularização por consistência, pseudo-rótulos modernos (MixMatch, FixMatch), aprendizado contrastivo e modelos generativos profundos. *(Capítulos 9 a 13.)*
- **Parte IV — Prática e aplicações:** texto, imagens, séries temporais e dados tabulares, além de avaliação, armadilhas e ética. *(Capítulos 14 a 17.)*
- **Parte V — Fechamento:** projeto integrador, glossário e apêndices.

:::atencao Atenção: o que esta edição contém
Esta edição reúne a **Parte I completa** (Capítulos 1 a 3), com todos os exercícios, os projetos de capítulo e os apêndices correspondentes (gabarito, glossário, guia rápido, soluções e referências). As Partes II a V seguirão em novas edições, no mesmo formato. Por isso, algumas seções fazem remissões a capítulos futuros (por exemplo, "veja o Capítulo 8"): são o roteiro do livro completo.
:::

## Como cada capítulo é construído

Todos os capítulos seguem o mesmo roteiro.

1. **Objetivos de aprendizagem**, para você saber aonde vai chegar.
2. **Explicação** em frases curtas, com analogias do cotidiano e muitas figuras.
3. **Código em Python** com a biblioteca scikit-learn, **executado de verdade**, com a saída mostrada.
4. **Resumo** com as ideias para memorizar.
5. **30 questões de múltipla escolha** e **20 questões discursivas**, com gabarito no Apêndice A.
6. **Exercícios práticos em Python** (cinco por capítulo), com soluções no Apêndice D.
7. **Projeto integrador do capítulo**, com solução de referência e critérios de avaliação no Apêndice E.

## Os quadros coloridos

Ao longo do texto, você encontrará quadros com funções específicas.

:::exemplo Exemplo
Mostra um caso concreto, com números ou código.
:::

:::cotidiano No seu dia a dia
Liga o conceito a algo que você conhece.
:::

:::dica Dica
Sugestão prática para trabalhar melhor.
:::

:::atencao Atenção
Um cuidado importante, uma armadilha comum ou um limite do método.
:::

:::memorize Memorize
A ideia central, para fixar.
:::

:::saibamais Saiba mais
Um aprofundamento opcional.
:::

## Sobre os códigos e as bases de dados

Os códigos usam **Python** e **scikit-learn**. Todos os códigos e exercícios foram executados na elaboração do livro, com as versões de biblioteca mostradas na seção 1.12, e as saídas apresentadas são as obtidas nessas execuções. Pequenas diferenças numéricas podem ocorrer com outras versões.

Os exercícios usam bases do repositório **UCI Machine Learning** que já acompanham o scikit-learn (Iris, Wine, Breast Cancer Wisconsin e Digits), além de dados simulados. Assim, **tudo roda sem baixar nada**.

Ao final de cada capítulo, o quadro "Para ir além" lista outras bases públicas da UCI e do Kaggle para você praticar. Essas bases **não foram usadas nem testadas** nos códigos do livro: confira, nos sites, a descrição, a licença e as colunas antes de usá-las.

## Sobre as referências

As referências são artigos de periódicos e de conferências de referência da área, além de livros reconhecidos. Elas aparecem no texto no formato autor-data, e a lista completa está no final, em formato ABNT.

:::saibamais Nota sobre a conferência das referências
As referências foram preparadas a partir do conhecimento do autor da edição. Não foi possível consultar bases bibliográficas online durante a elaboração desta edição. Antes de citar em um trabalho acadêmico, confira volumes, páginas e DOI nas fontes originais.
:::

## Uma sugestão de estudo

Estude em blocos curtos. Leia uma seção, execute os códigos, mude um número e observe. Depois responda às questões do capítulo e resolva os exercícios práticos. Por fim, faça o projeto. **Errar faz parte**: cada resultado inesperado é uma pista sobre o seu método ou sobre os seus dados.

Bons estudos!

*Araken de Medeiros Santos*
