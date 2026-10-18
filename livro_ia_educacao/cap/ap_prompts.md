# Apêndice E — Banco de prompts comentados

Os modelos abaixo são pontos de partida. Substitua os trechos entre colchetes pelas informações do seu contexto, **sem incluir dados pessoais de estudantes**. Sempre leia, verifique e adapte a resposta antes de usá-la. A estrutura geral segue a anatomia do pedido apresentada no Capítulo 24: papel, contexto, tarefa, formato e critérios.

## Planejamento

:::prompt Rascunho de plano de aula
Você é um(a) assistente de planejamento para professores de [componente curricular], [ano/etapa]. Contexto: turma de [número] estudantes, [duração] de aula, recursos disponíveis: [lista]. Objetivo de aprendizagem: [objetivo]. Tarefa: proponha um roteiro de aula com abertura, desenvolvimento, atividade prática e fechamento, indicando o tempo de cada parte. Formato: tabela com colunas Momento, Tempo, Ação do professor, Ação dos estudantes. Critérios: linguagem adequada à idade, uma alternativa para quem não tem acesso a dispositivos, uma pergunta de verificação da aprendizagem. Aponte também o que eu devo conferir antes de usar.
:::

*Comentário:* o pedido informa objetivo, tempo e recursos; a última frase pede que a ferramenta indique pontos de checagem, o que reforça a revisão docente.

:::prompt Sequência didática em etapas
Atue como apoio à elaboração de uma sequência didática de [número] aulas sobre [tema] para [ano/etapa]. Parta das seguintes habilidades/objetivos: [lista]. Para cada aula, apresente objetivo, atividade central, produto e forma de avaliação formativa. Inclua uma aula de retomada e uma tarefa final integradora. Evite exemplos com marcas comerciais.
:::

## Materiais

:::prompt Texto adaptado por nível
Adapte o texto abaixo para estudantes do [ano/etapa], mantendo os conceitos centrais e sem acrescentar informações novas. Use frases curtas, explique termos difíceis entre parênteses e termine com três perguntas de compreensão. Texto: [cole aqui um texto que você tem direito de usar e que não contém dados pessoais].
:::

:::prompt Banco de exercícios com gabarito comentado
Crie [número] exercícios de [tema] para [ano/etapa], com dificuldade crescente. Para cada um, apresente o gabarito e uma explicação em duas frases do raciocínio. Indique qual erro comum cada exercício ajuda a detectar.
:::

*Comentário:* resolva os exercícios você mesmo(a) antes de aplicá-los: erros em gabaritos gerados por IA são frequentes.

## Avaliação e feedback

:::prompt Rubrica analítica
Elabore uma rubrica analítica para avaliar [produto] de estudantes do [ano/etapa], com [número] critérios e 4 níveis de desempenho. Descreva cada nível em linguagem observável e positiva. Os critérios são: [lista]. Não inclua critérios sobre comportamento ou caligrafia.
:::

:::prompt Devolutiva a partir de critérios
Com base na rubrica a seguir e no trecho de trabalho anonimizado, escreva uma devolutiva em três partes: um ponto forte, um aspecto a melhorar e um próximo passo concreto. Use tom encorajador e linguagem simples. Rubrica: [...]. Trecho: [sem nome nem dados pessoais].
:::

*Comentário:* a devolutiva gerada é um rascunho. O professor decide o que dizer ao estudante.

## Acessibilidade e organização

:::prompt Versão em linguagem simples
Reescreva o comunicado a seguir em linguagem simples, com frases curtas, sem siglas e com os pontos principais em lista. Mantenha todas as informações e as datas. Texto: [...].
:::

:::prompt Descrição de imagem para texto alternativo
Escreva uma descrição objetiva, com até duas frases, de uma imagem usada em aula, que sirva de texto alternativo para estudantes cegos. A imagem mostra: [descreva com suas palavras]. Destaque o que é relevante para o objetivo [objetivo].
:::

## Verificação e pensamento crítico

:::prompt Pedir a fonte e o grau de certeza
Responda à pergunta a seguir e, ao final, liste as afirmações que podem estar incorretas ou desatualizadas e que devo verificar em fontes primárias. Se não tiver certeza, diga isso claramente em vez de supor. Pergunta: [...].
:::

:::prompt Debate de estudantes com a resposta da IA
Gere uma explicação sobre [tema] contendo dois erros sutis, sem indicar quais. Depois, entregue a chave dos erros em uma seção separada. Uso: atividade de caça aos erros com a turma.
:::

## Boas práticas ao usar este banco

- Teste cada prompt com um tema simples antes de usá-lo em um planejamento real.
- Guarde as versões que funcionam bem para você; registre data e ferramenta.
- Compare respostas de pedidos ligeiramente diferentes; refine em etapas.
- Nunca inclua nomes, imagens, laudos ou informações de saúde de estudantes.
