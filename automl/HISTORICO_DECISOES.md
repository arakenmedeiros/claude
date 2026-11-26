# Histórico resumido da conversa e das decisões

Complemento de `CONTINUIDADE.md`. É um resumo cronológico, não a transcrição literal.

## 1. Pedido inicial
- O usuário invocou as skills de professor de IA, de elaboração de livro didático e de exercícios práticos, com o tema **AutoML**.
- Respostas às perguntas prévias:
  - Autor: Araken de Medeiros Santos.
  - Público: misto, sem pré-requisitos.
  - Título: "AutoML: Teoria e Prática".
  - Extensão: sem limite.
  - Idioma e normas: português do Brasil, ABNT.
  - Estrutura: 14 capítulos em 5 partes e apêndices A–E, aprovada pelo usuário.
  - Referências: o usuário preferiu que a verificação fosse feita por mim.
  - Escrita: "escreva todo de uma vez".
- Requisitos por capítulo: 30 questões de múltipla escolha e 20 discursivas, gabarito no Apêndice A, exercícios práticos em Python/scikit-learn com bases públicas (UCI, Kaggle, OpenML) e um projeto. Projeto integrador no Cap. 14.

## 2. Entregas sucessivas
1. Livro completo em `.docx`; depois o usuário pediu o link de download (links do GitHub).
2. Slides: um `.pptx` por capítulo (14 decks), tom didático, analogias e tópicos curtos. Houve um "Tente novamente" do usuário após um reinício do worker durante a geração.
3. Revisão conforme o parecer colado pelo usuário, com inclusão de temas novos e saída em `.docx`, `.pdf` e `.epub` justificados.
4. `CONTINUIDADE.md` e este histórico, para retomada em outra sessão.

## 3. Parecer e como cada ponto foi tratado
- **Justiça (Cap. 13):** três partes (treino/validação/teste); limiares por grupo escolhidos na validação; paridade demográfica distinta de igualdade de oportunidade; variante calculada no teste rotulada como transdutiva.
- **Campeão e desafiante (Cap. 14):** análise pareada (bootstrap pareado e McNemar exato, melhoria mínima δ = 0,005); "ausência de evidência" não prova ausência de melhoria. Resultado: o desafiante do mês 6 não assume pela regra pareada (diferença +0,0137; IC [+0,0007; +0,0270]; p = 0,047).
- **Comparação de ferramentas (Cap. 10):** recursos iguais e medidos (1 núcleo, mesmo tempo, processo novo), `max_iter` no espaço de busca. Resultado: nenhum método claramente superior (Friedman, p = 0,222, com pouco poder em 4 bases).
- **TPOT:** exemplo identificado como série 0.x, não executado.
- **Caminhos absolutos:** substituídos por `resultados/` criada pelo código.
- **Série temporal (Cap. 12):** preenchimento causal (último valor observado) antes das janelas.
- **BOHB, SHAP, NAS:** BOHB descrito com precisão; SHAP em log-odds com valor-base; NAS com custos e variação de sementes separados.
- **Ambiente:** versões fixadas e `requirements.txt`.
- **Bibliografia:** Auto-WEKA, OpenML e Lorena completados; 12 referências novas, todas conferidas por busca; DOIs/URLs só onde confirmados.

## 4. Temas acrescentados
- Cap. 2: dados desbalanceados.
- Cap. 3: target encoding com cross-fitting.
- Cap. 6: ASHA e busca assíncrona.
- Cap. 10: modelos fundacionais tabulares (TabPFN).
- Cap. 12: multirrótulo e LLM/RAG.
- Cap. 13: calibração, predição conforme e abstenção.
- Cap. 14: AutoML com restrições.

## 5. Problemas encontrados e correções
- Pacotes Python ausentes após reinício do ambiente: reinstalados nas mesmas versões.
- Código inserido entre marcadores `#@nb`/`#@saida` removia o marcador: reposicionado.
- Barra vertical dentro de célula de tabela quebrava o `.docx`: texto reescrito.
- Aspas no texto alternativo de imagem quebravam o XHTML do EPUB: aspas passaram a ser escapadas.
- Laço do EX3 do Cap. 10 gerava NaN: reescrito.
- Numeração de figuras duplicada no Cap. 12: corrigida.
- RAG de brinquedo era trivial (recall 1,0): tornado difícil e texto ajustado aos números reais.
- Latências não determinísticas: o texto diz que variam por execução.
- Conversão `docx` → `epub` por pandoc falhava: EPUB gerado a partir de HTML próprio.

## 6. Restrições e preferências do usuário e do ambiente
- Não criar PR sem pedido.
- Push apenas em `claude/gifted-ritchie-i7q8s4`.
- Commits com os trailers de coautoria e de sessão da plataforma.
- Não incluir identificador de modelo em artefatos do repositório.
- Referências só com dados confirmados.
- Texto justificado.

## 7. Limitações declaradas ao usuário
- Auto-sklearn, TPOT, H2O, AutoGluon, Auto-Keras, TabPFN e FastAPI sem execução.
- Simulações de ASHA, NAS com ruído, restrições e RAG rotuladas como tais.
- Comparativo de ferramentas em apenas 4 bases pequenas.
- EPUB sem `epubcheck`; `.docx`/`.pdf` conferidos só por amostra.
- Slides dos Caps. 2, 3, 6 e 10–14 desatualizados.

## 8. Pergunta final sobre o Cowork
O usuário perguntou se era possível continuar no Cowork mantendo o histórico. Não foi possível confirmar nem transferir a sessão: nenhum ambiente do Cowork aparecia na conta, só o `Default`, e não há ferramenta para mover histórico entre sessões. A alternativa adotada foi este histórico mais `CONTINUIDADE.md`. No Cowork, abrir sessão com acesso a `arakenmedeiros/claude`, branch `claude/gifted-ritchie-i7q8s4`, e pedir a leitura de `automl/CONTINUIDADE.md` e deste arquivo.
