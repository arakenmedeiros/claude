# Apêndice K — Matrizes de decisão e de rastreabilidade

Este apêndice tem duas partes. A primeira ajuda a **escolher a estratégia** para uma tarefa: prompts, RAG, ajuste fino, contexto longo ou agentes. A segunda liga cada capítulo às **competências** que ele desenvolve e ao lugar onde você pode verificá-las.

:::cuidado Cuidado com esta interpretação
As classificações da matriz de decisão são **tendências gerais**, derivadas das discussões dos Capítulos 6, 11, 12, 14 e 15, e **não são medições**. Dependem do modelo, dos dados, do volume, dos preços vigentes e da sua tarefa. Nenhum experimento do livro comparou essas estratégias com LLMs reais: os experimentos usaram simuladores e modelos didáticos. Use a matriz para **organizar a conversa e escolher o que medir**, e decida com os seus números.
:::

## K.1 As cinco estratégias em uma frase

- **Prompts** (Capítulo 8): instruções, exemplos e formato bem escritos, sem alterar o modelo nem acrescentar fontes.
- **RAG** (Capítulo 11): recuperar trechos de uma base e incluí-los no prompt para ancorar a resposta.
- **Ajuste fino** (Capítulos 7 e 12): continuar o treinamento com dados próprios (por exemplo, com LoRA) para mudar comportamento, estilo ou formato.
- **Contexto longo** (Capítulos 4, 6 e 11): colocar muito material diretamente na janela de contexto, sem recuperação.
- **Agentes e fluxos** (Capítulo 14): combinar o modelo com ferramentas e etapas; fluxos fixos são mais previsíveis, agentes decidem os passos.

As estratégias **se combinam**: um sistema real pode usar prompt bem estruturado, RAG para o conhecimento, um modelo ajustado para o formato e ferramentas para ações.

## K.2 Matriz de decisão: conhecimento, custo, latência e controle

| Estratégia | Conhecimento atualizado | Custo | Latência | Controle sobre o comportamento |
| --- | --- | --- | --- | --- |
| Prompts | baixo: limitado ao que o modelo já traz e ao que você cola | baixo para começar; cresce com o tamanho do prompt, cobrado por token | baixa a média | médio: instruções não são garantias; formato validável por código |
| RAG | alto: atualiza-se o índice, sem retreinar | médio: índice, recuperação e mais tokens por consulta | média: acrescenta a etapa de busca e prompts maiores | médio a alto sobre as fontes (citações, permissões); o texto gerado segue probabilístico |
| Ajuste fino | baixo: o conhecimento fica congelado na data do treino do ajuste | alto no início (dados, treino, avaliação); pode reduzir o custo por consulta ao permitir prompts curtos ou modelo menor | baixa, se o prompt ficar curto | alto sobre estilo e formato; baixo sobre fatos novos; risco de esquecimento |
| Contexto longo | médio a alto: o que você inclui na janela | alto por consulta: paga-se todos os tokens da janela a cada chamada | média a alta: o preenchimento cresce com a entrada | baixo a médio: o modelo pode usar mal o meio do contexto; sem citações automáticas |
| Agentes e fluxos | alto, se as ferramentas buscam dados atuais | alto e variável: várias chamadas; o custo de entrada cresce de forma quadrática com os passos | alta e variável | fluxos: alto; agentes: baixo a médio (decidem os passos, e isso gera variabilidade) |

## K.3 Matriz de decisão: risco, esforço e avaliação

| Estratégia | Risco principal | Esforço de construção e manutenção | Dificuldade de avaliação |
| --- | --- | --- | --- |
| Prompts | alucinação; injeção de prompt; dependência de um serviço e de sua versão | baixo; exige versionar e testar prompts | baixa a média: conjunto de testes e métricas por tarefa |
| RAG | recuperação errada com citação convincente; documentos hostis ou desatualizados; vazamento de acesso | médio a alto: curadoria, fragmentação, índice, permissões, atualização | média a alta: avaliar recuperação, geração, fidelidade e abstenção **separadamente** |
| Ajuste fino | memorização e vazamento de dados pessoais; viés; esquecimento; licenças | alto: dados, treino, mesclagem, reversão, reajuste a cada novo modelo-base | média a alta: avaliação antes e depois, por subgrupo, e na tarefa original |
| Contexto longo | custo; informação ignorada no meio; exposição de mais dados a cada chamada | baixo para começar | média: medir em perguntas cuja resposta está em posições diferentes |
| Agentes e fluxos | ações indevidas; injeção indireta por saída de ferramenta; laços e custo descontrolado | alto: ferramentas, segurança em camadas, auditoria | alta: avaliar a tarefa inteira, trajetórias, custo dos erros |

## K.4 Como decidir, passo a passo

1. **Comece pelo mais simples.** Um prompt bem escrito, com formato validável e conjunto de testes, é a linha de base de todas as outras estratégias (Capítulos 8 e 12).
2. **Falta conhecimento?** Se a informação é privada, muda com frequência ou precisa ser **citada**, considere RAG (Capítulo 11). Se a base é pequena e o custo por chamada for aceitável, **meça** contexto longo contra RAG.
3. **Falta comportamento?** Se o modelo tem o conhecimento, mas erra o **estilo, o formato ou a política de resposta** de modo estável e com volume que justifique o investimento, considere ajuste fino (Capítulo 12). Ele **não** é um bom caminho para colocar fatos novos e voláteis.
4. **Falta ação?** Se a tarefa exige consultar sistemas, calcular ou executar, use ferramentas. Prefira **fluxos determinísticos**; use agentes só se os passos não puderem ser fixados antes (Capítulo 14).
5. **Calcule o custo.** Estime tokens de entrada e saída, volume e latência com **os seus** preços, conferidos na documentação (Capítulos 11 e 15).
6. **Meça.** Compare as alternativas no **mesmo** conjunto de testes, com intervalos de confiança e diferenças pareadas (Capítulo 16).
7. **Avalie o risco.** Dados pessoais, ações com efeito, impacto em pessoas: aplique os checklists do Apêndice J e a governança do Capítulo 17.

## K.5 Quando evitar cada estratégia

| Estratégia | Evite quando | Em vez disso |
| --- | --- | --- |
| Prompts | a resposta exige fatos privados ou muito recentes que você não consegue colar com segurança | RAG, ou ferramentas que buscam os dados |
| RAG | a base é ruim, desatualizada ou sem donos; ou o problema é de formato e estilo, não de conhecimento | curadoria antes de indexar; prompt ou ajuste fino |
| Ajuste fino | o conhecimento muda toda semana; faltam dados limpos, licenciados e sem dados pessoais; não há conjunto de teste | RAG e prompts; reúna e documente dados primeiro |
| Contexto longo | o custo por chamada é crítico; a informação relevante é pequena diante do material; há dados sensíveis desnecessários | RAG com fragmentos selecionados |
| Agentes | uma sequência fixa resolve; ações são irreversíveis sem supervisão; não há como auditar | fluxo determinístico com confirmação humana |

## K.6 Matriz de rastreabilidade: capítulos e competências

As competências abaixo são as que o livro pretende desenvolver. Cada uma é ligada aos objetivos de aprendizagem (`@obj`) dos capítulos.

| Código | Competência |
| --- | --- |
| C1 | Explicar o que são LLMs, o que fazem e o que não fazem; julgar quando usá-los |
| C2 | Usar a matemática e a computação básicas (vetores, probabilidade, entropia, gradiente, NumPy) |
| C3 | Representar texto: tokens, embeddings e similaridade |
| C4 | Explicar e calcular a atenção e a arquitetura Transformer |
| C5 | Descrever o treinamento e o pós-treinamento (dados, perda, SFT, preferências) |
| C6 | Entender a geração e a inferência (decodificação, contexto, cache, variabilidade) |
| C7 | Projetar, testar e versionar prompts |
| C8 | Construir aplicações robustas em Python, com provedores trocáveis |
| C9 | Construir e avaliar busca semântica e RAG |
| C10 | Decidir sobre e aplicar adaptação de modelos (LoRA, quantização) |
| C11 | Lidar com dados multimodais |
| C12 | Construir sistemas com ferramentas, fluxos e agentes |
| C13 | Estimar e reduzir custo e latência; planejar implantação |
| C14 | Avaliar modelos e aplicações com rigor estatístico |
| C15 | Identificar riscos e aplicar segurança, privacidade, ética e governança |
| C16 | Registrar, reproduzir e relatar experimentos com honestidade |

| Capítulo | Principais | Secundárias | Onde verificar |
| --- | --- | --- | --- |
| 1. Introdução aos LLMs | C1 | C2, C14, C15 | práticos 1.1 a 1.3 (n-gramas, linha de base, plausível não é verdadeiro); projeto: três formas de achar o documento certo |
| 2. Fundamentos matemáticos e computacionais | C2 | C16 | práticos de cosseno, softmax e entropia cruzada, regressão logística; projeto: mini-laboratório de otimização reprodutível |
| 3. Texto, tokens e embeddings | C3 | C2, C9 | práticos de BPE, embeddings LSA e normalização de acentos; projeto: embeddings por coocorrência |
| 4. Atenção e Transformer | C4 | C2, C6 | práticos de atenção com máscara, cabeças e ordem; projeto: custo da atenção |
| 5. Como os modelos aprendem | C5 | C2, C13, C16 | práticos de curvas de perda e de aprendizagem; projeto: auditoria de dados (deduplicação e contaminação) |
| 6. Geração e inferência | C6 | C4, C13, C16 | práticos de decodificação, latência e cache KV; projeto: painel de decisão de parâmetros de geração |
| 7. Ajuste por instruções e preferências | C5 | C15 | práticos de modelo de recompensa, DPO tabular e seleção entre n respostas; projeto: auditoria de um modelo de recompensa |
| 8. Engenharia de prompts | C7 | C14, C16 | práticos de avaliação de versões de prompt e tamanho de teste; projeto: classificador com validação, evidência e plano B |
| 9. Aplicações em Python | C8 | C7, C15, C16 | práticos de recuo exponencial, mascaramento e memória em GPU; projeto: aplicação em camadas com provedor simulado |
| 10. Embeddings e busca semântica | C9, C3 | C14 | práticos de normalização, segmentação e RRF; projeto: buscador com recusa calibrada e relatório com incerteza |
| 11. RAG | C9 | C7, C13, C14, C15 | práticos de recuperação, limiar de abstenção e injeção; projeto: assistente de normas com avaliação completa |
| 12. Ajuste fino, LoRA e QLoRA | C10 | C5, C14, C15 | práticos de posto, escala e quantização; projeto: ajuste com mitigação de esquecimento |
| 13. Modelos multimodais | C11 | C3, C14, C15 | práticos de patches, temperatura e STFT; projeto: busca imagem-texto com abstenção e avaliação por fatia |
| 14. Ferramentas, fluxos e agentes | C12 | C8, C14, C15 | práticos de registro de ferramentas, limite de passos e fluxo versus agente; projeto: agente com guardrails e avaliação |
| 15. Eficiência, custo e implantação | C13 | C6, C10 | práticos de orçamento, quantização e cache semântico; projeto: plano de capacidade, custo e metas |
| 16. Avaliação de LLMs e de aplicações | C14 | C7, C9, C16 | práticos de intervalos de Wilson, viés do juiz e comparação pareada; projeto: bancada de avaliação com relatório automático |
| 17. Segurança, ética e governança | C15 | C8, C9, C12, C14 | práticos de ataques e camadas, detector de dados pessoais e equidade; projeto: relatório de risco e bateria de segurança |
| 18. Projetos integradores | C1 a C16 (integração) | todas | critérios de aceitação de cada projeto, no próprio Capítulo 18 |

Todo capítulo de 1 a 17 tem **30 questões de múltipla escolha, 20 discursivas, três exercícios práticos e um projeto**; as questões de programação e os práticos têm solução de referência nos Apêndices.

## K.7 Como usar a matriz de rastreabilidade

- **Estudante:** marque as competências em que você ainda não resolveu os práticos sem consultar a solução; volte aos capítulos principais dessa competência.
- **Docente:** monte uma disciplina escolhendo as competências desejadas e os capítulos que as desenvolvem; use o projeto de cada capítulo ou os projetos integradores como avaliação.
- **Equipe:** use a coluna "Onde verificar" para propor atividades de capacitação, e o Apêndice J para acompanhar a prática.
