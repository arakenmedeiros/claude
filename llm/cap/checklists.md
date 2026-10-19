# Apêndice J — Listas de verificação

Listas para usar **antes, durante e depois** de construir uma aplicação com LLMs. Cada item é **verificável**: dá para responder "sim" ou "não" mostrando um arquivo, um número, um teste ou um registro. Os números entre parênteses indicam o capítulo em que o assunto é tratado.

:::cuidado Cuidado com esta interpretação
Uma lista de verificação **não garante** qualidade, segurança nem conformidade legal. Ela reduz esquecimentos. Adapte os itens ao seu contexto, acrescente os seus e **revise com especialistas** (segurança, jurídico, privacidade, domínio) quando houver risco. Itens de natureza legal e regulatória mudam: confira a fonte oficial vigente.
:::

## J.1 Avaliação (Capítulos 8, 10, 11, 16)

- [ ] A tarefa, o critério de sucesso e o custo de cada tipo de erro estão escritos (16).
- [ ] Há uma linha de base simples (regra, busca por palavras, modelo menor) medida com a mesma régua (12, 16).
- [ ] O conjunto de teste é representativo, estratificado e inclui casos difíceis e **sem resposta** (10, 16).
- [ ] O tamanho do conjunto foi calculado para a precisão desejada, e não escolhido por conveniência (16).
- [ ] Teste e desenvolvimento estão separados; o teste não foi usado para ajustar prompts, limiares ou hiperparâmetros (10, 16).
- [ ] Verificou-se contaminação: os dados de teste não aparecem no treino, no índice nem nos exemplos do prompt (5, 16).
- [ ] As métricas combinam com o problema (por exemplo, F1 macro para classes desbalanceadas; EM e F1 de tokens para respostas curtas) (16).
- [ ] Todos os resultados vêm com **intervalo de confiança** (bootstrap ou Wilson); comparações usam diferenças **pareadas** (8, 16).
- [ ] Declarou-se vencedor apenas quando a diferença supera a incerteza e a margem de relevância definida antes (16).
- [ ] Ao fazer muitas comparações, aplicou-se correção para múltiplas comparações (16).
- [ ] Resultados são reportados por fatia (tipo de pergunta, grupo, idioma), e não só em média (16, 17).
- [ ] A variabilidade foi medida: várias sementes, a temperatura de produção e a ordem de exemplos (6, 16).
- [ ] Se há juiz automático: rubrica explícita, troca de ordem, controle de tamanho, juiz de outra família e concordância com humanos medida em amostra rotulada (16).
- [ ] O cartão de avaliação registra versões, sementes, data, conjunto de teste, intervalos e limitações (16).

## J.2 Segurança e privacidade (Capítulos 9, 11, 14, 17)

- [ ] Existe um modelo de ameaças com ativos, atores, canais de entrada e impactos (17).
- [ ] Toda entrada de usuário, documento recuperado e saída de ferramenta é tratada como **não confiável** (11, 14, 17).
- [ ] Instruções e dados estão separados por delimitadores **e** há sanitização de sequências que imitam os delimitadores (8, 11).
- [ ] O prompt de sistema **não contém segredos** e é tratado como legível (17).
- [ ] Nenhuma credencial está no código, nos registros ou no repositório; há varredura automática e plano de rotação e revogação (9).
- [ ] Ferramentas seguem o **menor privilégio** por papel; argumentos são validados em código; ações externas ou destrutivas exigem confirmação humana, com padrão "negar" (14, 17).
- [ ] Há filtro de saída para segredos, links e dados pessoais (17).
- [ ] Há limites de pedidos, tokens, tempo e orçamento por usuário (9, 17).
- [ ] Dados pessoais são mascarados antes de registro ou envio a serviços externos; o mascaramento por regras foi testado e seus limites foram documentados (9, 17).
- [ ] A coleta segue a **minimização**: só o necessário à finalidade; retenção e exclusão estão definidas (11, 17).
- [ ] Há uma bateria de **red teaming** com variantes e repetições, taxa de sucesso por objetivo com intervalo, e reteste a cada mudança (17).
- [ ] Dependências estão fixadas e verificadas; formatos de modelo que executam código foram evitados; hashes de arquivos de pesos conferidos (17).
- [ ] Registros de auditoria são imutáveis ou encadeados e mascaram dados pessoais (14).
- [ ] Há plano de incidentes com responsáveis nomeados (17).

## J.3 RAG (Capítulos 10 e 11)

- [ ] A necessidade de RAG foi justificada frente a prompt melhor e a ajuste fino, com medição (11, 12).
- [ ] Cada documento tem **procedência**, versão, vigência e dono; documentos desatualizados ou conflitantes têm tratamento (11, 17).
- [ ] Fragmentação, tamanho e sobreposição foram escolhidos por medição na sua base, e cada fragmento guarda metadados (10, 11).
- [ ] Os embeddings foram normalizados, e o modelo de embeddings e a métrica estão registrados; trocar o modelo implica reindexar (3, 10).
- [ ] A recuperação foi avaliada com recall@k, MRR ou nDCG, com intervalo, **por tipo de consulta** (10, 11).
- [ ] Busca híbrida, reescrita, multiconsulta e reranqueamento só entram se **melhoraram** na sua métrica (10, 11).
- [ ] Há limiar de abstenção calibrado em dados à parte; mede-se a abstenção correta **e** a indevida (10, 11).
- [ ] O contexto tem delimitadores, citações, limite de tokens e política de corte por fragmentos inteiros (11).
- [ ] As citações são verificadas por código: o id existe e a afirmação aparece no texto citado (11).
- [ ] Recuperação, geração, fidelidade e abstenção são avaliadas **separadamente** (11, 16).
- [ ] **Permissões** por usuário são aplicadas **antes** do modelo, no código, e testadas com usuários de papéis diferentes (11).
- [ ] Há defesa em camadas contra injeção via documentos e controle de escrita no índice (11, 17).
- [ ] Atualização e **exclusão** de documentos do índice são testadas (o documento removido não aparece mais) (11).
- [ ] Custo e latência por consulta foram estimados em função de k e do tamanho dos fragmentos (11, 15).

## J.4 Agentes e ferramentas (Capítulo 14)

- [ ] O problema exige agente? Um fluxo determinístico mais simples foi considerado primeiro (14).
- [ ] Cada ferramenta tem esquema JSON, validação (por exemplo, `pydantic`), mensagens de erro legíveis e **tempo limite** (9, 14).
- [ ] Ferramentas com efeito têm **chave de idempotência** (14).
- [ ] O laço tem paradas: resposta final, limite de passos, orçamento, detecção de repetição e escalonamento a humano (14).
- [ ] As chamadas truncadas ou recusadas nunca são executadas (14).
- [ ] O modelo só vê as ferramentas permitidas para aquela tarefa e aquele usuário (14, 17).
- [ ] Saídas de ferramentas entram no contexto como dado delimitado, não como instrução (14).
- [ ] Ações irreversíveis ou externas exigem confirmação humana; o padrão é negar (14).
- [ ] Execução de código ou comandos ocorre em **sandbox**, com limites de tempo e memória (14).
- [ ] Toda ação é registrada em trilha de auditoria com dados pessoais mascarados (14).
- [ ] A avaliação é por **tarefa inteira**: taxa de sucesso com intervalo (bootstrap sobre tarefas), número de passos, custo e trajetórias (14).
- [ ] O **custo dos erros** foi considerado, e não apenas a taxa de acerto (14).
- [ ] Ferramentas de terceiros (por exemplo, via MCP) foram revisadas com as mesmas regras de segurança (14).
- [ ] Memória de longo prazo tem política de retenção, correção e exclusão (14, 17).

## J.5 Ajuste fino (Capítulo 12)

- [ ] O problema não se resolve com prompt melhor ou RAG; a linha de base foi medida (12).
- [ ] Há volume e estabilidade de uso que justifiquem o ajuste; a conta de custo foi feita (12, 15).
- [ ] Os dados foram deduplicados, não contêm dados pessoais, e têm origem e licença registradas (5, 12, 17).
- [ ] A divisão treino, validação e teste não tem vazamento, e o teste está trancado (12, 16).
- [ ] As licenças do modelo-base e dos dados permitem o uso pretendido (12, 17).
- [ ] Hiperparâmetros, sementes e versões de dados, modelo-base e adaptador estão registrados (2, 12).
- [ ] A perda é calculada apenas nos tokens da resposta, quando for o caso (7, 12).
- [ ] A avaliação antes e depois é **pareada**, com intervalo, por subgrupo e **também na tarefa original** (para detectar esquecimento) (12, 16).
- [ ] A mesclagem de adaptadores foi verificada numericamente; se houve requantização, ela foi avaliada (12).
- [ ] Há plano de reversão (adaptador separado) e de monitoramento após a implantação (12, 15).
- [ ] O modelo ajustado tem cartão de modelo com uso pretendido, dados e limitações (12, 17).

## J.6 Multimodal (Capítulo 13)

- [ ] Para cada modalidade, definiu-se o formato de entrada, a conversão em tokens ou embeddings e o custo em tokens (13).
- [ ] Há avaliação **por fatia** (tipo de imagem, qualidade de áudio, idioma, layout) com intervalos (13).
- [ ] Testou-se a **alucinação visual**: perguntas sobre detalhes que não estão na imagem (13).
- [ ] No OCR, os erros silenciosos foram medidos e há validação (por exemplo, soma conferida, formato de datas) e revisão humana nos casos de risco (13).
- [ ] Em fala para texto, o erro foi medido na sua variedade de português, e a propagação para as etapas seguintes foi avaliada (13).
- [ ] Imagens, áudios e documentos com dados pessoais têm base legal, minimização, retenção e anonimização definidas (13, 17).
- [ ] Texto alternativo e transcrições gerados foram **revisados** por pessoas antes de uso em acessibilidade (13).
- [ ] Imagens geradas têm política de rotulagem, direitos autorais e uso indevido (13, 17).
- [ ] O custo e a latência por item foram medidos na sua configuração (13, 15).

## J.7 Implantação (Capítulos 9 e 15)

- [ ] Nomes de modelos, versões e parâmetros de geração vêm de configuração, e não de constantes espalhadas (9).
- [ ] A lógica da aplicação está separada do provedor (portas e adaptadores), e há testes com provedor simulado (9).
- [ ] Há tempo limite global, nova tentativa com recuo exponencial e jitter **só** para erros transitórios, e disjuntor (9).
- [ ] Há plano B para falhas do serviço e limites de taxa respeitados (9, 15).
- [ ] O custo foi modelado com **os seus** preços e volumes, conferidos na documentação oficial, e há orçamento por requisição e por período (9, 15).
- [ ] Latência é acompanhada por **percentis** (p95, p99), e não só pela média; há SLO e orçamento de erro (15).
- [ ] Caches não alteram resultados indevidamente; o cache semântico teve o erro medido e as entidades são verificadas (15).
- [ ] A quantização ou o modelo menor foi avaliado **na sua tarefa**, e não só em benchmarks (15, 16).
- [ ] Há testes de carga, canário, versionamento de prompt, modelo e índice, e **reversão** ensaiada (15).
- [ ] Observabilidade: registros estruturados sem o texto do usuário, métricas, rastros e alertas (15).
- [ ] Monitora-se deriva (por exemplo, PSI sobre intenções) e qualidade em produção, com revisão por amostragem (15, 16).
- [ ] A regressão contínua roda a cada mudança, com portão de liberação (16).
- [ ] Estimou-se o consumo de energia e consideraram-se alternativas mais eficientes (15).

## J.8 Governança (Capítulo 17)

- [ ] Existe um **inventário** de sistemas de IA, com finalidade, dados, modelos, fornecedores e responsáveis (17).
- [ ] Cada sistema tem classificação de risco e papéis definidos (quem constrói, quem aprova, quem opera, quem audita) (17).
- [ ] A base legal e a minimização foram definidas com o encarregado; foi avaliada a necessidade de relatório de impacto (RIPD) (17).
- [ ] As obrigações de transparência foram atendidas: o usuário sabe que interage com IA, conhece limites e sabe como pedir revisão humana (17).
- [ ] Há supervisão humana definida para decisões de maior impacto, e canal de contestação (17).
- [ ] Equidade foi avaliada por grupo, com intervalos, e a métrica escolhida foi justificada (17).
- [ ] Direitos autorais e licenças de dados e de modelos foram revisados com apoio jurídico (12, 17).
- [ ] Há registro de incidentes, aprendizado e prazos de correção (17).
- [ ] Há auditoria periódica e reteste a cada mudança relevante (16, 17).
- [ ] A conformidade com leis e normas aplicáveis (LGPD e demais) foi **confirmada na fonte oficial vigente** e com profissionais habilitados; este livro não substitui aconselhamento jurídico (17).
- [ ] Uso em saúde, jurídico e administração pública tem supervisão profissional, sem aconselhamento clínico ou decisão automática sem revisão (1, 17).

## J.9 Projetos integradores (Capítulo 18)

- [ ] Os **critérios de aceitação** (número e intervalo que dizem "pronto") foram escritos **antes** de medir, e o relatório os compara por **veredito**: atende, não atende ou inconclusivo (16, 18).
- [ ] Existe um **conjunto cego**, separado do de desenvolvimento, que não foi usado para escrever regras nem ajustar limiares, e o resultado dele aparece ao lado do de desenvolvimento (16, 18).
- [ ] Os **estratos críticos** (urgência, ataques, perguntas sem resposta, grupos de usuários) são medidos à parte, cada um com **IC e veredito**, e não só na média (16, 18).
- [ ] Se o sistema usa normas ou documentos com datas, há **teste de fronteira de vigência** (o dia exato de início e de fim) e a resposta informa a norma e a data de referência (11, 18).
- [ ] Em serviços de saúde, a resposta a **urgência** e o encaminhamento a profissional são **texto fixo**, nunca gerado, e a revocação de urgência é medida no conjunto cego, com informação de que o sistema não faz aconselhamento clínico (17, 18).
- [ ] Mudanças de prompt, limiar, modelo ou base passam por um **portão de regressão** com **comparação pareada** com a versão aprovada, nos mesmos casos (16, 18).
- [ ] Cada resultado guarda o **hash dos casos e da configuração**, sementes e versões, no **registro de reprodutibilidade** (16, 18).
- [ ] Diferenças entre subgrupos só são atribuídas ao grupo depois de checar **confundimento** com outro fator (por exemplo, tipo de pergunta) (17, 18).
- [ ] A trilha de auditoria é íntegra (hash encadeado), o usuário aparece **pseudonimizado** e há retenção definida (17, 18).
- [ ] O relatório declara o que o simulador **não** permite afirmar e que os números não valem para modelos ou produtos reais (18).

