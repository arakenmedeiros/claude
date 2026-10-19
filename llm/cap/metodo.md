# Apêndice L — Nota metodológica e guia de reprodução

Este apêndice descreve **como o livro foi produzido**, o que foi **verificado** e o que **não foi**, quais são os **limites** dos experimentos e como **reproduzir** o material. Foi escrito para ser lido com desconfiança saudável: onde algo não foi feito, está dito.

## L.1 Data de corte e escopo

A revisão bibliográfica e a verificação de informações voláteis foram feitas **até 18 de outubro de 2026**. Fatos posteriores a essa data não foram incorporados. O livro evita depender de modelos, preços e interfaces específicos; quando os cita, manda **conferir a documentação oficial**. As contagens deste apêndice referem-se aos **capítulos 1 a 18** na data do corte; o Capítulo 18 (projetos integradores) segue as mesmas regras e está incluído nelas.

## L.2 O que foi consultado de Ozdemir (2024; 2025)

O livro de Ozdemir é a **referência orientadora de escopo**: serviu para decidir quais grandes temas um livro didático introdutório sobre LLMs deveria cobrir (fundamentos, embeddings, prompts, ajuste fino, multimodalidade). Declaramos com clareza:

- **O conteúdo do livro de Ozdemir NÃO foi lido na íntegra.** Não tivemos acesso ao texto completo. Usamos apenas a **descrição editorial pública** (título, subtítulo e escopo declarado) como guia.
- **Nenhuma afirmação técnica deste livro é atribuída a Ozdemir.** A seleção de marcos, as explicações, os exemplos, o código e os exercícios são **nossos**; as afirmações técnicas vêm dos artigos e das documentações citados, que o leitor pode consultar.
- O status das duas entradas na lista de referências reflete isso: a edição em inglês (2024) consta como **V**, porque a existência e os dados bibliográficos foram conferidos online; a edição em português (2025) consta como **P**, ou seja, dado fornecido pelo solicitante e **não localizado** na verificação. O status **V** atesta a conferência dos **dados bibliográficos**, e não a leitura da obra.
- Os capítulos não reproduzem nem parafraseiam capítulos dessa obra. Se houver semelhança de temas, ela decorre do escopo comum do assunto.

## L.3 Status de verificação das referências

Cada referência tem um status no arquivo `cap/referencias.txt`:

- **V**: existência e dados bibliográficos **conferidos online** na revisão.
- **C**: **conferida pela memória** do autor e ainda **a verificar na fonte**.
- **P**: dado fornecido pelo solicitante, **não localizado** na verificação.

Na data do corte, a lista continha **151 entradas**: **149 V, 1 C e 1 P** (C: McCloskey; Cohen, 1989, sobre interferência catastrófica; P: a edição brasileira de Ozdemir, 2025). Das entradas efetivamente citadas nos capítulos 1 a 18 (147; o Capítulo 18 só reutiliza referências já citadas), 145 são V, 1 é C e 1 é P. No livro impresso, as entradas que não são V trazem o sinal †, que significa: **confira na fonte original antes de depender do dado**. Para recontar:

```bash
cut -d'|' -f2 cap/referencias.txt | tr -d ' ' | sort | uniq -c
```

(a linha de cabeçalho do arquivo é contada como "status" e deve ser descontada). **Atenção:** o status V garante que a obra existe e que seus dados bibliográficos foram conferidos, **não** que cada afirmação do livro sobre ela foi relida na íntegra. Afirmações atribuídas a artigos foram escritas com cautela e remetem ao texto original para verificação; leia a fonte antes de citar números ou conclusões.

## L.4 O que foi executado de fato

Os experimentos foram executados em um ambiente com **Python 3.13.16** e as bibliotecas do Apêndice H (por exemplo, NumPy 2.5.3 e scikit-learn 1.9.1), **sem GPU e sem acesso a serviços de LLM**. Contagens nos capítulos 1 a 18, a partir dos registros de execução em `build/code_out.json`, `build/ex_ok.json` e `build/pr_out.json`:

| Item | Quantidade | Situação |
| --- | --- | --- |
| Blocos de código do texto, `python` (35) e `python+` (174) | 209 | executados; as saídas reais foram registradas e exibidas no livro (208 produzem saída impressa) |
| Códigos de questões discursivas de programação (`CODE:`) | 78 | executados; todos terminaram sem erro |
| Resultados de exercícios práticos | 54 exercícios e 18 projetos (72 no total) | executados; as notas de discussão devem citar apenas números realmente impressos |
| Questões | 900 (540 de múltipla escolha e 360 discursivas) | escritas e validadas quanto ao formato (30 e 20 por capítulo); **não** foram aplicadas a estudantes |
| Figuras dos capítulos | 140 | 71 baseadas em dados calculados pelos códigos e 69 conceituais (esquemas) |
| Glossário | 395 termos | redigidos para o livro |

**O que NÃO foi executado:**

- Os **14 blocos `python-ni`** (adaptadores de API e de modelos locais, `transformers` e `peft`, modelos multimodais reais, servidores de inferência, e, no Capítulo 18, adaptadores de API real para o gerador, o agente com ferramentas e a extração estruturada) e **2 blocos `ilustrativo`**. Foram escritos com base na documentação oficial e **não foram testados**.
- Qualquer **serviço real** de LLM ou de busca, e qualquer **modelo real** (nenhum peso foi baixado).
- Qualquer uso de **GPU**, e qualquer medição de tempo, custo ou consumo em hardware real. Preços, hardware e tempos dos capítulos 6 e 15 são **hipotéticos** e estão marcados como tal.
- Não houve **teste com usuários** nem **revisão por pares externos** do texto.
- O uso em Google Colab, Windows e macOS **não foi testado**.

Um verificador de formato (`build/chk.py`) confere a estrutura das questões, as referências e as figuras citadas; ele **não** confere se cada número do texto coincide com a saída do código. Essa conferência é feita pela leitura das saídas registradas, e **pode ter falhas**: se você encontrar um número no texto que não bate com a saída do código, confie na saída e avise o autor.

## L.5 Declaração: simuladores e modelos didáticos, nenhum LLM real

**Todo experimento do livro que menciona um "LLM", "assistente", "modelo", "juiz" ou "agente" usou simuladores baseados em regras ou modelos didáticos em NumPy. Nenhum LLM real foi usado em nenhum experimento.** Os principais componentes, todos no pacote `llmdidatico`:

- `SimuladorLLM` (Capítulo 8): simulador de regras que imita o efeito de três versões de prompt; **não é um LLM**.
- `NgramaLM` e `MLPLM`: modelos de linguagem minúsculos (n-gramas e uma rede com retropropagação escrita à mão), treinados em corpora sintéticos.
- `GeradorSimulado` (Capítulo 11): lê o prompt, mede a cobertura das palavras da pergunta e responde ou se abstém por regras.
- `ModeloSimulado` e `ModeloRoteirizado` (Capítulo 14): "modelos" que decidem chamadas de ferramentas por regras.
- `JuizEnviesado` (Capítulo 16): "juiz" com vieses de posição, tamanho e estilo **programados**.
- `AssistenteSimulado` (Capítulo 17): assistente com probabilidades de obediência **programadas**, para demonstrar ataques e defesas.
- Módulos dos projetos do Capítulo 18 (`projetos`, `saude_sintetica`, `normas_sinteticas`, entre outros): gerador, classificador, agente e extrator **por regras** (por exemplo, `TriagemSaude`, `ConsultaNormas`, `ExtratorSimulado` e `SistemaIntegrado`), com casos escolhidos por nós; preços e latências são **hipotéticos**.
- Ferramentas multimodais em NumPy (Capítulo 13), `MLPLMLoRA` (Capítulo 12), `IndiceIVF` e `IndiceLSH` (Capítulo 10), `BM25` e `LSAEmbedder`.

Consequência importante: **os efeitos observados nesses experimentos foram, em boa parte, escritos por nós**. Eles demonstram **procedimentos** (como medir, como calcular intervalos, como detectar um problema), e não o comportamento de modelos reais. Não generalize os números para produtos ou modelos reais.

## L.6 Limites dos experimentos

- **Corpora sintéticos em português**, gerados por regras com semente fixa: o vocabulário e a variedade são muito menores que os da língua real.
- **Base fictícia** (60 documentos e 56 consultas da "Universidade Federal Aurora"): prazos, valores e regras são inventados. Resultados de recuperação nessa base **não** se transferem a bases reais.
- **n pequeno em várias avaliações** (por exemplo, subconjuntos de poucas consultas): os intervalos de confiança são largos, e o livro os mostra para que isso fique visível. Diferenças dentro da incerteza **não** são evidência de diferença real.
- **Sementes fixas**: garantem repetição, mas a variação entre sementes precisa ser medida quando a conclusão depender dela (Capítulo 16).
- **Modelos minúsculos**: LoRA, quantização, atenção e difusão são demonstrados em escala de milhares de parâmetros; fenômenos de escala não aparecem.
- **Detectores e filtros por regras** (dados pessoais, injeção) são ilustrações, e **não** proteção suficiente.
- **Informações voláteis** (preços, limites, licenças, versões, regulação) mudam depois do corte.
- **Aconselhamento**: o livro não oferece aconselhamento jurídico, clínico nem de segurança; casos de saúde usam apenas dados públicos ou sintéticos.

## L.7 Como reproduzir o livro

O livro é gerado por um pipeline: `cap/*.md` e arquivos de apoio vão para `build/build.js`, que produz o DOCX. Pré-requisitos (Apêndice H): Python com as bibliotecas fixadas, Node.js, os pacotes `docx` e `katex`, Playwright com Chromium, LibreOffice e as ferramentas `pdfinfo` e `pdftotext`. Os scripts têm caminhos absolutos do ambiente de elaboração; ajuste-os.

```bash
cd build
# 1. executar os códigos e registrar as saídas reais (todos os capítulos; ou passe os números)
python3 runcode.py
python3 runex.py
python3 runpr.py
# 2. gerar figuras (uma por capítulo; saída em ../figs)
for f in ../codigo/figuras/figs_cap*.py; do python3 "$f"; done
# 3. verificar formato, questões, referências e figuras citadas
python3 chk.py
# 4. montar o documento (sumário com páginas calculadas em duas passagens)
bash make.sh
```

Observações: `make.sh` usa um script do LibreOffice cujo caminho depende da instalação; ele constrói o DOCX duas vezes e confere se a paginação ficou estável. Para um teste rápido, sem o processo completo, rode `NCH=1 node build.js /tmp/teste.docx`, que monta um capítulo e todos os apêndices e acusa erros de formato. A variável `NCH` indica o número de capítulos a incluir.

## L.8 Política de atualização

O que **muda rápido** e deve ser conferido na fonte oficial antes de qualquer decisão:

| Conteúdo volátil | O que fazer |
| --- | --- |
| Preços e cobrança de serviços | consultar a página de preços do provedor; refazer as contas do Capítulo 15 com os valores vigentes |
| Nomes e versões de modelos, parâmetros de API e limites | consultar a documentação da versão instalada; o livro lê o nome do modelo de variável de ambiente |
| Licenças de modelos e de dados | ler a licença do modelo e da versão específicos; pesos disponíveis não equivalem a modelo aberto |
| Versões de bibliotecas | usar o `requirements.txt` para reproduzir; ao atualizar, rodar de novo os códigos e conferir as saídas |
| Regulação (LGPD, regras do uso de IA, normas setoriais) | consultar o texto legal vigente e profissionais habilitados |
| Benchmarks e rankings | tratar como triagem; conferir contaminação e adequação à sua tarefa |

Em cada edição: (1) reexecutar os códigos e comparar as saídas; (2) reconferir as referências **C** e **P** e as marcadas com †; (3) revisar as passagens sobre conteúdo volátil; (4) registrar a nova data de corte neste apêndice.

## L.9 Como relatar erros

Se você encontrar um erro, uma referência que não confere ou uma saída que não reproduz, anote o capítulo, a seção, as versões das bibliotecas e o que esperava. A honestidade deste livro depende de correções.
