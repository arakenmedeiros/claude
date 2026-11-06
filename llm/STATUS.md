# STATUS do projeto — "Grandes Modelos de Linguagem" (Araken de Medeiros Santos)

Data: 19/out/2026 · Corte bibliográfico: 18/out/2026

## Título
Considerados: (1) *Grandes Modelos de Linguagem: do básico à aplicação*; (2) *Entendendo e Construindo com LLMs*; (3) *LLMs na Prática: fundamentos, aplicações e responsabilidade*. Adotado na capa/arquivo: **Grandes Modelos de Linguagem** (subtítulo: "Dos fundamentos aos sistemas, com Python").

## Entregue
- `Grandes_Modelos_de_Linguagem.docx` (946 páginas, sumário como campo Word com páginas verificadas em duas passagens: "PAGINAS ESTAVEIS") e cópia em PDF.
- 18 capítulos (`cap/chNN.md`), 18 conjuntos de 30 questões de múltipla escolha + 20 discursivas (`cap/exNN.txt`, gabarito no Apêndice A), 18 listas de exercícios práticos + projeto (`cap/pr/prNN.txt`), 12 apêndices, 140 figuras.
- 18 apresentações `slides/CapNN.pptx` com notas do apresentador.
- `codigo/capNN/`: scripts (`.py`) e notebooks (`.ipynb`) por capítulo; pacote `codigo/comum/llmdidatico`; `requirements.txt`; `README.md`.

## Verificado (executado)
Blocos de código dos capítulos, códigos de referência das questões e soluções dos práticos foram executados com saída real (contagens exatas no Apêndice L). Referências: ver Apêndice L (150 V; 1 C — Benjamini; Yekutieli, 2001, conferida pela memória; a edição brasileira de Ozdemir (2025), não confirmada em registro editorial, foi retirada da bibliografia).

## NÃO executado / limitações (declarado no livro)
- Nenhum LLM real, API, modelo local ou GPU foi usado. Experimentos "com LLM" usam simuladores de regras e modelos didáticos NumPy sobre corpora sintéticos (declarado no texto).
- Adaptadores `python-ni` (Anthropic, HTTP, transformers, peft, sentence-transformers etc.): só a existência dos nomes foi verificada nas versões instaladas.
- Notebooks `.ipynb` foram gerados a partir dos scripts executados, mas **não foram reexecutados como notebooks** nem salvos com saídas; as saídas reais estão nos capítulos.
- Livro de Ozdemir (2024, edição inglesa): não lido na íntegra; usado só como referência-guia de escopo (Apêndice L).
- Colab, Windows e macOS não testados.
- Slides: inspeção visual por folha de contato (uma passada); revisão pedagógica humana recomendada.
- Revisão editorial humana (texto, questões e figuras) ainda necessária; licença do repositório: a definir pelo autor.

## Revisão v0.1 (parecer técnico-editorial)
- **Equidade (Cap. 17, Seção 17.13.2):** protocolo treino/validação/teste (3 000/1 500/1 500). Alvos e limiares por grupo são escolhidos só na validação; o teste apenas mede. Tabela, texto, Figura 17.8, notas dos slides e o prático 17.3 foram recalculados (o alvo não é atingido exatamente no teste: ΔDP +0,03 e ΔIO −0,01).
- **Testes múltiplos (Cap. 16, Seção 16.14.2):** Bonferroni/Holm controlam a taxa de erro por família (FWER); Benjamini-Hochberg controla a taxa de falsas descobertas (FDR), sob independência ou dependência positiva. O gabarito e o glossário foram corrigidos; a simulação é apresentada como ilustração de uma única execução.
- **Hit@k × recall@k (Caps. 10 e 13):** a métrica do laboratório multimodal passou a se chamar Hit@k (função `hit_em_k`; `recall_em_k` mantida como alias); o exemplo dos cinco relevantes está no texto.
- **Equações em linha:** conversor TeX→texto linear reescrito (frações, raízes, acentos, matrizes, normas). Verificar visualmente após a reexportação.
- **Referências:** McCloskey; Cohen (1989) com DOI e sem †; MCP (Anthropic, 25/11/2024) sem a ressalva; Ji et al. (ACM Computing Surveys, 2023) e Su et al. (Neurocomputing, 2024) completos; Ozdemir só na edição inglesa (ISBN 978-0-13-534651-8).
- **Formatação:** texto corrido justificado (referências, código e tabelas continuam à esquerda); sumário com páginas verificadas em duas passagens.
- A verificação das referências do parecer **não foi refeita online nesta sessão** (sem acesso à rede externa): os dados (DOIs, volumes, artigos) vêm do parecer.
