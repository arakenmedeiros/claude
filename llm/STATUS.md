# STATUS do projeto — "Grandes Modelos de Linguagem" (Araken de Medeiros Santos)

Data: 19/out/2026 · Corte bibliográfico: 18/out/2026

## Título
Considerados: (1) *Grandes Modelos de Linguagem: do básico à aplicação*; (2) *Entendendo e Construindo com LLMs*; (3) *LLMs na Prática: fundamentos, aplicações e responsabilidade*. Adotado na capa/arquivo: **Grandes Modelos de Linguagem** (subtítulo: "Dos fundamentos aos sistemas, com Python").

## Entregue
- `Grandes_Modelos_de_Linguagem.docx` (944 páginas, sumário como campo Word com páginas verificadas em duas passagens: "PAGINAS ESTAVEIS") e cópia em PDF.
- 18 capítulos (`cap/chNN.md`), 18 conjuntos de 30 questões de múltipla escolha + 20 discursivas (`cap/exNN.txt`, gabarito no Apêndice A), 18 listas de exercícios práticos + projeto (`cap/pr/prNN.txt`), 12 apêndices, 140 figuras.
- 18 apresentações `slides/CapNN.pptx` com notas do apresentador.
- `codigo/capNN/`: scripts (`.py`) e notebooks (`.ipynb`) por capítulo; pacote `codigo/comum/llmdidatico`; `requirements.txt`; `README.md`.

## Verificado (executado)
Blocos de código dos capítulos, códigos de referência das questões e soluções dos práticos foram executados com saída real (contagens exatas no Apêndice L). Referências: ver Apêndice L (maioria V; 1 C; 1 P — Ozdemir 2025, não localizada).

## NÃO executado / limitações (declarado no livro)
- Nenhum LLM real, API, modelo local ou GPU foi usado. Experimentos "com LLM" usam simuladores de regras e modelos didáticos NumPy sobre corpora sintéticos (declarado no texto).
- Adaptadores `python-ni` (Anthropic, HTTP, transformers, peft, sentence-transformers etc.): só a existência dos nomes foi verificada nas versões instaladas.
- Notebooks `.ipynb` foram gerados a partir dos scripts executados, mas **não foram reexecutados como notebooks** nem salvos com saídas; as saídas reais estão nos capítulos.
- Livro de Ozdemir (2025): não lido na íntegra; usado só como referência-guia de escopo (Apêndice L).
- Colab, Windows e macOS não testados.
- Slides: inspeção visual por folha de contato (uma passada); revisão pedagógica humana recomendada.
- Revisão editorial humana (texto, questões e figuras) ainda necessária; licença do repositório: a definir pelo autor.
