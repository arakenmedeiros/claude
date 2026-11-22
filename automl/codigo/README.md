# Códigos do livro "AutoML: Teoria e Prática"

Esta pasta reúne os códigos do livro, um script independente por bloco de código.

## Como executar

1. Crie um ambiente virtual e instale as versões usadas no livro:
   `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. Execute os scripts **a partir desta pasta** (`codigo/`), por exemplo: `python capitulos/cap02_bloco03.py`.
   Os resultados intermediários (arquivos .json) são gravados em `resultados/`, criada automaticamente.
3. Alguns blocos dependem de arquivos produzidos por blocos anteriores do mesmo capítulo (por exemplo, o Capítulo 11 lê `resultados/f11_cache.json`). Execute os blocos **em ordem**.

## Conteúdo

- `capitulos/`: um script por bloco de código dos capítulos (blocos "encadeados" já trazem os anteriores do capítulo).
- `exercicios/`: as soluções de referência dos exercícios práticos e dos projetos.
- `nao_executados/`: trechos que **não foram executados** na elaboração do livro (ferramentas que não puderam ser instaladas). Fixe versões e consulte a documentação.

## Avisos de reprodutibilidade

- Códigos com **orçamento de tempo** (FLAML, Optuna com `timeout`, medidas de latência) dependem da máquina; os números podem variar.
- O Capítulo 10 usa processos novos (`spawn`) e leva vários minutos.
- Os exemplos foram executados em Python 3.13 com as versões do `requirements.txt`.
