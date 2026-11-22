"""Extrai os códigos dos capítulos e das soluções em scripts independentes (pasta codigo/)."""
import re, glob, os, shutil
ROOT = "/home/user/claude/automl"; OUT = ROOT + "/codigo"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT + "/capitulos"); os.makedirs(OUT + "/exercicios"); os.makedirs(OUT + "/nao_executados")

def flags_of(code):
    fl, lines = set(), code.split("\n")
    while lines and lines[0].startswith("#@"): fl.add(lines[0][2:].strip()); lines = lines[1:]
    return fl, "\n".join(lines)

indice = []
for f in sorted(glob.glob(ROOT + "/src/ch*.md")):
    n = int(re.search(r"ch(\d+)", f).group(1)); s = open(f).read()
    titulo = re.search(r"^# (.+)$", s, re.M).group(1); prefix = ""; k = 0
    for m in re.finditer(r"```python\n(.*?)```\n", s, re.S):
        k += 1; fl, code = flags_of(m.group(1))
        cab = f"# Capítulo {n} — {titulo}\n# Bloco {k}. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').\n"
        if "naoexec" in fl:
            open(f"{OUT}/nao_executados/cap{n:02d}_bloco{k:02d}.py", "w").write(cab + "# ATENÇÃO: este trecho NÃO foi executado na elaboração do livro. Fixe versões e confira a documentação.\n" + code)
            indice.append((n, k, "nao_executados", "não executado")); continue
        if "nb" in fl or "nbx" in fl:
            corpo = (("# --- blocos anteriores deste capítulo (encadeados) ---\n" + prefix + "# --- fim dos blocos anteriores ---\n") if prefix else "") + code
            if "nb" in fl: prefix += code + ("\n" if not code.endswith("\n") else "")
        else: corpo = code
        open(f"{OUT}/capitulos/cap{n:02d}_bloco{k:02d}.py", "w").write(cab + corpo); indice.append((n, k, "capitulos", ""))
for f in sorted(glob.glob(ROOT + "/cap/pr/pr*.txt")):
    n = int(re.search(r"pr(\d+)", f).group(1)); txt = open(f).read(); partes = re.split(r"^@@(EX \d+|PROJ)", txt, flags=re.M)
    for i in range(1, len(partes), 2):
        nome = partes[i].replace(" ", "").lower(); m = re.search(r"@@SOL\n```python\n(.*?)```", partes[i + 1], re.S)
        if m: open(f"{OUT}/exercicios/cap{n:02d}_{nome.replace('proj','projeto')}.py", "w").write(f"# Capítulo {n} — solução de referência ({partes[i]})\n" + m.group(1))
open(OUT + "/README.md", "w").write("""# Códigos do livro "AutoML: Teoria e Prática"

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
""")
shutil.copy(ROOT + "/requirements.txt", OUT + "/requirements.txt")
print(len(indice), "blocos;", len(os.listdir(OUT + "/exercicios")), "soluções")
