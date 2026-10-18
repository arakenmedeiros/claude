"""Gera notebooks .ipynb por capítulo (SEM saídas: não foram executados como notebooks)."""
import re, os, glob, nbformat as nbf
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIT = {1: "Introdução", 2: "Fundamentos matemáticos e Python/scikit-learn", 3: "Perceptron", 4: "Backpropagation e MLP",
       5: "Quickprop", 6: "RPROP e variantes", 7: "Levenberg–Marquardt (LMBP)", 8: "Delta-Bar-Delta e EDBD"}
sys_nb = {}
def parse_pr(path):
    out, cur, mode = [], None, None
    for ln in open(path, encoding="utf8").read().split("\n"):
        if ln.startswith("@@EX ") or ln.startswith("@@PROJ "):
            cur = {"t": ln.split(" ", 1)[1].split("|")[0].strip(), "stmt": [], "sol": []}; out.append(cur); mode = "stmt"; continue
        if ln.startswith("@@SOL"): mode = "sol"; continue
        if ln.startswith("@@"): mode = "x"; continue
        if cur is not None and mode in ("stmt", "sol"): cur[mode].append(ln)
    return out
os.makedirs(os.path.join(R, "notebooks"), exist_ok=True)
for n in range(1, 9):
    nb = nbf.v4.new_notebook(); c = nb.cells
    c.append(nbf.v4.new_markdown_cell(f"# Capítulo {n} — {TIT[n]}\n\nCódigos do livro *Backpropagation e Redes Neurais* (Araken de Medeiros Santos).\n\n"
        "> **Aviso:** este notebook foi gerado a partir dos scripts do livro e **não foi executado como notebook**; as saídas reportadas no texto vêm da execução dos scripts `.py` correspondentes. "
        "Execute as células na ordem, a partir desta pasta, com o ambiente de `requirements.txt` (ver Apêndice E). Valores podem diferir ligeiramente em outras plataformas."))
    c.append(nbf.v4.new_code_cell("import os, sys\nRAIZ = os.path.abspath(os.path.join(os.getcwd(), '..'))\nsys.path.insert(0, os.path.join(RAIZ, 'codigo'))\nos.environ.setdefault('FIGS', os.path.join(RAIZ, 'figs'))\nprint('rnbook em', os.path.join(RAIZ, 'codigo'))"))
    for f in sorted(glob.glob(os.path.join(R, "codigo", f"cap{n:02d}", "*.py"))):
        src = open(f, encoding="utf8").read()
        src = re.sub(r"^sys\.path\.insert\(0, os\.path\.join\(os\.path\.dirname\(__file__\), \"\.\.\"\)\)\n", "", src, flags=re.M)
        src = src.replace("os.path.dirname(__file__)", "os.path.join(RAIZ, 'codigo', 'cap%02d')" % n)
        c.append(nbf.v4.new_markdown_cell(f"## Script `{os.path.relpath(f, os.path.join(R, 'codigo'))}`"))
        partes = re.split(r"\n(?=# -{10,})", src)
        for p in partes:
            if p.strip(): c.append(nbf.v4.new_code_cell(p.strip("\n")))
    pf = os.path.join(R, "cap", "pr", f"pr{n:02d}.txt")
    if os.path.exists(pf):
        c.append(nbf.v4.new_markdown_cell("## Exercícios práticos e projeto: soluções de referência"))
        for it in parse_pr(pf):
            code = re.findall(r"```python\n(.*?)```", "\n".join(it["sol"]), re.S)
            c.append(nbf.v4.new_markdown_cell("### " + it["t"] + "\n\n" + "\n".join(it["stmt"]).strip()))
            if code: c.append(nbf.v4.new_code_cell(code[0].strip("\n")))
    if n == 8:
        pf = os.path.join(R, "cap", "pr", "pr_final.txt")
        c.append(nbf.v4.new_markdown_cell("## Projeto integrador final"))
        for it in parse_pr(pf):
            code = re.findall(r"```python\n(.*?)```", "\n".join(it["sol"]), re.S)
            c.append(nbf.v4.new_markdown_cell("### " + it["t"] + "\n\n" + "\n".join(it["stmt"]).strip()))
            if code: c.append(nbf.v4.new_code_cell(code[0].strip("\n")))
    nbf.validate(nb); nbf.write(nb, os.path.join(R, "notebooks", f"cap{n:02d}.ipynb")); print("ok", n, len(c), "células")
