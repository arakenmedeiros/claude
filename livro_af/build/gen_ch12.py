"""Gera cap/ch12.md a partir de build/ch12_template.md.
{{lib:a,b,...}}  -> bloco com o código-fonte das funções de cap/lib/avancado.py ('@header' = importações)
{{exp:eN}}       -> bloco com o experimento build/exp12/eN.py"""
import ast, re, os
B = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(B, "..", "cap", "lib", "avancado.py"), encoding="utf-8").read()
tree = ast.parse(src); linhas = src.split("\n")
funcs = {n.name: "\n".join(linhas[n.lineno - 1:n.end_lineno]) for n in tree.body if isinstance(n, ast.FunctionDef)}
HEADER = "import heapq\nimport numpy as np\nfrom fedlib import *\nfrom personalizacao import dividir_local, acuracias_locais"
parte = [0]
def lib(m):
    parte[0] += 1
    nomes = [x.strip() for x in m.group(1).split(",")]
    corpo = [HEADER if n == "@header" else funcs[n] for n in nomes]
    return f"```python\n# avancado.py - parte {parte[0]} (acrescente ao final do arquivo)\n" + "\n\n\n".join(corpo) + "\n```"
def exp(m):
    return "```python\n" + open(os.path.join(B, "exp12", m.group(1) + ".py"), encoding="utf-8").read().rstrip() + "\n```"
t = open(os.path.join(B, "ch12_template.md"), encoding="utf-8").read()
t = re.sub(r"\{\{lib:([^}]+)\}\}", lib, t)
t = re.sub(r"\{\{exp:([^}]+)\}\}", exp, t)
open(os.path.join(B, "..", "cap", "ch12.md"), "w", encoding="utf-8").write(t)
print("ch12.md gerado,", t.count("```python") , "blocos de código")
