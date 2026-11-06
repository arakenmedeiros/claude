"""Converte codigo/capNN/capNN_{texto,praticos}.py (células '# %% [título]') em .ipynb (sem saídas salvas)."""
import json, re, sys, ast
for path in sys.argv[1:]:
    src = open(path).read()
    doc = ast.get_docstring(ast.parse(src)) or ""
    body = src.split('"""', 2)[2] if src.lstrip().startswith('"""') else src
    parts = re.split(r"^# %% ?\[(.*?)\]\n", body, flags=re.M)
    cells = [{"cell_type": "markdown", "metadata": {}, "source": (f"# {doc}\n\nPré-requisito: `PYTHONPATH=codigo/comum` (pacote `llmdidatico`). Veja o Apêndice H.").splitlines(True)},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["import sys; sys.path.insert(0, '../comum')"]}]
    for i in range(1, len(parts), 2):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"### [{parts[i]}]"]})
        code = parts[i + 1].strip("\n")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(True)})
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(path[:-3] + ".ipynb", "w"), ensure_ascii=False, indent=1)
    print("ok", path, len(cells))
