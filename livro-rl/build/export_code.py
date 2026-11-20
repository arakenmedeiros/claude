"""Exporta os blocos de código do livro como arquivos .py executáveis (pasta codigo/)."""
import re, os, glob, subprocess, sys, importlib.metadata as md
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cap = os.path.join(root, "cap"); out = os.path.join(root, "codigo"); os.makedirs(out, exist_ok=True)
for f in glob.glob(out + "/*.py"): os.remove(f)
files = sorted(glob.glob(cap + "/ch*.md")) + [cap + "/integrador.md"] + sorted(glob.glob(cap + "/ap*.md"))
n_files = 0
for f in files:
    s = open(f).read()
    title = re.search(r"^# (.+)$", s, re.M).group(1)
    blocks = re.findall(r"```(python|python-nr)\n(.*?)```", s, re.S)
    if not blocks: continue
    base = os.path.basename(f)[:-3]
    head = ['"""', f"Livro: Aprendizado por Reforço — {title}", "Autor do livro: Araken de Medeiros Santos",
            "", "Os blocos abaixo seguem a ordem do texto e devem ser executados em sequência (como células de um notebook).",
            "Os trechos marcados como 'não executado' (por exemplo, downloads do Kaggle) aparecem comentados.", '"""',
            "import warnings; warnings.filterwarnings('ignore')", ""]
    body = []
    for i, (lang, code) in enumerate(blocks):
        body.append(f"# {'=' * 70}\n# Bloco {i + 1}" + (" (trecho não executado: requer conta/download externo)" if lang == "python-nr" else "") + f"\n# {'=' * 70}")
        body.append(code.rstrip() if lang == "python" else "\n".join("# " + l for l in code.rstrip().split("\n")))
        body.append("")
    open(os.path.join(out, base + ".py"), "w").write("\n".join(head + body) + "\n"); n_files += 1
pk = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "gymnasium", "torch", "stable-baselines3", "ucimlrepo"]
req = []
for p in pk:
    try: req.append(f"{p}=={md.version(p)}")
    except Exception: req.append(p)
open(os.path.join(out, "requirements.txt"), "w").write(f"# Python {sys.version.split()[0]}\n" + "\n".join(req) + "\n")
open(os.path.join(out, "LEIA-ME.md"), "w").write("""# Códigos do livro "Aprendizado por Reforço"

Cada arquivo `chNN.py` reúne, na ordem do texto, os blocos de código do capítulo NN; `integrador.py` é o projeto integrador e `apX_*.py` são os apêndices.

## Como executar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ch02.py
```

- As versões das bibliotecas usadas na verificação estão em `requirements.txt`.
- Os capítulos que usam bases do UCI baixam os dados pela biblioteca `ucimlrepo` (é preciso acesso à internet).
- Os capítulos 3, 4, 5, 7, 11 e o projeto integrador usam arquivos gerados no Capítulo 3 (`painel_clientes.csv`, `P_clientes.npy`, `r_clientes.npy`): execute `ch03.py` primeiro, na mesma pasta.
- Os resultados variam entre execuções e entre versões das bibliotecas e do hardware (sementes não garantem reprodutibilidade completa).
""")
print(n_files, "arquivos;", ", ".join(req))
