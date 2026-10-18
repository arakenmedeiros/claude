"""Executa a solução de cada exercício/projeto (pr/prNN.txt), guarda saídas em pr_out.json e grava codigo/capNN/capNN_praticos.py."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re, sys, os, subprocess, json, glob, tempfile
ROOT = "/home/user/claude/llm"
env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=ROOT + "/codigo/comum", PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
def parse(path):
    items = []; cur = None; mode = None
    for ln in open(path).read().replace("\r", "").split("\n"):
        if ln.startswith("@@EX ") or ln.startswith("@@PROJ "):
            kind = "EX" if ln.startswith("@@EX ") else "PROJ"
            parts = [p.strip() for p in ln.split(" ", 1)[1].split("|")]
            if kind == "EX": num = int(parts[0]); title = parts[1]
            else: num = 0; title = parts[0]
            cur = {"kind": kind, "num": num, "title": title, "sol": []}; items.append(cur); mode = "stmt"; continue
        if ln.startswith("@@ALEM"): cur = None; mode = None; continue
        if ln.startswith("@@SOL"): mode = "sol"; continue
        if ln.startswith("@@RUBRICA") or ln.startswith("@@NOTE"): mode = "other"; continue
        if cur is not None and mode == "sol": cur["sol"].append(ln)
    return items
outp = ROOT + "/build/pr_out.json"
out = json.load(open(outp)) if os.path.exists(outp) else {}
sel = None if len(sys.argv) < 2 else {int(a) for a in sys.argv[1:]}
fails = 0
for f in sorted(glob.glob(ROOT + "/cap/pr/pr*.txt")):
    ch = int(re.search(r"pr(\d+)", f).group(1))
    if sel and ch not in sel: continue
    cells = []
    for k in [k for k in out if k.startswith(f"{ch}-")]: del out[k]
    for it in parse(f):
        m = re.findall(r"```python\n(.*?)```", "\n".join(it["sol"]), re.S)
        if not m: print("SEM CODIGO", f, it["title"]); fails += 1; continue
        key = f"{ch}-{it['kind']}{it['num']}"
        cells.append(f"# %% [{'Exercício ' + str(ch) + '.' + str(it['num']) if it['kind']=='EX' else 'Projeto'}: {it['title']}]\n" + m[0])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.py"); open(p, "w").write(m[0])
            try: r = subprocess.run([sys.executable, "-W", "ignore", p], capture_output=True, text=True, timeout=1500, cwd=d, env=env)
            except subprocess.TimeoutExpired: print("TIMEOUT", key); fails += 1; continue
        if r.returncode != 0: print("FAIL", key); print(r.stderr[-900:]); fails += 1
        else: print("OK  ", key, flush=True); out[key] = r.stdout
    d = f"{ROOT}/codigo/cap{ch:02d}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/cap{ch:02d}_praticos.py", "w").write(f'"""Capítulo {ch}: soluções dos exercícios práticos e do projeto. Cada "# %%" é uma célula independente."""\n\n' + "\n\n".join(cells) + "\n")
from jmerge import merge_write
merge_write(outp, out, {int(re.search(r"pr(\d+)", f).group(1)) for f in glob.glob(ROOT + "/cap/pr/pr*.txt") if not sel or int(re.search(r"pr(\d+)", f).group(1)) in sel}, ensure_ascii=False, indent=1)
print("falhas:", fails); sys.exit(1 if fails else 0)
