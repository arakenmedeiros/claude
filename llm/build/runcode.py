"""Executa os blocos ```python / ```python+ de cada capítulo e guarda as saídas reais em code_out.json.
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
Também grava codigo/capNN/capNN_texto.py com os blocos (células # %%)."""
import re, sys, os, subprocess, json, glob, tempfile
ROOT = "/home/user/claude/llm"
env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=ROOT + "/codigo/comum", PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
files = sorted(glob.glob(ROOT + "/cap/ch*.md")) if len(sys.argv) < 2 else [ROOT + f"/cap/ch{int(a):02d}.md" for a in sys.argv[1:]]
outp = ROOT + "/build/code_out.json"
res = json.load(open(outp)) if os.path.exists(outp) else {}
fails = 0
for f in files:
    ch = int(re.search(r"ch(\d+)", f).group(1))
    s = open(f).read()
    blocks = re.findall(r"^```(python\+?)\n(.*?)^```", s, re.S | re.M)
    for k in [k for k in res if k.startswith(f"{ch}-")]: del res[k]
    buf = ""; cells = []; M = '\nprint("<<<BLOCO>>>")\n'
    for i, (lang, code) in enumerate(blocks, 1):
        buf = (buf + M + code) if lang == "python+" else code
        cells.append(f"# %% [bloco {i}]\n" + code)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.py"); open(p, "w").write(buf)
            try:
                r = subprocess.run([sys.executable, "-W", "ignore", p], capture_output=True, text=True, timeout=900, cwd=d, env=env)
            except subprocess.TimeoutExpired:
                print("TIMEOUT", ch, i); fails += 1; continue
        if r.returncode != 0:
            print("FAIL", f"cap{ch} bloco {i}"); print(r.stderr[-900:]); fails += 1; continue
        out = r.stdout.split("<<<BLOCO>>>\n")[-1]
        res[f"{ch}-{i}"] = out
        print("OK  ", f"cap{ch} bloco {i}", f"({len(out.splitlines())} linhas)", flush=True)
    d = f"{ROOT}/codigo/cap{ch:02d}"; os.makedirs(d, exist_ok=True)
    if cells:
        open(f"{d}/cap{ch:02d}_texto.py", "w").write(f'"""Capítulo {ch}: blocos de código do texto, em ordem. Cada "# %%" é uma célula."""\n\n' + "\n\n".join(cells) + "\n")
from jmerge import merge_write
merge_write(outp, res, {int(re.search(r"ch(\d+)", f).group(1)) for f in files}, ensure_ascii=False, indent=1)
print("falhas:", fails)
sys.exit(1 if fails else 0)
