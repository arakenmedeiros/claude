"""Executa os códigos de referência (CODE:...ENDCODE) das questões de programação; grava build/ex_ok.json com as chaves 'cap-questão' que passaram."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re, sys, os, json, glob, subprocess, tempfile
ROOT = "/home/user/claude/llm"
env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=ROOT + "/codigo/comum", PYTHONHASHSEED="0")
outp = ROOT + "/build/ex_ok.json"; ok = json.load(open(outp)) if os.path.exists(outp) else {}
sel = None if len(sys.argv) < 2 else {int(a) for a in sys.argv[1:]}
fails = 0
for f in sorted(glob.glob(ROOT + "/cap/ex*.txt")):
    ch = int(re.search(r"ex(\d+)", f).group(1))
    if sel and ch not in sel: continue
    for k in [k for k in ok if k.startswith(f"{ch}-")]: del ok[k]
    lines = open(f).read().split("\n"); qi = 0; i = 0; cells = []
    while i < len(lines):
        if lines[i].startswith("DISC["): qi += 1
        if lines[i].startswith("CODE:"):
            code = []; i += 1
            while lines[i].strip() != "ENDCODE": code.append(lines[i]); i += 1
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "q.py"); open(p, "w").write("\n".join(code))
                r = subprocess.run([sys.executable, "-W", "ignore", p], capture_output=True, text=True, timeout=600, cwd=d, env=env)
            if r.returncode: print("FAIL", ch, qi); print(r.stderr[-600:]); fails += 1
            else: ok[f"{ch}-{qi}"] = True; print("OK  ", ch, qi, r.stdout.strip()[:60].replace("\n", " | "))
            cells.append(f"# %% [Questão discursiva {qi}]\n" + "\n".join(code))
        i += 1
    if cells:
        d = f"{ROOT}/codigo/cap{ch:02d}"; os.makedirs(d, exist_ok=True)
        open(f"{d}/cap{ch:02d}_questoes.py", "w").write(f'"""Capítulo {ch}: códigos de referência das questões discursivas de programação."""\n\n' + "\n\n".join(cells) + "\n")
from jmerge import merge_write
merge_write(outp, ok, {int(re.search(r"ex(\d+)", f).group(1)) for f in glob.glob(ROOT + "/cap/ex*.txt") if not sel or int(re.search(r"ex(\d+)", f).group(1)) in sel}, indent=1); print("falhas:", fails); sys.exit(1 if fails else 0)
