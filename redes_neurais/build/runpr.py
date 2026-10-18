"""Executa as soluções de cap/pr/prNN.txt e pr_final.txt; grava build/pr_out.json."""
import re, sys, os, subprocess, json, glob
os.environ["MPLBACKEND"] = "Agg"
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONPATH"] = os.path.join(RAIZ, "codigo")

def parse(path):
    items, cur, mode = [], None, None
    for ln in open(path, encoding="utf8").read().replace("\r", "").split("\n"):
        if ln.startswith("@@EX ") or ln.startswith("@@PROJ "):
            kind = "EX" if ln.startswith("@@EX ") else "PROJ"
            parts = [p.strip() for p in ln.split(" ", 1)[1].split("|")]
            num = int(parts[0]) if kind == "EX" else 0
            cur = {"kind": kind, "num": num, "sol": []}; items.append(cur); mode = "stmt"; continue
        if ln.startswith("@@SOL"): mode = "sol"; continue
        if ln.startswith("@@"): mode = "x"; continue
        if cur is not None and mode == "sol": cur["sol"].append(ln)
    return items

if __name__ == "__main__":
    arqs = sorted(glob.glob(os.path.join(RAIZ, "cap", "pr", "pr*.txt"))) if len(sys.argv) < 2 else sys.argv[1:]
    saidas = json.load(open(os.path.join(RAIZ, "build", "pr_out.json"))) if os.path.exists(os.path.join(RAIZ, "build", "pr_out.json")) else {}
    falhas = 0
    for f in arqs:
        m = re.search(r"pr(\d+|_final)", f); ch = m.group(1).lstrip("_") if m else "?"
        ch = ch if ch == "final" else str(int(ch))
        for it in parse(f):
            code = re.findall(r"```python\n(.*?)```", "\n".join(it["sol"]), re.S)
            key = f"{ch}-{it['kind']}{it['num']}"
            if not code: print("SEM CÓDIGO", key); falhas += 1; continue
            p = f"/tmp/claude-0/scratch/pr_{key}.py"; open(p, "w").write(code[0])
            try:
                r = subprocess.run(["python3", "-W", "ignore", p], capture_output=True, text=True, timeout=1200, cwd="/tmp/claude-0/scratch")
            except subprocess.TimeoutExpired:
                print("TIMEOUT", key); falhas += 1; continue
            if r.returncode: print("FALHA", key); print(r.stderr[-900:]); falhas += 1
            else: print("OK  ", key, flush=True); saidas[key] = r.stdout
    json.dump(saidas, open(os.path.join(RAIZ, "build", "pr_out.json"), "w"), ensure_ascii=False, indent=1)
    print("falhas:", falhas)
