"""Executa os códigos de cap/code/*.py e grava a saída real em *.out (com cache por hash).
Uso: python3 run_code.py [prefixo ...]   (ex.: python3 run_code.py ch04)  | --force para reexecutar."""
import sys, os, subprocess, hashlib, glob, time
ROOT = "/home/user/claude/semissupervisionado"
CODE = ROOT + "/cap/code"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
force = "--force" in sys.argv
files = sorted(glob.glob(CODE + "/*.py"))
if args: files = [f for f in files if any(os.path.basename(f).startswith(a) for a in args)]
falhas = 0
for f in files:
    code = open(f).read(); h = hashlib.md5(code.encode()).hexdigest()
    base = f[:-3]
    if not force and os.path.exists(base + ".hash") and open(base + ".hash").read() == h and os.path.exists(base + ".out"):
        print("cache", os.path.basename(f)); continue
    t = time.time()
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
    r = subprocess.run(["python3", "-W", "ignore", f], capture_output=True, text=True, cwd=ROOT, env=env, timeout=6000)
    if r.returncode != 0:
        print("FALHA", os.path.basename(f)); print(r.stderr[-1200:]); falhas += 1; continue
    open(base + ".out", "w").write(r.stdout); open(base + ".hash", "w").write(h)
    print("ok   %-28s %5.1fs" % (os.path.basename(f), time.time() - t))
print("falhas:", falhas)
