import re, subprocess, sys, os, glob
os.environ["MPLBACKEND"] = "Agg"
tot = ok = 0
for f in sorted(glob.glob("/home/user/claude/semissupervisionado/cap/ch*.md")) + ["/home/user/claude/semissupervisionado/cap/guia.md"]:
    s = open(f).read()
    # pares: bloco python seguido (opcionalmente) de bloco saída
    for m in re.finditer(r"```python\n(.*?)```(\s*\n```saída\n(.*?)```)?", s, re.S):
        code, saida = m.group(1), m.group(3)
        tot += 1
        p = f"/tmp/v_{os.path.basename(f)}_{tot}.py"
        open(p, "w").write(code)
        r = subprocess.run(["python3", "-W", "ignore", p], capture_output=True, text=True, timeout=900, cwd="/tmp")
        name = f"{os.path.basename(f)} bloco {tot}"
        if r.returncode != 0:
            print("FALHA", name); print(r.stderr[-500:]); continue
        if saida is None:
            print("ok (sem saída esperada)", name); ok += 1; continue
        a = "\n".join(l.rstrip() for l in r.stdout.strip().split("\n")); b = "\n".join(l.rstrip() for l in saida.strip().split("\n"))
        if a == b: print("ok  idêntica", name); ok += 1
        else:
            print("DIFERENTE", name); print("--- obtido:\n" + a); print("--- livro:\n" + b)
print(f"{ok}/{tot} blocos verificados")
