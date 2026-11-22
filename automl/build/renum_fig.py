"""Renumera as figuras de cada capítulo em ordem sequencial (Figura N.M) e atualiza as referências no texto."""
import re, glob
for f in sorted(glob.glob("/home/user/claude/automl/src/ch*.md")):
    s = open(f).read(); n = int(re.search(r"ch(\d+)", f).group(1))
    caps = list(re.finditer(r"!\[([^\]]+)\]\(Figura (\d+)\.(\d+) —", s))
    mapa = {}
    for k, m in enumerate(caps, 1):
        mapa[f"{m.group(2)}.{m.group(3)}"] = f"{n}.{k}"
    if all(a == b for a, b in mapa.items()):
        continue
    # passo 1: marcadores temporários, para evitar colisões
    def sub(m):
        ref = f"{m.group(1)}.{m.group(2)}"
        return f"Figura @@{mapa[ref]}@@" if ref in mapa else m.group(0)
    s2 = re.sub(r"Figura (\d+)\.(\d+)", sub, s)
    s2 = re.sub(r"Figura @@([\d.]+)@@", r"Figura \1", s2)
    open(f, "w").write(s2); print("renumerado", f, {a: b for a, b in mapa.items() if a != b})
