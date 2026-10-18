"""Valida arquivos de exercícios (30 MC, 20 DISC), referências e figuras citadas."""
import re, glob, os, sys
ROOT = "/home/user/claude/llm"
refs = {}
for ln in open(ROOT + "/cap/referencias.txt"):
    if ln.strip() and not ln.startswith("#"):
        m = re.match(r"^(\w+)\s*\|\s*([VCP])\s*\|\s*(.*)$", ln)
        if not m: print("REF MALFORMADA", ln[:70])
        else: refs[m.group(1)] = m.group(3)
bad = 0
for n in range(1, 19):
    nn = f"{n:02d}"
    ex = f"{ROOT}/cap/ex{nn}.txt"; ch = f"{ROOT}/cap/ch{nn}.md"
    if os.path.exists(ex):
        s = open(ex).read()
        mc = len(re.findall(r"^MC:", s, re.M)); dc = re.findall(r"^DISC\[([MPACD])\]:", s, re.M)
        eq = len(re.findall(r"^= ", s, re.M)); wr = len(re.findall(r"^- ", s, re.M))
        from collections import Counter
        print(f"ex{nn}: MC={mc} DISC={len(dc)} tipos={dict(Counter(dc))} ok_alt={eq==mc and wr==4*mc}")
        if mc != 30 or len(dc) != 20 or eq != mc or wr != 4 * mc: bad += 1
        for q in re.findall(r"^MC:.*", s, re.M):
            if "todas as anteriores" in q.lower() or "nenhuma das" in q.lower(): print("  evitar:", q[:60])
    if os.path.exists(ch):
        s = open(ch).read()
        for k in re.findall(r"^@ref:\s*(.*)$", s, re.M):
            for key in k.split():
                if key not in refs: print(f"  cap{nn}: ref desconhecida {key}"); bad += 1
        for fid in re.findall(r"^!\[(\w+)\]", s, re.M):
            if not os.path.exists(f"{ROOT}/figs/{fid}.png"): print(f"  cap{nn}: figura ausente {fid}"); bad += 1
        w = len(re.sub(r"```.*?```", "", s, flags=re.S).split())
        print(f"ch{nn}: ~{w} palavras (sem código); {len(re.findall(r'^!\[', s, re.M))} figuras; {len(re.findall(r'^```python', s, re.M))} blocos py")
print("PROBLEMAS:", bad)
