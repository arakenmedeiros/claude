"""Converte cap/src/exNN.src.txt -> cap/exNN.txt.
Fonte MC:   MC: enunciado / + correta / - errada (x4) / R: comentário
Fonte DISC: DISC: enunciado / R: resposta (várias linhas) até a próxima entrada.
Posição da correta: 30 questões, 6 de cada letra A-E, ordem pseudoaleatória fixa
(sem mais de 2 iguais seguidas e sem padrão cíclico)."""
import sys, random, re, glob, os

def permutacao(cap):
    rng = random.Random(1000 + cap)
    while True:
        p = list("abcde") * 6
        rng.shuffle(p)
        if any(p[i] == p[i+1] == p[i+2] for i in range(len(p) - 2)):
            continue
        if any(p[i:i+5] == list("abcde") or p[i:i+5] == list("edcba") for i in range(len(p) - 4)):
            continue
        return p

def converter(arq):
    cap = int(re.search(r"ex(\d+)", arq).group(1))
    txt = open(arq, encoding="utf8").read().replace("\r", "")
    linhas = txt.split("\n")
    entradas, cur = [], None
    for ln in linhas:
        if ln.startswith("MC:") or ln.startswith("DISC:"):
            cur = [ln]; entradas.append(cur)
        elif cur is not None:
            cur.append(ln)
    mcs = [e for e in entradas if e[0].startswith("MC:")]
    dcs = [e for e in entradas if e[0].startswith("DISC:")]
    if len(mcs) != 30 or len(dcs) != 20:
        print(f"ERRO {arq}: {len(mcs)} MC e {len(dcs)} DISC (esperado 30 e 20)"); sys.exit(1)
    perm = permutacao(cap)
    saida = []
    for q, pos in zip(mcs, perm):
        corpo = [l for l in q[1:] if l.strip()]
        cert = [l[2:].strip() for l in corpo if l.startswith("+ ")]
        err = [l[2:].strip() for l in corpo if l.startswith("- ")]
        r = [l for l in corpo if l.startswith("R:")]
        assert len(cert) == 1 and len(err) == 4 and len(r) == 1, (arq, q[0])
        i = "abcde".index(pos)
        ops = err[:i] + [cert[0]] + err[i:]
        saida.append("\n".join([q[0]] + [f"{'abcde'[k]}) {o}" for k, o in enumerate(ops)] + [f"R: {pos} | {r[0][2:].strip()}"]))
    for q in dcs:
        corpo = q[1:]
        while corpo and not corpo[-1].strip(): corpo.pop()
        saida.append("\n".join([q[0]] + corpo))
    out = arq.replace("/src/", "/").replace(".src.txt", ".txt")
    open(out, "w", encoding="utf8").write("\n\n".join(saida) + "\n")
    cont = {c: perm.count(c) for c in "abcde"}
    print(out, "ok", cont)

if __name__ == "__main__":
    alvo = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "cap", "src", "ex*.src.txt")))
    for a in alvo: converter(a)
