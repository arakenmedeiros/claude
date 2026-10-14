"""Embaralha as alternativas das questões de múltipla escolha (resposta correta escrita sempre como 'a'). Uso: python3 build/shuf.py cap/exNN.txt [semente]"""
import re, sys, random
f = sys.argv[1]; r = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
blocks = re.split(r"\n\s*\n", open(f).read().strip()); out = []
for b in blocks:
    L = b.split("\n")
    if b.startswith("MC:") and len(L) == 6 and L[5].startswith("R: a | "):
        opts = [l[3:] for l in L[1:5]]; ordem = list(range(4)); r.shuffle(ordem)
        novo = [opts[i] for i in ordem]; pos = "abcd"[ordem.index(0)]
        L = [L[0]] + [f"{'abcd'[i]}) {t}" for i, t in enumerate(novo)] + ["R: " + pos + L[5][4:]]
    out.append("\n".join(L))
open(f, "w").write("\n\n".join(out) + "\n")
