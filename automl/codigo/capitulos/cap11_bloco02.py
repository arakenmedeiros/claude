# Capítulo 11 — Busca de arquitetura neural (NAS)
# Bloco 2. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import json, time
import numpy as np
import os; os.makedirs("resultados", exist_ok=True)

T = json.load(open("resultados/f11_cache.json"))
chave = lambda a: (tuple(a[0]), a[1])
V = {chave(r["arq"]): r["v40"] for r in T}; TESTE = {chave(r["arq"]): r["teste"] for r in T}
ARQS = list(V); LARGURAS = [8, 16, 32, 64, 128]; ATIVACOES = ["relu", "tanh"]
posto = lambda a: 1 + sum(v > V[a] for v in V.values())

def mutacao(a, r):
    h, ativ = list(a[0]), a[1]; op = r.integers(4)
    if op == 0: h[r.integers(len(h))] = int(r.choice(LARGURAS))        # muda uma largura
    elif op == 1 and len(h) < 3: h.insert(r.integers(len(h) + 1), int(r.choice(LARGURAS)))   # insere camada
    elif op == 2 and len(h) > 1: h.pop(r.integers(len(h)))             # remove camada
    else: ativ = [x for x in ATIVACOES if x != ativ][0]                # troca a ativação
    return (tuple(h), ativ)

def aleatoria(B, r):
    return [ARQS[i] for i in r.choice(len(ARQS), B, replace=False)]

def evolucao(B, r, P=8, S=3):
    hist, pop, vistas = [], [], set()
    while len(hist) < P:
        a = ARQS[r.integers(len(ARQS))]
        if a not in vistas: vistas.add(a); hist.append(a); pop.append(a)
    while len(hist) < B:
        pai = max([pop[i] for i in r.choice(len(pop), S, replace=False)], key=V.get)
        for _ in range(50):
            filha = mutacao(pai, r)
            if filha not in vistas: break
        else: filha = ARQS[r.integers(len(ARQS))]
        if filha in vistas: continue
        vistas.add(filha); hist.append(filha); pop.append(filha); pop.pop(0)    # a mais velha sai
    return hist

B, R = 30, 300
curvas = {}; resumo = {}; t0 = time.perf_counter()
for nome, fn in (("aleatória", aleatoria), ("evolução", evolucao)):
    linhas, cs = [], []
    for s in range(R):
        h = fn(B, np.random.default_rng(s)); b = max(h, key=V.get)
        linhas.append((V[b], posto(b), TESTE[b]))
        cs.append(np.maximum.accumulate([V[a] for a in h]))
    L = np.array(linhas); curvas[nome] = np.array(cs)
    resumo[nome] = {"val": float(np.median(L[:, 0])), "posto": float(np.median(L[:, 1])), "top5": float(np.mean(L[:, 1] <= 16)), "teste": float(L[:, 2].mean())}
    print(f"{nome:<10s} val. = {np.median(L[:, 0]):.3f} | posto = {np.median(L[:, 1]):.1f} | "
          f"top 5% = {100 * np.mean(L[:, 1] <= 16):.0f}% | teste = {L[:, 2].mean():.3f}")
print("Melhor do espaço (validação): %.3f | teste médio das 310 redes: %.3f" % (max(V.values()), np.mean(list(TESTE.values()))))
print("Consultas à tabela: %d (2 estratégias x %d repetições x %d avaliações) em %.2f s" % (2 * R * B, R, B, time.perf_counter() - t0))
json.dump({"resumo": resumo, "curvas": {k: v.tolist() for k, v in curvas.items()}}, open("resultados/f11_busca.json", "w"))
