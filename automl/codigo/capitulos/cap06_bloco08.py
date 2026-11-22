# Capítulo 6 — Métodos multifidelidade: Successive Halving, Hyperband e BOHB
# Bloco 8. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import heapq
import numpy as np

EPOCAS = [1, 3, 9, 27, 81]; ETA = 3

def cria(N, semente):
    r = np.random.default_rng(semente)
    seg_por_epoca = np.exp(r.normal(0, 1.0, N))                # alguns candidatos são MUITO mais lentos que outros
    qualidade = 0.5 + 0.5 * r.beta(2, 5, N)                     # nota final verdadeira de cada candidato
    tau = r.uniform(2, 30, N); ruido = r.normal(0, 0.005, (N, len(EPOCAS)))
    nota = lambda i, k: qualidade[i] * (1 - np.exp(-EPOCAS[k] / tau[i])) + ruido[i, k]     # nota observada no degrau k
    return seg_por_epoca, qualidade, nota

def duracao(c, i, k, rng):                                     # tempo do degrau k, retomando do checkpoint do degrau k-1
    return (EPOCAS[k] - (EPOCAS[k - 1] if k > 0 else 0)) * c[i] * (1 if rng is None else np.exp(rng.normal(0, 0.2)))

def sincrono(N, W, semente, rng=None):
    c, q, nota = cria(N, semente); vivos = list(range(N)); t = 0.0; ocupado = 0.0
    for k in range(len(EPOCAS)):
        livre = [t] * W; notas = {}
        for i in vivos:
            w = int(np.argmin(livre)); d = duracao(c, i, k, rng); livre[w] += d; ocupado += d; notas[i] = nota(i, k)
        t = max(livre)                                           # barreira: espera o último terminar
        if k < len(EPOCAS) - 1: vivos = sorted(vivos, key=lambda i: -notas[i])[:len(vivos) // ETA]
    escolhido = max(vivos, key=lambda i: notas[i])
    return t, ocupado / (W * t), escolhido, q[escolhido]

def assincrono(N, W, semente, rng=None):
    c, q, nota = cria(N, semente); K = len(EPOCAS); feitos = [dict() for _ in range(K)]; promovidos = [set() for _ in range(K)]
    novos = list(range(N)); eventos = []; t = 0.0; ocupado = 0.0; livres = W
    def proxima_tarefa():
        for k in range(K - 2, -1, -1):                           # promove, se houver candidato entre os 1/ETA melhores do degrau
            ordem = sorted(feitos[k], key=lambda i: -feitos[k][i])
            for i in ordem[:len(ordem) // ETA]:
                if i not in promovidos[k]: promovidos[k].add(i); return i, k + 1
        return (novos.pop(0), 0) if novos else None             # senão, começa um candidato novo
    while True:
        while livres > 0 and (tarefa := proxima_tarefa()):
            i, k = tarefa; d = duracao(c, i, k, rng); ocupado += d; heapq.heappush(eventos, (t + d, i, k)); livres -= 1
        if not eventos: break
        t, i, k = heapq.heappop(eventos); feitos[k][i] = nota(i, k); livres += 1
    topo = feitos[K - 1]; escolhido = max(topo, key=topo.get)
    return t, ocupado / (W * t), escolhido, q[escolhido]

print("Média de 200 problemas (243 candidatos; tempos em 'segundos simulados'):")
print(f"{'trabalhadores':<14s} {'método':<11s} {'tempo total':>12s} {'ocupação':>9s} {'qualidade do escolhido':>23s}")
for W in (8, 32):
    for nome, f in (("síncrono", sincrono), ("assíncrono", assincrono)):
        L = np.array([[f(243, W, s)[k] for k in (0, 1, 3)] for s in range(200)])
        print(f"{W:<14d} {nome:<11s} {L[:, 0].mean():12.0f} {L[:, 1].mean():9.2f} {L[:, 2].mean():23.3f}")

print("\nReprodutibilidade: mesmos candidatos, 200 execuções com ruído de tempo de 20% (W = 32):")
for nome, f in (("síncrono", sincrono), ("assíncrono", assincrono)):
    esc = [f(243, 32, 0, np.random.default_rng(100 + j))[2] for j in range(200)]
    vals, cont = np.unique(esc, return_counts=True)
    print(f"  {nome:<11s} candidatos distintos escolhidos: {len(vals)} | o mais frequente apareceu em {100 * cont.max() / 200:.0f}% das execuções")
