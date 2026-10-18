"""Figuras do Capítulo 15. As figuras [D] são calculadas com as mesmas funções e sementes do texto (hardware, preços e tempos HIPOTÉTICOS;
modelos didáticos, SimuladorLLM e simulações de filas; nenhum LLM real). As demais são esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
import json, re
from fc import *
from llmdidatico import corpus
from llmdidatico.modelos import MLPLM, LSAEmbedder
from llmdidatico.simulador import SimuladorLLM, INTENCOES, _PALAVRAS, _sem_acentos
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------------ esquemas
def mapa():
    fig, ax = canvas(9.4, 4.6)
    rbox(ax, 2, 66, 30, 28, "Origem do custo\ntokens de entrada e saída\nmemória · cálculo", fc="#E0E7FF", ec=IND, fs=8.6, bold=True)
    rbox(ax, 35, 66, 30, 28, "Alavancas\nquantização · modelo menor\ncache · lote · especulação", fc="#CFFAFE", ec=CYA, fs=8.6, bold=True)
    rbox(ax, 68, 66, 30, 28, "Roteamento e filas\nbarato→caro\nlatência × carga", fc="#D1FAE5", ec=GRN, fs=8.6, bold=True)
    arrow(ax, (32.5, 80), (34.5, 80), c=INK); arrow(ax, (65.5, 80), (67.5, 80), c=INK)
    rbox(ax, 2, 10, 30, 36, "Implantação\nAPI gerenciada · servidor\npróprio · borda", fc="#FEF3C7", ec=AMB, fs=8.6, bold=True)
    rbox(ax, 35, 10, 30, 36, "Operação\nlimites de taxa · escalonamento\nSLOs · testes de carga", fc="#FCE7F3", ec=PNK, fs=8.6, bold=True)
    rbox(ax, 68, 10, 30, 36, "Observabilidade\nmétricas · logs · rastros\nderiva · versões · reversão", fc="#EDE9FE", ec=VIO, fs=8.6, bold=True)
    arrow(ax, (50, 65.5), (50, 47), c=GRY, ls="--"); arrow(ax, (83, 65.5), (83, 47), c=GRY, ls="--"); arrow(ax, (17, 65.5), (17, 47), c=GRY, ls="--")
    save(fig, "f15_mapa")


def implantacao():
    fig, ax = canvas(9.4, 4.4)
    cols = [("API gerenciada", IND, ["pouca operação", "custo por token", "dados saem da rede", "versão do provedor"]),
            ("Servidor próprio", GRN, ["controle de dados e versão", "equipe e equipamentos", "capacidade ociosa custa", "bom em volume estável"]),
            ("Borda", AMB, ["latência baixa", "dados ficam locais", "modelo pequeno (quantizado)", "atualização difícil"])]
    for i, (t, c, itens) in enumerate(cols):
        x = 2 + i * 33
        rbox(ax, x, 78, 30, 16, t, fc=c, ec=c, tc="white", fs=10, bold=True)
        for k, it in enumerate(itens):
            rbox(ax, x, 60 - k * 14, 30, 11, it, fc="white", ec=c, fs=8.4)
    ax.text(50, 2, "ou uma combinação: o comum na borda ou no servidor próprio, o difícil na API", ha="center", fontsize=8.8, color=GRY, style="italic")
    save(fig, "f15_implantacao")


def observ():
    fig, ax = canvas(9.4, 4.0)
    rbox(ax, 2, 40, 16, 22, "Pedido", fc=IND, ec=IND, tc="white", fs=10, bold=True)
    for k, (t, c) in enumerate([("métricas\np50/p95/p99 · custo", CYA), ("logs\nversões · tokens", GRN), ("rastros\netapas e tempos", AMB)]):
        y = 72 - k * 26; rbox(ax, 26, y - 8, 22, 17, t, fc="white", ec=c, fs=8.4, bold=True); arrow(ax, (18.4, 51), (25.6, y), c=GRY, lw=1.2, ms=9)
    rbox(ax, 56, 62, 20, 22, "Alertas\nSLOs · orçamento\nde erro", fc="#FEE2E2", ec=RED, fs=8.6, bold=True)
    rbox(ax, 54, 22, 24, 22, "Deriva\nPSI · KS\nqualidade\namostrada", fc="#FEF3C7", ec=AMB, fs=8.6, bold=True)
    rbox(ax, 82, 36, 17, 30, "Ação\navaliar\ncorrigir\nnova versão\nreversão", fc="#D1FAE5", ec=GRN, fs=8.4, bold=True)
    for y0 in (72, 46, 20): pass
    arrow(ax, (48.4, 72), (55.6, 74), c=INK, lw=1.2, ms=9); arrow(ax, (48.4, 46), (55.6, 70), c=INK, lw=1.2, ms=9); arrow(ax, (48.4, 46), (55.6, 36), c=INK, lw=1.2, ms=9); arrow(ax, (48.4, 20), (55.6, 30), c=INK, lw=1.2, ms=9)
    arrow(ax, (76.4, 73), (81.6, 55), c=INK); arrow(ax, (76.4, 33), (81.6, 47), c=INK)
    save(fig, "f15_observ")


# ------------------------------------------------------------------ figuras baseadas em dados
def roofline():
    BW, PICO, N = 2.0e12, 3.0e14, 7e9; KV = 2 * 32 * 32 * 128 * 2
    bs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]); T = 1000
    tm = (N * 2 + bs * T * KV) / BW * 1e3; tc = bs * 2 * N / PICO * 1e3
    tm4 = (N * 0.5 + bs * T * KV) / BW * 1e3
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(bs, tm, "o-", c=IND, label="leitura de memória (pesos float16 + cache)")
    ax.plot(bs, tm4, "s--", c=CYA, label="leitura de memória (pesos 4 bits + cache float16)")
    ax.plot(bs, tc, "^-", c=AMB, label="cálculo ($2Nb/P$)")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequências simultâneas (lote)"); ax.set_ylabel("tempo de um passo (ms)")
    ax.set_title("Decodificação na máquina hipotética: memória domina", fontsize=9.5); ax.legend(fontsize=7.5, frameon=False)
    save(fig, "f15_roofline")


def quantizar(W, bits=8, eixo=None, grupo=None):
    qmax = 2 ** (bits - 1) - 1
    if grupo: Wg = W.reshape(-1, grupo); s = np.abs(Wg).max(axis=1, keepdims=True) / qmax
    elif eixo is None: Wg, s = W, np.abs(W).max() / qmax
    else: Wg, s = W, np.abs(W).max(axis=eixo, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s); return np.clip(np.round(Wg / s), -qmax, qmax), s


def reconstruir(W, bits=8, **kw):
    q, s = quantizar(W, bits, **kw); return (q * s).reshape(W.shape)


def quant():
    rng = np.random.default_rng(0); W = rng.normal(0, 0.02, (256, 512)); Wo = W.copy(); Wo[:, ::64] *= 30
    er = lambda w, wq: float(np.sqrt(np.mean((w - wq) ** 2)) / np.sqrt(np.mean(w ** 2)))
    bits = [8, 6, 4, 3, 2]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axs[0]
    for nome, kw, c, ls in (("tensor", {}, IND, "-"), ("coluna", {"eixo": 0}, GRN, "-"), ("grupo 64", {"grupo": 64}, AMB, "-")):
        ax.plot(bits, [er(Wo, reconstruir(Wo, b, **kw)) for b in bits], "o" + ls, c=c, label=f"com outliers, escala por {nome}")
    ax.plot(bits, [er(W, reconstruir(W, b)) for b in bits], "o--", c=GRY, label="sem outliers, por tensor")
    ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("bits"); ax.set_ylabel("erro relativo (RMSE/RMS)"); ax.set_title("Erro nos pesos sintéticos", fontsize=9.5); ax.legend(fontsize=7, frameon=False)
    fr = corpus.frases_lm(3000, seed=0); te = corpus.frases_lm(600, seed=1); vocab = sorted({t for f in fr for t in f} | {"<s>"})
    lm = MLPLM(vocab, ctx=3, d=16, h=48, seed=0); lm.treinar(fr, epocas=8); P0 = {k: v.copy() for k, v in lm.P.items()}; ppl0 = lm.perplexidade(te)
    def ppl(b, **kw):
        for k in ("E", "W1", "W2"): lm.P[k] = reconstruir(P0[k], b, **kw)
        return lm.perplexidade(te)
    ax = axs[1]
    ax.plot(bits, [ppl(b) for b in bits], "o-", c=IND, label="escala por tensor"); ax.plot(bits, [ppl(b, eixo=0) for b in bits], "o-", c=GRN, label="escala por coluna")
    ax.axhline(ppl0, c=GRY, ls="--", label="original"); ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("bits"); ax.set_ylabel("perplexidade no teste"); ax.set_title("Perplexidade do MLPLM quantizado", fontsize=9.5); ax.legend(fontsize=7.5, frameon=False)
    save(fig, "f15_quant")


def cache():
    ENT = ["fatura", "nota fiscal", "boleto", "serviço", "aplicativo", "contrato", "internet", "1042", "8873", "2201", "5560", "premium", "básico", "família", "empresarial"]
    ents = lambda t: tuple(e for e in ENT if e in t.lower())
    X, y = corpus.intencoes(100, seed=5); Z = LSAEmbedder(dim=30).ajustar(X).transformar(X)
    def run(tau, ex=False):
        cz, ck, ce, hit, err = [], [], [], 0, 0
        for t, r, z in zip(X, y, Z):
            k = (r, ents(t))
            if cz:
                s = np.array(cz) @ z; j = int(s.argmax())
                if s[j] >= tau and (not ex or ce[j] == k[1]): hit += 1; err += ck[j] != k; continue
            cz.append(z); ck.append(k); ce.append(k[1])
        return hit / len(X), err / max(hit, 1)
    taus = [0.99, 0.95, 0.90, 0.80, 0.60]; a = [run(t) for t in taus]; b = [run(t, True) for t in taus]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(taus, [h for h, _ in a], "o-", c=IND, label="acerto do cache"); ax.plot(taus, [e for _, e in a], "o-", c=RED, label="erro entre os acertos")
    ax.plot(taus, [h for h, _ in b], "o--", c=IND, alpha=.6, label="acerto (com verificação de entidades)"); ax.plot(taus, [e for _, e in b], "o--", c=RED, alpha=.6, label="erro (com verificação de entidades)")
    ax.invert_xaxis(); ax.set_xlabel("limiar de similaridade $\\tau$ (cosseno)"); ax.set_ylabel("fração"); ax.set_ylim(-0.02, 1.4); ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    ax.set_title("Cache semântico: acertos e erros (corpus sintético)", fontsize=9.5); save(fig, "f15_cache")


def lote():
    rng = np.random.default_rng(0); L = np.clip(rng.geometric(1 / 60, size=400), 5, 400)
    est = lambda B: int(sum(L[i:i + B].max() for i in range(0, len(L), B)))
    def con(B):
        fila, at, p = list(L[::-1]), [], 0
        while fila or at:
            while fila and len(at) < B: at.append(fila.pop())
            p += 1; at = [r - 1 for r in at if r - 1 > 0]
        return p
    Bs = [1, 2, 4, 8, 16, 32]; u = L.sum()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(Bs, [u / (est(B) * B) for B in Bs], "o-", c=RED, label="lote estático"); ax.plot(Bs, [u / (con(B) * B) for B in Bs], "o-", c=GRN, label="lote contínuo")
    ax.set_xscale("log", base=2); ax.set_xlabel("tamanho do lote $B$"); ax.set_ylabel("fração dos espaços usados"); ax.set_ylim(0, 1.05); ax.legend(frameon=False)
    ax.set_title("Ocupação do lote (saídas de comprimento variável)", fontsize=9.5); save(fig, "f15_lote")


def simular_lote(lam, B, T0=1.0, T1=0.1, h=0.0, n=40_000, seed=0, t_hit=0.01):
    rng = np.random.default_rng(seed); ch = np.cumsum(rng.exponential(1 / lam, n)); hit = rng.random(n) < h
    lat = np.full(n, t_hit); idx = np.where(~hit)[0]; a = ch[idx]; m = len(a); livre, i = 0.0, 0
    while i < m:
        ini = max(livre, a[i]); j = i
        while j < m and a[j] <= ini and j - i < B: j += 1
        livre = ini + T0 + T1 * (j - i); lat[idx[i:j]] = livre - a[i:j]; i = j
    return lat[n // 20:]


def filas():
    lams = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for nome, B, h, c, ls in (("B=1", 1, 0.0, RED, "-"), ("B=1 + cache 40%", 1, 0.4, AMB, "-"), ("B=8", 8, 0.0, CYA, "-"), ("B=8 + cache 40%", 8, 0.4, GRN, "-")):
        ax.plot(lams, [min(simular_lote(l, B, h=h).mean(), 1e3) for l in lams], "o" + ls, c=c, label=nome)
    ax.set_yscale("log"); ax.set_ylim(0.3, 1.5e3); ax.axvline(1 / 1.1, c=GRY, ls=":"); ax.text(1 / 1.1 + .03, 0.4, "capacidade\nsem lote", fontsize=7.5, color=GRY)
    ax.set_xlabel("carga: pedidos por segundo"); ax.set_ylabel("latência média (s; truncada em 1000)"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Latência × carga (simulação, parâmetros hipotéticos)", fontsize=9.5); save(fig, "f15_filas")


def especulativa():
    ganho = lambda a, g, c: (1 - a ** (g + 1)) / ((1 - a) * (g * c + 1))
    al = np.linspace(0.3, 0.98, 60); fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for g, c in ((2, IND), (4, CYA), (8, GRN), (16, AMB)): ax.plot(al, [ganho(a, g, 0.05) for a in al], c=c, label=f"$\\gamma={g}$")
    ax.axhline(1, c=GRY, ls="--"); ax.set_xlabel("taxa de aceitação $\\alpha$"); ax.set_ylabel("ganho em tempo ($c=0{,}05$)"); ax.legend(frameon=False)
    ax.set_title("Decodificação especulativa: ganho esperado", fontsize=9.5); save(fig, "f15_especulativa")


def roteamento():
    V1 = ("cancelamento", "segunda_via", "suporte_tecnico")
    def sinais(t):
        t = _sem_acentos(t); h = {k: sum(w in t for w in ws) for k, ws in _PALAVRAS.items()}; p1 = sorted((h[k] for k in V1), reverse=True)
        return [p1[0], p1[1], sum(h[k] for k in h if k not in V1), len(t.split())]
    def rodar(sem, n=60):
        X, y = corpus.intencoes(n, seed=sem); b1, b3 = SimuladorLLM("v1", 0), SimuladorLLM("v3", 0); b = []
        for t in X:
            r = b1.responder(t); a = [c for c in INTENCOES if c.replace("_", " ") in r]; b.append(a[0] if a else "indefinido")
        return X, np.array(y), np.array(b), np.array([json.loads(b3.responder(t))["intencao"] for t in X])
    Xa, ya, ba, ca = rodar(61); Xt, yt, bt, ct = rodar(62)
    conf = LogisticRegression(max_iter=1000).fit([sinais(t) for t in Xa], ba == ya).predict_proba([sinais(t) for t in Xt])[:, 1]
    okb, okc = bt == yt, ct == yt; rng = np.random.default_rng(0); custo, acc, accs = [], [], []
    for tau in (0.0, 0.3, 0.6, 0.8, 0.95, 1.01):
        esc = conf < tau; custo.append(1.0 + 10.0 * esc.mean()); acc.append(np.where(esc, okc, okb).mean())
        accs.append(np.mean([np.where(rng.random(len(esc)) < esc.mean(), okc, okb).mean() for _ in range(300)]))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(custo, acc, "o-", c=IND, label="roteamento por confiança"); ax.plot(custo, accs, "o--", c=GRY, label="roteamento por sorteio")
    ax.axhline(okc.mean(), c=GRN, ls=":"); ax.text(1.2, okc.mean() + .015, "sempre o modelo caro (acerto)", fontsize=7.5, color=GRN)
    ax.scatter([10.0], [okc.mean()], c=GRN, zorder=5); ax.text(10.0, okc.mean() - .07, "caro\n(custo 10)", fontsize=7.5, color=GRN, ha="center")
    ax.set_xlabel("custo médio por requisição (u.m. hipotéticas)"); ax.set_ylabel("acerto"); ax.legend(frameon=False, loc="lower right"); ax.set_ylim(0.3, 1.02)
    ax.set_title("Custo × qualidade do roteamento (SIMULADOR, não LLM)", fontsize=9.5); save(fig, "f15_roteamento")


if __name__ == "__main__":
    mapa(); implantacao(); observ(); roofline(); quant(); cache(); lote(); filas(); especulativa(); roteamento()
