"""Figuras do Capítulo 5. f05_lr, f05_curvas e f05_escala são baseadas em dados/cálculos; f05_mapa e f05_treino são esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *
from llmdidatico import corpus, modelos


def mapa():
    fig, ax = canvas(9.4, 4.4)
    rbox(ax, 1, 58, 20, 32, "Dados\ncoleta, limpeza,\nfiltro, dedup.,\ndocumentação", fc="#E0E7FF", ec=IND, fs=8.6, bold=True)
    rbox(ax, 26, 58, 22, 32, "Ciclo de\ntreinamento\nautossupervisionado", fc="#CFFAFE", ec=CYA, fs=9, bold=True)
    rbox(ax, 53, 58, 20, 32, "Modelo-base\n(continua texto)", fc=IND, ec=IND, tc="white", fs=9, bold=True)
    rbox(ax, 78, 58, 21, 32, "Pós-treinamento\n(Cap. 7)", fc="#FEF3C7", ec=AMB, fs=9, bold=True)
    for x in (21.5, 48.5, 73.5): arrow(ax, (x, 74), (x + 4, 74), c=INK)
    rbox(ax, 1, 8, 30, 34, "Custo\nFLOPs ≈ 6·N·D\nmemória ≈ 16 B/param.\nenergia", fc="#FCE7F3", ec=PNK, fs=8.6, bold=True)
    rbox(ax, 36, 8, 30, 34, "Contaminação\ntreino × teste", fc="#FEE2E2", ec=RED, fs=9, bold=True)
    rbox(ax, 71, 8, 28, 34, "Avaliação\ne generalização\n(Cap. 16)", fc="#ECFDF5", ec=GRN, fs=9, bold=True)
    arrow(ax, (37, 57.5), (37, 43), c=GRY, ls="--"); arrow(ax, (63, 57.5), (51, 43), c=GRY, ls="--"); arrow(ax, (66.5, 25), (70.5, 25), c=GRY, ls="--")
    save(fig, "f05_mapa")


def treino():
    fig, ax = canvas(9.2, 4.0)
    passos = [("1. Lote\nde texto", IND), ("2. Passada\npara frente", CYA), ("3. Perda\n(entropia\ncruzada)", GRN), ("4. Passada\npara trás", AMB), ("5. Otimizador\n(AdamW)", PNK)]
    for k, (t, c) in enumerate(passos):
        x = 2 + k * 19.5; rbox(ax, x, 50, 17, 30, t, fc=c, ec=c, tc="white", fs=8.8, bold=True)
        if k < 4: arrow(ax, (x + 17.3, 65), (x + 19.2, 65), c=INK)
    arrow(ax, (92, 49.5), (92, 30), c=INK, style="-"); arrow(ax, (92, 30), (10, 30), c=INK, style="-"); arrow(ax, (10, 30), (10, 49.5), c=INK)
    ax.text(50, 24, "repetir milhões de vezes, com agenda de taxa de aprendizado e recorte do gradiente", ha="center", fontsize=8.6, color=INK)
    rbox(ax, 22, 3, 56, 14, "a cada N passos: validação em dados separados", fc="#EEF2FF", ec=IND, fs=8.6, bold=True)
    save(fig, "f05_treino")


def lr():
    S, w = 1000, 100; s = np.arange(1, S + 1)
    eta = np.where(s <= w, 1e-3 * s / w, 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + np.cos(np.pi * (s - w) / (S - w))))
    fig, ax = plt.subplots(figsize=(6.2, 3.4)); ax.plot(s, eta, c=IND, lw=2.4); ax.axvline(w, c=GRY, ls="--", lw=1); ax.text(w + 15, 1.02e-3, "fim do aquecimento", fontsize=8, color=GRY)
    ax.set_xlabel("passo de treinamento"); ax.set_ylabel("taxa de aprendizado"); ax.set_title("Aquecimento linear e decaimento em cosseno"); ax.set_ylim(0, 1.15e-3)
    save(fig, "f05_lr")


def _dados():
    frases = corpus.frases_lm(12000, seed=0); val = frases[-1000:]; vocab = sorted({t for f in frases for t in f} | {"<s>"})
    return frases, val, vocab


def curvas():
    frases, val, vocab = _dados(); m = modelos.MLPLM(vocab, ctx=3, d=16, h=64, seed=0); tr_l, va_l = [], []
    for ep in range(1, 13):
        h = m.treinar(frases[:400], epocas=1, lote=64, lr=0.01, seed=ep); tr_l.append(h[0])
        X, y = m.dados(val); _, _, z = m.frente(X); va_l.append(float(-modelos.log_softmax(z)[np.arange(len(y)), y].mean()))
    fig, ax = plt.subplots(figsize=(6.2, 3.6)); ax.plot(range(1, 13), tr_l, "-o", c=IND, label="treino"); ax.plot(range(1, 13), va_l, "-o", c=RED, label="validação")
    ax.set_xlabel("época"); ax.set_ylabel("perda (nats por token)"); ax.set_title("Curvas de perda do modelo-brinquedo"); ax.legend(); ax.set_ylim(0.8, 2.2)
    save(fig, "f05_curvas")


def escala():
    frases, val, vocab = _dados(); ns = [20, 50, 100, 200, 400, 800, 1600, 3200]; pt, pv = [], []
    for n in ns:
        mm = modelos.MLPLM(vocab, ctx=3, d=16, h=64, seed=0); mm.treinar(frases[:n], epocas=int(max(6, min(60, 18000 / n))), lote=64, lr=0.01)
        pt.append(np.log(mm.perplexidade(frases[:min(n, 400)]))); pv.append(np.log(mm.perplexidade(val)))
    fig, ax = plt.subplots(figsize=(6.2, 3.7)); ax.plot(ns, pt, "-o", c=IND, label="treino"); ax.plot(ns, pv, "-o", c=RED, label="validação"); ax.fill_between(ns, pt, pv, color=AMB, alpha=0.18, label="lacuna (sobreajuste)")
    ax.set_xscale("log"); ax.set_xlabel("frases de treino"); ax.set_ylabel("perda (nats por token)"); ax.legend(); ax.set_title("Mais dados, menos sobreajuste")
    save(fig, "f05_escala")


if __name__ == "__main__":
    mapa(); treino(); lr(); curvas(); escala()
