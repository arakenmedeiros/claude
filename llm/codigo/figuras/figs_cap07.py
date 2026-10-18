"""Figuras do Capítulo 7. f07_bt, f07_kl e f07_dpo são baseadas em cálculos; as demais são esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *

sig = lambda x: 1 / (1 + np.exp(-x))


def mapa():
    fig, ax = canvas(9.4, 4.2)
    cx = [("Modelo-base\n(pré-treinado)", "#E0E7FF", IND), ("SFT\ninstrução → resposta", "#CFFAFE", CYA), ("Preferências\n(comparações)", "#FEF3C7", AMB), ("Modelo alinhado\nao comportamento\ndesejado", IND, IND)]
    for k, (t, fc_, ec) in enumerate(cx):
        x = 2 + k * 24.5; rbox(ax, x, 56, 21, 30, t, fc=fc_, ec=ec, tc="white" if fc_ == IND else INK, fs=8.8, bold=True)
        if k < 3: arrow(ax, (x + 21.4, 71), (x + 24.1, 71), c=INK)
    rbox(ax, 26, 8, 22, 30, "RLHF\nrecompensa + KL + RL", fc="#FCE7F3", ec=PNK, fs=8.6, bold=True)
    rbox(ax, 52, 8, 22, 30, "DPO\notimização direta", fc="#D1FAE5", ec=GRN, fs=8.8, bold=True)
    rbox(ax, 78, 8, 20, 30, "Não garante\ncorreção factual", fc="#FEE2E2", ec=RED, fs=8.6, bold=True)
    arrow(ax, (62, 55.5), (40, 39), c=GRY, ls="--"); arrow(ax, (62, 55.5), (62, 39), c=GRY, ls="--"); arrow(ax, (86, 55.5), (88, 39), c=GRY, ls="--")
    save(fig, "f07_mapa")


def pipeline():
    fig, ax = canvas(9.4, 3.6)
    et = [("1. Pré-treino", "texto bruto\npróximo token", IND), ("2. SFT", "demonstrações\nperda só na resposta", CYA), ("3. Preferências", "pares preferida/\nrejeitada", AMB), ("4. Avaliação e\nsegurança", "testes, auditoria,\nrevisão humana", GRN)]
    for k, (t, d, c) in enumerate(et):
        x = 2 + k * 24.5; rbox(ax, x, 50, 21, 20, t, fc=c, ec=c, tc="white", fs=9, bold=True); rbox(ax, x, 20, 21, 24, d, fc="white", ec=c, fs=8.2)
        if k < 3: arrow(ax, (x + 21.4, 60), (x + 24.1, 60), c=INK)
    ax.text(50, 8, "cada etapa usa dados e objetivos diferentes, e nenhuma garante correção factual", ha="center", fontsize=8.8, color=GRY, style="italic")
    save(fig, "f07_pipeline")


def bt():
    d = np.linspace(-6, 6, 300)
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
    axs[0].plot(d, sig(d), c=IND, lw=2.4); axs[0].axhline(0.5, c=GRY, lw=1, ls="--"); axs[0].set_xlabel("diferença de recompensas  r(preferida) − r(rejeitada)"); axs[0].set_ylabel("P(preferida vence)"); axs[0].set_title("Bradley-Terry", fontsize=10)
    axs[1].plot(d, -np.log(sig(d)), c=RED, lw=2.4); axs[1].set_xlabel("mesma diferença de recompensas"); axs[1].set_ylabel("perda  −ln σ(diferença)"); axs[1].set_title("Perda do modelo de recompensa", fontsize=10)
    axs[1].scatter([0.8], [-np.log(sig(0.8))], c=AMB, zorder=3); axs[1].text(1.0, 0.9, "exemplo do texto:\n0,8 → 0,371", fontsize=8)
    save(fig, "f07_bt")


def kl():
    pref = np.array([0.5, 0.3, 0.15, 0.05]); r = np.array([0., 1, 2, 3]); betas = [0.1, 0.5, 1.0, 5.0]; fig, axs = plt.subplots(1, 5, figsize=(9.6, 2.7), sharey=True)
    axs[0].bar(range(4), pref, color=GRY); axs[0].set_title("referência", fontsize=9)
    for ax, b in zip(axs[1:], betas):
        q = pref * np.exp(r / b); q /= q.sum(); ax.bar(range(4), q, color=[IND, CYA, GRN, AMB]); ax.set_title(f"β = {b}", fontsize=9)
        for i, v in enumerate(q): ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    for ax in axs: ax.set_xticks(range(4)); ax.set_xticklabels(["1", "2", "3", "4"]); ax.set_ylim(0, 1.05)
    axs[0].set_ylabel("probabilidade"); axs[2].set_xlabel("resposta candidata (recompensas 0, 1, 2, 3)")
    save(fig, "f07_kl")


def dpo():
    def ls(z): z = z - z.max(); return z - np.log(np.exp(z).sum())
    pref = np.array([0.5, 0.3, 0.15, 0.05]); r = np.array([0., 1, 2, 3]); lref = np.log(pref); rng = np.random.default_rng(0); N = 4000
    a, b = rng.integers(0, 4, N), rng.integers(0, 4, N); m = a != b; a, b = a[m], b[m]; va = rng.random(len(a)) < sig(r[a] - r[b]); W, L = np.where(va, a, b), np.where(va, b, a)
    th = lref.copy(); I = np.eye(4)
    for _ in range(3000):
        lp = ls(th); h = ((lp[W] - lref[W]) - (lp[L] - lref[L])); g = (-(1 - sig(h))[:, None] * (I[W] - I[L])).mean(0); th -= g
    pol = np.exp(ls(th)); alvo = pref * np.exp(r); alvo /= alvo.sum()
    fig, ax = plt.subplots(figsize=(6.4, 3.6)); x = np.arange(4); w = 0.27
    ax.bar(x - w, pref, w, label="referência", color=GRY); ax.bar(x, pol, w, label="DPO (aprendida)", color=IND); ax.bar(x + w, alvo, w, label="RLHF (teoria)", color=AMB)
    ax.set_xticks(x); ax.set_xticklabels(["1\n(r=0)", "2\n(r=1)", "3\n(r=2)", "4\n(r=3)"]); ax.set_ylabel("probabilidade"); ax.legend(fontsize=8); ax.set_title("DPO aproxima a solução do RLHF (β = 1)")
    save(fig, "f07_dpo")


if __name__ == "__main__":
    mapa(); pipeline(); bt(); kl(); dpo()
