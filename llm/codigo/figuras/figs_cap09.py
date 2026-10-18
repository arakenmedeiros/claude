"""Figuras do Capítulo 9. f09_backoff é calculada (fórmula do recuo); as demais são esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build")
from fc import *


def mapa():
    fig, ax = canvas(9.4, 4.2)
    rbox(ax, 36, 66, 28, 26, "Lógica da aplicação\nprompt · validação\nplano B", fc=IND, ec=IND, tc="white", fs=9, bold=True)
    rbox(ax, 36, 34, 28, 18, "Interface\nProvedorLLM", fc="#E0E7FF", ec=IND, fs=9, bold=True)
    arrow(ax, (50, 65.5), (50, 52.5), c=INK)
    for k, (t, c) in enumerate([("Adaptador\nserviço por API", CYA), ("Adaptador\nmodelo local", GRN), ("Simulado\n(testes)", AMB)]):
        x = 3 + k * 33; rbox(ax, x, 4, 30, 18, t, fc="white", ec=c, fs=8.6, bold=True); arrow(ax, (x + 15, 22.5), (50, 33.5), c=GRY, ls="--")
    rbox(ax, 70, 62, 28, 30, "Robustez\nrecuo · tempo limite\norçamento · disjuntor", fc="#FEF3C7", ec=AMB, fs=8.2, bold=True)
    rbox(ax, 2, 62, 28, 30, "Segurança\nsegredos · mascaramento\nlicenças", fc="#FCE7F3", ec=PNK, fs=8.2, bold=True)
    arrow(ax, (30.5, 77), (35.5, 77), c=GRY, ls="--"); arrow(ax, (69.5, 77), (64.5, 77), c=GRY, ls="--")
    save(fig, "f09_mapa")


def camadas():
    fig, ax = canvas(8.6, 3.8, (0, 100), (0, 62))
    ncam = [("Interface do usuário", "formulário · API · linha de comando", MID, INK), ("Lógica da aplicação", "recuperação · prompt · validação · registro", IND, "white"), ("Interface ProvedorLLM", "gerar(mensagens, parâmetros) → resposta", CYA, "white"), ("Adaptadores", "serviço por API · modelo local · simulado", GRN, "white")]
    for k, (a, b, c, tc) in enumerate(ncam):
        y = 46 - k * 15; rbox(ax, 4, y, 92, 12, "", fc=c, ec=c)
        ax.text(8, y + 6, a, fontsize=9.2, color=tc, fontweight="bold", va="center"); ax.text(92, y + 6, b, fontsize=8.2, color=tc, va="center", ha="right")
    for k in range(3): arrow(ax, (50, 46 - k * 15 - 0.2), (50, 46 - k * 15 - 3), c=INK, lw=1.2, ms=9)
    save(fig, "f09_camadas")


def backoff():
    k = np.arange(0, 8); t0, teto, j = 1.0, 30.0, 0.25
    base = np.minimum(teto, t0 * 2.0 ** k)
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.fill_between(k, base, base * (1 + j), color=CYA, alpha=0.3, label="faixa do jitter (até +25%)")
    ax.plot(k, base, "o-", c=IND, label="espera base min(30, 2^k)")
    ax.axhline(teto, c=RED, ls="--", lw=1, label="teto (30 s)")
    ax.set_xlabel("número da falha anterior (k)"); ax.set_ylabel("espera (s)"); ax.set_title("Recuo exponencial com teto e jitter (t0 = 1 s)", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left"); save(fig, "f09_backoff")


if __name__ == "__main__":
    mapa(); camadas(); backoff()
