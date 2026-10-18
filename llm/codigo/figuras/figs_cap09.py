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


def memoria():
    # N*b/8 pesos + KV por usuario: 2*L*T*n_kv*d*2 bytes (modelo hipotetico 7 bi, 32 camadas, 8 cabecas KV de 128, T=4096)
    u = np.arange(0, 41); kv = 2 * 32 * 4096 * 8 * 128 * 2 / 1e9
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for bits, c in [(16, RED), (8, AMB), (4, IND)]:
        ax.plot(u, 7e9 * bits / 8 / 1e9 + u * kv, c=c, label=f"pesos em {bits} bits + cache")
    ax.axhline(24, c=GRY, ls="--", lw=1); ax.text(0.5, 24.8, "GPU hipotética de 24 GB", fontsize=8, color=GRY)
    ax.set_ylim(0, 60); ax.set_xlabel("usuários simultâneos (4 096 tokens cada)"); ax.set_ylabel("memória (GB)")
    ax.set_title("Memória: pesos + cache KV (modelo hipotético de 7 bi)", fontsize=9.5); ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.9)); save(fig, "f09_memoria")


def disjuntor():
    fig, ax = canvas(8.6, 3.2, (0, 100), (0, 40))
    rbox(ax, 3, 12, 24, 16, "Fechado\nchamadas passam", fc="#D1FAE5", ec=GRN, fs=9, bold=True)
    rbox(ax, 38, 12, 24, 16, "Aberto\nplano B direto", fc="#FEE2E2", ec=RED, fs=9, bold=True)
    rbox(ax, 73, 12, 24, 16, "Semiaberto\numa chamada de teste", fc="#FEF3C7", ec=AMB, fs=8.6, bold=True)
    arrow(ax, (27.5, 24), (37.5, 24), c=INK); ax.text(32.5, 30, "falhas seguidas\n≥ limite", ha="center", fontsize=7.5)
    arrow(ax, (62.5, 24), (72.5, 24), c=INK); ax.text(67.5, 30, "tempo de\nresfriamento", ha="center", fontsize=7.5)
    arrow(ax, (80, 11.5), (20, 11.5), c=GRN, ls="--"); ax.text(50, 4, "teste bem-sucedido: volta ao normal", ha="center", fontsize=7.5, color=GRN)
    arrow(ax, (73.5, 16), (62.5, 16), c=RED, ls="--"); ax.text(68, 8.5, "falha: reabre", ha="center", fontsize=7.5, color=RED)
    save(fig, "f09_disjuntor")


def validacao():
    fig, ax = canvas(9.0, 3.2, (0, 100), (0, 36))
    pas = [("Entrada", "mascarar\nlimitar tokens", IND), ("Chamada", "tempo limite\nrecuo", CYA), ("Formato", "JSON válido?\nesquema", GRN), ("Conteúdo", "sem dados pessoais\ncita fonte?", AMB), ("Plano B", "abster ou\nencaminhar", PNK)]
    for k, (a, b, c) in enumerate(pas):
        x = 2 + k * 19.6; rbox(ax, x, 6, 17, 22, "", fc="white", ec=c, lw=1.8)
        ax.text(x + 8.5, 22.5, a, ha="center", fontsize=9, fontweight="bold", color=c); ax.text(x + 8.5, 13, b, ha="center", va="center", fontsize=7.8, color=INK)
        if k < 4: arrow(ax, (x + 17.3, 17), (x + 19.4, 17), c=INK, lw=1.2, ms=8)
    save(fig, "f09_validacao")


if __name__ == "__main__":
    mapa(); camadas(); backoff(); memoria(); disjuntor(); validacao()
