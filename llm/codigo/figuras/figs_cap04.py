"""Figuras do Capítulo 4. f04_pesos, f04_escala, f04_mascara e f04_posicional são baseados em dados/cálculos; as demais, esquemas."""
import sys; sys.path.insert(0, "/home/user/claude/llm/build"); sys.path.insert(0, "/home/user/claude/llm/codigo/comum")
from fc import *
from llmdidatico import modelos
from matplotlib.colors import LinearSegmentedColormap
HEAT = LinearSegmentedColormap.from_list("h", ["#FFFFFF", "#C7D2FE", IND, "#1E1B4B"])


def mapa():
    fig, ax = canvas(9.4, 4.4)
    rbox(ax, 2, 62, 22, 28, "Consulta,\nchave e valor\n(Q, K, V)", fc="#E0E7FF", ec=IND, fs=10, bold=True)
    rbox(ax, 30, 62, 22, 28, "Atenção\nsoftmax(QKᵀ/√dk)V", fc="#CFFAFE", ec=CYA, fs=9.2, bold=True)
    rbox(ax, 58, 62, 18, 28, "Máscara\ncausal", fc="#FEF3C7", ec=AMB, fs=10, bold=True)
    rbox(ax, 80, 62, 18, 28, "Múltiplas\ncabeças", fc="#FCE7F3", ec=PNK, fs=10, bold=True)
    for x in (24.5, 52.5, 76.5): arrow(ax, (x, 76), (x + 5, 76), c=INK)
    rbox(ax, 2, 14, 22, 28, "Posições\n(senoidais,\nRoPE…)", fc="#D1FAE5", ec=GRN, fs=10, bold=True)
    rbox(ax, 30, 14, 22, 28, "Residuais e\nnormalização\n+ FFN", fc="#FEE2E2", ec=RED, fs=10, bold=True)
    rbox(ax, 58, 14, 40, 28, "Bloco Transformer × N\n(encoder · decoder ·\nencoder-decoder)", fc=IND, ec=IND, tc="white", fs=9, bold=True)
    arrow(ax, (24.5, 28), (29.5, 28), c=INK); arrow(ax, (52.5, 28), (57.5, 28), c=INK); arrow(ax, (41, 61.5), (41, 42.5), c=INK, ls="--"); arrow(ax, (89, 61.5), (89, 42.5), c=INK, ls="--")
    save(fig, "f04_mapa")


def qkv():
    fig, ax = canvas(9.4, 4.4)
    for k, (t, c) in enumerate([("o", IND), ("estudante", CYA), ("entrega", GRN), ("o", AMB), ("trabalho", PNK)]):
        x = 4 + k * 14.5; rbox(ax, x, 6, 13, 10, t, fc=c, ec=c, tc="white", fs=8.6, bold=True)
    ax.text(1.5, 20, "tokens (embeddings + posição)", fontsize=8.5, color=GRY)
    rbox(ax, 4, 32, 21, 12, "consulta do token 5:\nq₅ = x₅W_Q", fc="#FCE7F3", ec=PNK, fs=8.4, bold=True)
    rbox(ax, 28, 32, 22, 12, "chaves k₁…k₅\nK = XW_K", fc="#E0E7FF", ec=IND, fs=8.4, bold=True)
    rbox(ax, 53, 32, 22, 12, "valores v₁…v₅\nV = XW_V", fc="#D1FAE5", ec=GRN, fs=8.4, bold=True)
    rbox(ax, 4, 54, 46, 12, "pontuações q₅·kⱼ/√dk → softmax\n→ pesos (somam 1)", fc="#CFFAFE", ec=CYA, fs=8.2, bold=True)
    rbox(ax, 56, 54, 41, 12, "saída₅ = Σ pesoⱼ · vⱼ\n(média ponderada)", fc="#FEF3C7", ec=AMB, fs=8.2, bold=True)
    for xx in (14, 39, 64): arrow(ax, (xx, 17), (xx, 31.5), c=GRY, lw=1.2, ms=9)
    arrow(ax, (14.5, 44.5), (14.5, 53.5), c=INK); arrow(ax, (39, 44.5), (39, 53.5), c=INK); arrow(ax, (50.5, 60), (55.5, 60), c=INK); arrow(ax, (64, 44.5), (64, 53.5), c=INK)
    rbox(ax, 56, 78, 41, 14, "novo vetor do token 5,\nenriquecido pelo contexto", fc=IND, ec=IND, tc="white", fs=8.6, bold=True); arrow(ax, (76, 66.5), (76, 77.5), c=INK)
    save(fig, "f04_qkv")


def pesos():
    Q = K = np.array([[1., 0], [0, 1], [1, 1]]); V = np.array([[1., 0], [0, 2], [3, 3]])
    _, A1 = modelos.atencao(Q, K, V)
    rng = np.random.default_rng(4); Q2 = rng.normal(size=(6, 8)); K2 = rng.normal(size=(6, 8)); V2 = rng.normal(size=(6, 8))
    _, A2 = modelos.atencao(Q2, K2, V2, mascara_causal=True)
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, A, ttl, labs in [(axs[0], A1, "Exemplo de 3 tokens (sem máscara)", ["1", "2", "3"]), (axs[1], A2, "6 tokens, aleatório, com máscara causal", [str(i) for i in range(1, 7)])]:
        im = ax.imshow(A, cmap=HEAT, vmin=0, vmax=1)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]): ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if A[i, j] > 0.5 else INK)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs); ax.set_xlabel("token consultado (chave)"); ax.set_ylabel("token que consulta"); ax.set_title(ttl, fontsize=9)
    save(fig, "f04_pesos")


def escala():
    rng = np.random.default_rng(0); dks = [4, 16, 64, 256, 1024]; sem, com = [], []
    for dk in dks:
        q = rng.normal(size=(2000, 1, dk)); k = rng.normal(size=(2000, 10, dk)); s = (q * k).sum(-1)
        sem.append(modelos.softmax(s).max(1).mean()); com.append(modelos.softmax(s / np.sqrt(dk)).max(1).mean())
    fig, ax = plt.subplots(figsize=(6.2, 3.6)); ax.plot(dks, sem, "-o", c=RED, label="sem escala"); ax.plot(dks, com, "-o", c=IND, label="com escala 1/√dk")
    ax.axhline(0.1, c=GRY, ls="--", lw=1); ax.text(5, 0.12, "uniforme (0,1)", fontsize=8, color=GRY)
    ax.set_xscale("log", base=2); ax.set_xlabel("dimensão dk"); ax.set_ylabel("peso máximo médio do softmax"); ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Sem escala, o softmax satura")
    save(fig, "f04_escala")


def mascara():
    T = 6; bi = np.ones((T, T)); ca = np.tril(np.ones((T, T)))
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.5)); cm = LinearSegmentedColormap.from_list("m", ["#F3F4F6", IND])
    for ax, M, t in [(axs[0], bi, "Bidirecional (encoder)"), (axs[1], ca, "Causal (decoder)")]:
        ax.imshow(M, cmap=cm, vmin=0, vmax=1)
        for i in range(T):
            for j in range(T): ax.text(j, i, "✓" if M[i, j] else "✗", ha="center", va="center", color="white" if M[i, j] else GRY, fontsize=10)
        ax.set_xticks(range(T)); ax.set_xticklabels(range(1, T + 1)); ax.set_yticks(range(T)); ax.set_yticklabels(range(1, T + 1)); ax.set_xlabel("token consultado"); ax.set_ylabel("token que consulta"); ax.set_title(t, fontsize=10)
    save(fig, "f04_mascara")


def posicional():
    T, d = 40, 32; pos = np.arange(T)[:, None]; i = np.arange(d // 2)[None, :]; ang = pos / (10000 ** (2 * i / d)); PE = np.zeros((T, d)); PE[:, 0::2] = np.sin(ang); PE[:, 1::2] = np.cos(ang)
    fig, ax = plt.subplots(figsize=(7, 3.8)); im = ax.imshow(PE, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1); ax.set_xlabel("dimensão"); ax.set_ylabel("posição"); fig.colorbar(im, ax=ax, fraction=0.03); ax.set_title("Codificação posicional senoidal (40 posições, 32 dimensões)", fontsize=10)
    save(fig, "f04_posicional")


def bloco():
    fig, ax = canvas(8.6, 6.0, (0, 100), (0, 76))
    rbox(ax, 8, 2, 36, 7, "tokens → embeddings + posições", fc="#E0E7FF", ec=IND, fs=8.6, bold=True)
    ax.add_patch(FancyBboxPatch((4, 12), 44, 52, boxstyle="round,pad=0,rounding_size=2", fc="#F9FAFB", ec=GRY, lw=1.4, ls="--")); ax.text(6, 61, "bloco × N", fontsize=8.5, color=GRY, fontweight="bold")
    rbox(ax, 10, 16, 32, 6, "LayerNorm", fc="#FEF3C7", ec=AMB, fs=8.4); rbox(ax, 10, 25, 32, 8, "Atenção multicabeça\n(causal)", fc="#CFFAFE", ec=CYA, fs=8.4, bold=True)
    ax.text(26, 36.5, "+", ha="center", fontsize=14, fontweight="bold", color=RED)
    rbox(ax, 10, 41, 32, 5, "LayerNorm", fc="#FEF3C7", ec=AMB, fs=8.4); rbox(ax, 10, 49, 32, 8, "FFN\n(expande e comprime)", fc="#FCE7F3", ec=PNK, fs=8.4, bold=True)
    ax.text(26, 59.5, "+", ha="center", fontsize=14, fontweight="bold", color=RED)
    arrow(ax, (26, 9.5), (26, 15.5), c=INK); arrow(ax, (26, 22.5), (26, 24.5), c=INK); arrow(ax, (26, 33.5), (26, 35.5), c=INK); arrow(ax, (26, 38), (26, 40.5), c=INK); arrow(ax, (26, 46.5), (26, 48.5), c=INK); arrow(ax, (26, 57.5), (26, 59), c=INK)
    arrow(ax, (8, 14), (8, 36.5), c=RED, lw=1.4, rad=0.0, style="-"); arrow(ax, (8, 36.5), (24, 36.5), c=RED, lw=1.4); arrow(ax, (8, 38), (8, 59.5), c=RED, lw=1.4, style="-"); arrow(ax, (8, 59.5), (24, 59.5), c=RED, lw=1.4)
    rbox(ax, 10, 66, 32, 8, "LayerNorm final →\nprojeção ao vocabulário", fc="#D1FAE5", ec=GRN, fs=7.8, bold=True); arrow(ax, (26, 60.5), (26, 65.5), c=INK)
    rbox(ax, 56, 50, 40, 22, "Atalhos vermelhos:\nconexões residuais\ny = x + F(x)", fc="#FEE2E2", ec=RED, fs=9, bold=True)
    rbox(ax, 56, 24, 40, 20, "Parâmetros por bloco ≈ 12d²\n(atenção 4d² + FFN 8d²)", fc="#EEF2FF", ec=IND, fs=9, bold=True)
    rbox(ax, 56, 2, 40, 16, "Saída: logits do próximo token\npara cada posição", fc="#ECFDF5", ec=GRN, fs=9, bold=True)
    save(fig, "f04_bloco")


def arquiteturas():
    fig, axs = plt.subplots(1, 3, figsize=(9.4, 3.4)); T = 5
    cm = LinearSegmentedColormap.from_list("m", ["#F3F4F6", IND])
    axs[0].imshow(np.ones((T, T)), cmap=cm, vmin=0, vmax=1); axs[0].set_title("Somente encoder\n(bidirecional)", fontsize=9)
    axs[1].imshow(np.tril(np.ones((T, T))), cmap=cm, vmin=0, vmax=1); axs[1].set_title("Somente decoder\n(causal)", fontsize=9)
    M = np.zeros((T, 2 * T)); M[:, T:] = np.tril(np.ones((T, T))); M[:, :T] = 0.55
    axs[2].imshow(M, cmap=LinearSegmentedColormap.from_list("m2", ["#F3F4F6", CYA, IND]), vmin=0, vmax=1); axs[2].axvline(T - 0.5, c=INK, lw=1.4); axs[2].set_title("Encoder-decoder\n(à esq.: atenção cruzada ao encoder)", fontsize=8.6)
    for ax in axs: ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel("quem consulta"); ax.set_xlabel("quem é consultado")
    save(fig, "f04_arquiteturas")


if __name__ == "__main__":
    mapa(); qkv(); pesos(); escala(); mascara(); posicional(); bloco(); arquiteturas()
