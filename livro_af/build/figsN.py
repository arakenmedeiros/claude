from icons import *
import sys
sys.path.insert(0, "/home/user/claude/livro_af/cap/lib")
from avancado import *
from personalizacao import ifca


def f12_mapa():
    fig, ax = canvas(10, 3.7)
    itens = [("12.2", "Assíncrono", "quem é lento\nnão trava a rodada", IND), ("12.3", "Seleção", "quem aparece não é\numa amostra justa", CYA),
             ("12.4", "Avaliação", "clientes novos e\npior caso", GRN), ("12.5", "FedBN", "normalização\nfica em casa", AMB),
             ("12.6", "Esquecimento", "remover a influência\nde um cliente", RED), ("12.7", "Poucos rótulos", "pseudorrótulos e\ncautelas", VIO),
             ("12.8", "Multirrótulo", "rótulo ausente não\né negativo", PNK), ("12.9", "Deriva", "o mundo muda\ncom o tempo", IND)]
    for i, (sec, t, d, c) in enumerate(itens):
        x, y = 3 + (i % 4) * 24, 52 if i < 4 else 6
        rbox(ax, x, y, 22, 40, "", fc="white", ec=c, lw=2.2, r=2.5)
        ax.add_patch(FancyBboxPatch((x, y + 29), 22, 11, boxstyle="round,pad=0,rounding_size=2.5", fc=c, ec=c))
        ax.text(x + 11, y + 34.5, f"{sec}  {t}", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
        ax.text(x + 11, y + 14, d, ha="center", va="center", fontsize=9, color=INK, linespacing=1.4)
    save(fig, "f12_mapa")


def f12_assincrono():
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 4.6), sharex=True)
    dur = [1.0, 1.2, 1.0, 4.0]; cor = [IND, CYA, GRN, AMB]
    ax = axs[0]
    t = 0
    for r in range(2):
        fim = t + max(dur)
        for c, d in enumerate(dur):
            ax.barh(c, d, left=t, color=cor[c], ec="white", height=0.7)
            if d < max(dur): ax.barh(c, max(dur) - d, left=t + d, color="none", ec=GRY, hatch="///", height=0.7, lw=0)
        ax.axvline(fim, c=INK, ls="--", lw=1.1); t = fim
    ax.set_title("Síncrono: a rodada espera o mais lento (tracejado = ociosidade)", fontsize=9.5, loc="left")
    ax = axs[1]
    for c, d in enumerate(dur):
        x = 0
        while x < 8.2:
            ax.barh(c, d, left=x, color=cor[c], ec="white", height=0.7, alpha=0.95)
            ax.scatter(x + d, c, marker="v", s=40, c=INK, zorder=5); x += d
    ax.set_ylim(-0.6, 4.5); ax.annotate("resposta atrasada: levou 4 unidades,\no modelo já mudou várias vezes", (4.0, 3.35), (4.6, 3.95), fontsize=8, arrowprops=dict(arrowstyle="->", color=RED), color=RED)
    ax.set_title("Assíncrono: ninguém espera (▼ = resposta chega ao servidor)", fontsize=9.5, loc="left")
    for a in axs:
        a.set_yticks(range(4)); a.set_yticklabels(["cliente 1", "cliente 2", "cliente 3", "cliente 4 (lento)"], fontsize=8.5); a.set_xlim(0, 8.2)
    axs[1].set_xlabel("tempo (unidades virtuais)")
    save(fig, "f12_assincrono")


def _e1():
    Xbr, Xteb, ytr, yte = carregar_digits(bruto=True); K, T = 20, 60
    tempos = np.where(np.arange(K) < 14, 1.0, 6.0); curvas, fl = {}, {}
    for s in range(3):
        rng = np.random.default_rng(s); partes = [[] for _ in range(K)]
        for i in range(len(ytr)):
            c = rng.integers(14, 20) if (ytr[i] <= 2 and rng.random() < 0.85) else rng.integers(0, 20)
            partes[c].append(i)
        treino, teste = dividir_local([np.array(p) for p in partes], 0.3, s)
        Xtr, Xte = padronizar_clientes(Xbr, Xteb, treino)
        rap, len_ = np.concatenate(teste[:14]), np.concatenate(teste[14:])
        for nome, f in {"síncrono": lambda: simular_sincrono(treino, Xtr, ytr, Xte, yte, tempos, T=T, seed=s),
                        "FedAsync": lambda: simular_assincrono(treino, Xtr, ytr, Xte, yte, "fedasync", tempos, T=T, seed=s),
                        "FedBuff (K=5)": lambda: simular_assincrono(treino, Xtr, ytr, Xte, yte, "fedbuff", tempos, T=T, K=5, seed=s)}.items():
            W, cv = f(); curvas.setdefault(nome, []).append(cv[:T, 1])
            fl.setdefault(nome, []).append((acuracia(W, Xtr[rap], ytr[rap]), acuracia(W, Xtr[len_], ytr[len_])))
    return curvas, fl


def f12_assinc_res():
    curvas, fl = _e1(); cols = {"síncrono": GRY, "FedAsync": AMB, "FedBuff (K=5)": GRN}
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for nome, v in curvas.items(): axs[0].plot(np.mean(v, 0), c=cols[nome], lw=2.2, label=nome)
    axs[0].axhline(0.9, c=GRY, ls=":", lw=1); axs[0].set_ylim(0.5, 1.0); axs[0].set_xlabel("tempo (unidades virtuais)"); axs[0].set_ylabel("acurácia global"); axs[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    axs[0].set_title("Acurácia ao longo do tempo", fontsize=9.5)
    w = 0.26
    for i, (nome, v) in enumerate(fl.items()):
        m = np.mean(v, 0); axs[1].bar(np.arange(2) + (i - 1) * w, m, w, color=cols[nome], ec=INK, lw=0.5)
        for j in range(2): axs[1].text(j + (i - 1) * w, m[j] + 0.006, f"{m[j]:.2f}", ha="center", fontsize=7.5)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["clientes rápidos", "clientes lentos"]); axs[1].set_ylim(0.8, 0.98); axs[1].set_title("Quem fica para trás", fontsize=9.5)
    save(fig, "f12_assinc_res")


def f12_participacao():
    Xbr, Xteb, ytr, yte = carregar_digits(bruto=True); K = 30; pols = ("uniforme", "disponivel", "ponderado", "poder"); M = {p: [] for p in pols}
    for s in range(5):
        partes = particao_dirichlet(ytr, K, 0.1, s); treino, teste = dividir_local(partes, 0.3, s)
        ok = [i for i in range(K) if len(teste[i]) >= 5 and len(treino[i]) > 0]; treino = [treino[i] for i in ok]
        Xtr, Xte = padronizar_clientes(Xbr, Xteb, treino); disp = np.where(classe_dominante(treino, ytr) < 5, 0.95, 0.05)
        for p in pols:
            W = fedavg_selecao(treino, Xtr, ytr, Xte, yte, p, disp, seed=s); pr = np.argmax(Xte @ W, 1)
            M[p].append([np.mean(pr[yte == c] == c) for c in range(10)])
    A = np.array([np.mean(M[p], 0) for p in pols])
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    im = ax.imshow(A, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(4):
        for j in range(10): ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(10)); ax.set_xlabel("dígito (os dígitos 5 a 9 pertencem a clientes quase sempre indisponíveis)")
    ax.set_yticks(range(4)); ax.set_yticklabels(["uniforme (ideal)", "disponíveis", "ponderado", "power-of-choice"], fontsize=9)
    for s_ in ax.spines.values(): s_.set_visible(False)
    save(fig, "f12_participacao")


def f12_estilos():
    Xbr, Xteb, ytr, yte = carregar_digits(bruto=True); rng = np.random.default_rng(3)
    idx = np.where(ytr == 3)[0][:6]; partes = [np.array([i]) for i in idx]
    Xs, g, d = federacao_com_estilo(Xbr, partes, seed=5, intensidade=1.6)
    fig = plt.figure(figsize=(9.6, 3.4)); gs = fig.add_gridspec(2, 3, left=0.02, right=0.42, wspace=0.08, hspace=0.25)
    for k in range(6):
        ax = fig.add_subplot(gs[k // 3, k % 3]); ax.imshow(Xs[idx[k]].reshape(8, 8), cmap="gray_r", vmin=0, vmax=24); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"cliente {k + 1}", fontsize=7.5)
    ax2 = fig.add_axes([0.52, 0.16, 0.46, 0.72])
    dados = {"sem estilo": (0.950, 0.933, 0.875), "estilo\nmoderado": (0.927, 0.912, 0.839), "estilo\nforte": (0.818, 0.822, 0.526), "novos de\noutra população": (0.927, 0.793, 0.453)}
    x = np.arange(4); ax2.bar(x - 0.2, [v[0] for v in dados.values()], 0.4, color=IND, ec=INK, lw=0.5, label="conhecidos (média)")
    ax2.bar(x + 0.2, [v[1] for v in dados.values()], 0.4, color=AMB, ec=INK, lw=0.5, label="novos (média)")
    ax2.scatter(x + 0.2, [v[2] for v in dados.values()], marker="v", c=RED, s=45, zorder=5, label="pior cliente novo")
    ax2.set_xticks(x); ax2.set_xticklabels(list(dados.keys()), fontsize=8); ax2.set_ylim(0.4, 1.2); ax2.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3); ax2.set_title("Conhecidos × novos", fontsize=9.5)
    save(fig, "f12_estilos", tight=True)


def f12_fedbn():
    fig = plt.figure(figsize=(9.8, 3.5)); axd = fig.add_axes([0.0, 0.02, 0.52, 0.9]); axd.set_xlim(0, 100); axd.set_ylim(0, 60); axd.axis("off")
    blocos = [("entrada", LIGHT, INK, 2, 14), ("W1", IND, "white", 20, 14), ("normalização\n(g, b, média,\nvariância)", AMB, INK, 38, 26), ("ReLU", LIGHT, INK, 68, 12), ("W2", IND, "white", 84, 14)]
    for t, fc_, tc, x, w in blocos:
        rbox(axd, x, 18, w, 22, t, fc=fc_, ec=INK, fs=7.5, tc=tc, bold=True)
    for a, b in [(16, 20), (34, 38), (64, 68), (80, 84)]: arrow(axd, (a, 29), (b, 29), c=INK, lw=1.2, ms=8)
    rbox(axd, 2, 46, 44, 9, "indigo: agregado pelo servidor", fc="white", ec=IND, fs=7.5, tc=IND, bold=True)
    rbox(axd, 52, 46, 46, 9, "âmbar: LOCAL no FedBN", fc="white", ec=AMB, fs=7.5, tc="#B45309", bold=True)
    axd.text(50, 6, "FedBN: só as camadas lineares são agregadas", ha="center", fontsize=8.5, style="italic")
    ax = fig.add_axes([0.6, 0.14, 0.38, 0.74])
    r = {"sem estilos": [(0.951, 0.924), (0.940, 0.902)], "com estilos": [(0.876, 0.732), (0.944, 0.902)]}
    x = np.arange(2)
    for k, (nome, (a, b)) in enumerate([("FedAvg", (0, 0)), ("FedBN", (1, 1))]): pass
    for j, (cen, v) in enumerate(r.items()):
        for k, (nome, c) in enumerate([("FedAvg", GRY), ("FedBN", GRN)]):
            ax.bar(j + (k - 0.5) * 0.36, v[k][0], 0.34, color=c, ec=INK, lw=0.5, label=nome if j == 0 else None)
            ax.scatter(j + (k - 0.5) * 0.36, v[k][1], marker="v", c=RED, s=40, zorder=5, label="percentil 10" if (j == 0 and k == 0) else None)
            ax.text(j + (k - 0.5) * 0.36, v[k][0] + 0.005, f"{v[k][0]:.2f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(list(r.keys())); ax.set_ylim(0.6, 1.1); ax.set_title("Acurácia média por cliente", fontsize=9.5); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    save(fig, "f12_fedbn")


def f12_desaprendizado():
    nomes = ["original", "só\nparar", "subtrair", "FedEraser", "retreino"]
    acc9 = [0.963, 0.867, 0.0, 0.0, 0.0]; perda_ = [0.16, 0.40, 6.80, 3.18, 3.08]; custo = [3000, 950, 0, 570, 2850]
    cols = [GRY, AMB, RED, GRN, IND]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    axs[0].bar(range(5), acc9, color=cols, ec=INK, lw=0.5); axs[0].set_title("Acerto no dígito 9\n(deve cair a zero)", fontsize=9.5)
    axs[1].bar(range(5), perda_, color=cols, ec=INK, lw=0.5); axs[1].axhline(3.08, c=IND, ls="--", lw=1.2); axs[1].set_title("Perda nos dados do cliente removido\n(alvo: igual ao retreino)", fontsize=9.5)
    axs[2].bar(range(5), custo, color=cols, ec=INK, lw=0.5); axs[2].set_title("Custo (épocas-cliente)", fontsize=9.5)
    for a, v in zip(axs, (acc9, perda_, custo)):
        a.set_xticks(range(5)); a.set_xticklabels(nomes, fontsize=7)
        for i, val in enumerate(v): a.text(i, val + max(v) * 0.02, f"{val:g}", ha="center", fontsize=7.5)
    save(fig, "f12_desaprendizado")


def f12_semi():
    nomes = ["só\nrotulados", "pseudo\n0,5", "pseudo\n0,8", "pseudo\n0,95", "todos\nrotulados"]
    acc = [0.865, 0.864, 0.845, 0.850, 0.959]; prec = [np.nan, 0.881, 0.962, 0.990, np.nan]
    fig, ax = plt.subplots(figsize=(7.6, 3.5)); ax.bar(range(5), acc, color=[GRY, AMB, AMB, AMB, GRN], ec=INK, lw=0.5, label="acurácia do modelo")
    ax.plot(range(5), prec, "D-", c=VIO, lw=2, label="precisão dos pseudorrótulos aceitos")
    for i, a in enumerate(acc): ax.text(i, a - 0.05, f"{a:.2f}", ha="center", color="white", fontweight="bold", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(nomes); ax.set_ylim(0.6, 1.15); ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    ax.set_title("Pseudorrótulos: precisão alta, ganho nulo", fontsize=9.5)
    save(fig, "f12_semi")


def f12_multirotulo():
    fig = plt.figure(figsize=(9.8, 3.4)); ax = fig.add_axes([0.04, 0.12, 0.36, 0.72])
    rng = np.random.default_rng(2); M = np.zeros((8, 3)); M[0] = 1; M[1] = [0, 0, 1]
    for i in range(2, 8): M[i, rng.permutation(3)[:rng.integers(1, 3)]] = 1
    for i in range(8):
        for j in range(3):
            ax.add_patch(Rectangle((j, 7 - i), 0.92, 0.88, fc=GRN if M[i, j] else LIGHT, ec="white"))
            ax.text(j + 0.46, 7 - i + 0.44, "✓" if M[i, j] else "?", ha="center", va="center", fontsize=9, color="white" if M[i, j] else GRY, fontweight="bold")
    ax.set_xlim(0, 3); ax.set_ylim(0, 8); ax.set_xticks([0.46, 1.46, 2.46]); ax.set_xticklabels(["par", "≥ 5", "primo"], fontsize=8.5); ax.set_yticks([7 - i + 0.44 for i in range(8)]); ax.set_yticklabels([f"cliente {i + 1}" for i in range(8)], fontsize=7.5)
    ax.set_title("Quem anota o quê (? = ausente)", fontsize=9.5)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(length=0)
    ax2 = fig.add_axes([0.5, 0.14, 0.48, 0.72]); est = ["teto\n(completos)", "ausente\n= 0", "máscara +\nmédia global", "máscara +\npor rótulo"]
    F = np.array([[0.911, 0.897, 0.936], [0.460, 0.335, 0.704], [0.906, 0.877, 0.932], [0.907, 0.882, 0.934]])
    for j, (nome, c) in enumerate([("par", IND), ("≥ 5", CYA), ("primo", AMB)]): ax2.bar(np.arange(4) + (j - 1) * 0.26, F[:, j], 0.25, color=c, ec=INK, lw=0.4, label=nome)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(est, fontsize=7.5); ax2.set_ylim(0, 1.05); ax2.set_ylabel("F1"); ax2.legend(frameon=False, fontsize=8, ncol=3, loc="upper right"); ax2.set_title("F1 por rótulo", fontsize=9.5)
    save(fig, "f12_multirotulo")


def f12_deriva():
    Xtr, Xte, ytr, yte = carregar_digits(); K, R0 = 20, 25; H = {"continuar": [], "reiniciar": []}; res = {"um modelo": [], "IFCA (K = 2)": []}
    for s in range(3):
        perm = np.random.default_rng(s).permutation(10); partes = particao_dirichlet(ytr, K, 1.0, s)
        for est in H: H[est].append(fedavg_com_deriva(partes, Xtr, ytr, Xte, yte, perm, R0, None, est, seed=s)[1])
        metade = list(range(0, K, 2)); W, _, _ = fedavg_com_deriva(partes, Xtr, ytr, Xte, yte, perm, R0, metade, "continuar", seed=s)
        res["um modelo"].append((acuracia(W, Xte, yte), acuracia(W, Xte, perm[yte])))
        y_novo = ytr.copy()
        for c in metade: y_novo[partes[c]] = perm[ytr[partes[c]]]
        M, esc = ifca(partes, Xtr, y_novo, K=2, seed=s)
        mA = np.bincount([esc[c] for c in range(K) if c not in metade], minlength=2).argmax(); mB = np.bincount([esc[c] for c in metade], minlength=2).argmax()
        res["IFCA (K = 2)"].append((acuracia(M[mA], Xte, yte), acuracia(M[mB], Xte, perm[yte])))
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    axs[0].plot(np.mean(H["continuar"], 0), c=GRY, lw=2.2, label="continuar"); axs[0].plot(np.mean(H["reiniciar"], 0), c=GRN, lw=2.2, label="reiniciar ao detectar")
    axs[0].axvline(R0, c=RED, ls="--", lw=1.2); axs[0].text(R0 + 0.8, 0.25, "deriva", color=RED, fontsize=8.5); axs[0].set_xlabel("rodada"); axs[0].set_ylabel("acurácia no conceito atual"); axs[0].legend(frameon=False, fontsize=8.5, loc="lower right"); axs[0].set_title("Deriva global", fontsize=9.5); axs[0].set_ylim(0, 1)
    for k, (nome, c) in enumerate([("um modelo", GRY), ("IFCA (K = 2)", GRN)]):
        m = np.mean(res[nome], 0); axs[1].bar(np.arange(2) + (k - 0.5) * 0.36, m, 0.34, color=c, ec=INK, lw=0.5, label=nome)
        for j in range(2): axs[1].text(j + (k - 0.5) * 0.36, m[j] + 0.01, f"{m[j]:.2f}", ha="center", fontsize=8)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["conceito antigo", "conceito novo"]); axs[1].set_ylim(0, 1.3); axs[1].legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2); axs[1].set_title("Deriva parcial", fontsize=9.5)
    save(fig, "f12_deriva")


if __name__ == "__main__":
    for f in [f12_mapa, f12_assincrono, f12_assinc_res, f12_participacao, f12_estilos, f12_fedbn, f12_desaprendizado, f12_semi, f12_multirotulo, f12_deriva]: f()
