"""Figuras do Capítulo 8 (grafos: Label Propagation e Label Spreading)."""
import re, gzip, pickle
from fc import *
import matplotlib.colors
from figs_d import out, pct
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph

_g = {}
exec(open("/home/user/claude/semissupervisionado/cap/code/ch08_do_zero.py").read().split("X, y = make_moons")[0], _g)
grafo_knn, propagacao, espalhamento = _g["grafo_knn"], _g["propagacao"], _g["espalhamento"]

def _luas():
    X, y = make_moons(300, noise=0.07, random_state=2)
    def mp(p, c):
        i = np.where(y == c)[0]; return i[np.argmin(((X[i] - p) ** 2).sum(1))]
    idx = [mp((0, 1.0), 0), mp((-0.8, 0.5), 0), mp((1, -0.5), 1), mp((1.8, -0.1), 1)]
    yp = np.full(len(y), -1); yp[idx] = y[idx]
    return X, y, yp, idx

def f8_telefone():
    fig, ax = canvas(9.0, 4.4, (0, 100), (0, 52))
    ax.text(50, 49.5, "Rótulos como boatos: passam de vizinho em vizinho", ha="center", fontsize=10.8, fontweight="bold")
    rng = np.random.default_rng(8)
    nos = np.array([[8, 26], [20, 36], [20, 16], [34, 28], [46, 38], [46, 18], [58, 26], [70, 36], [70, 16], [84, 26], [92, 36], [92, 16]])
    arestas = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6), (6, 7), (6, 8), (7, 9), (8, 9), (9, 10), (9, 11)]
    for a, b in arestas: ax.plot(*zip(nos[a], nos[b]), c=MID, lw=2, zorder=1)
    cores = {0: IND, 11: AMB}
    for i, (x, y) in enumerate(nos):
        c = cores.get(i)
        if c: ax.scatter(x, y, s=520, c=c, edgecolors="black", lw=1.6, zorder=3, marker="*")
        else:
            t = (i - 0) / 11; col = np.array(matplotlib.colors.to_rgb(IND)) * (1 - t) + np.array(matplotlib.colors.to_rgb(AMB)) * t
            ax.scatter(x, y, s=260, c=[col], edgecolors="white", lw=1.5, zorder=2)
    ax.text(8, 20, "rótulo A\n(conhecido)", ha="center", fontsize=8, color=IND, fontweight="bold"); ax.text(92, 10, "rótulo B\n(conhecido)", ha="center", fontsize=8, color=AMB, fontweight="bold")
    ax.text(50, 6, "Cada ponto \"escuta\" os vizinhos e adota a mistura de rótulos que chega até ele. Quem está perto de A fica azul; quem está perto de B, laranja.", ha="center", fontsize=8, style="italic")
    save(fig, "f8_telefone")

def f8_grafo():
    X, y, yp, idx = _luas()
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.5))
    for ax, k in zip(axs, [3, 7, 20]):
        G = kneighbors_graph(X, k, mode="connectivity").toarray(); G = np.maximum(G, G.T)
        ii, jj = np.where(np.triu(G) > 0)
        mix = 0
        for a, b in zip(ii, jj):
            ruim = y[a] != y[b]; mix += ruim
            ax.plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]], c=RED if ruim else "#9CA3AF", lw=1.2 if ruim else 0.5, alpha=0.9 if ruim else 0.5, zorder=1 if not ruim else 2)
        ax.scatter(X[:, 0], X[:, 1], c=[CL[0] if t == 0 else CL[1] for t in y], s=8, zorder=3, edgecolors="none")
        ax.set_title(f"k = {k}\n{len(ii)} arestas; {mix} ligam classes diferentes", fontsize=8.2); clean(ax, True)
    fig.text(0.5, -0.03, "arestas vermelhas = \"pontes\" entre classes diferentes (nocivas); cinza = arestas entre pontos da mesma classe", ha="center", fontsize=7.8, style="italic", color=GRY)
    save(fig, "f8_grafo")

def f8_propaga():
    X, y, yp, idx = _luas()
    W = grafo_knn(X, 7); F, hist = propagacao(W, yp, 2, iters=60)
    fig, axs = plt.subplots(1, 5, figsize=(11.6, 2.9))
    Fs = []
    P = W / W.sum(1, keepdims=True); rot = yp != -1; Y = np.zeros((300, 2)); Y[rot, yp[rot]] = 1; Fc = Y.copy(); Fc[~rot] = 0.5
    snaps = {0: Fc.copy()}
    for t in range(1, 21):
        Fc = P @ Fc; Fc[rot] = Y[rot]
        if t in (1, 3, 5, 10, 20): snaps[t] = Fc.copy()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("c", [IND, "#E5E7EB", AMB])
    for ax, t in zip(axs, [1, 3, 5, 10, 20]):
        F_ = snaps[t]; p1 = F_[:, 1] / (F_.sum(1) + 1e-12)
        ax.scatter(X[:, 0], X[:, 1], c=p1, cmap=cmap, vmin=0, vmax=1, s=10, edgecolors="none")
        ax.scatter(X[idx, 0], X[idx, 1], c=[CL[0] if t_ == 0 else CL[1] for t_ in y[idx]], marker="*", s=130, edgecolors="black", zorder=4)
        ax.set_title(f"passo {t}\nacurácia = {100 * np.mean(F_.argmax(1) == y):.0f}%", fontsize=9); clean(ax, True)
    fig.text(0.5, -0.04, "cor = probabilidade da classe laranja (azul = 0, cinza = 0,5, laranja = 1); estrelas = 4 rotulados", ha="center", fontsize=7.8, style="italic", color=GRY)
    save(fig, "f8_propaga")

def f8_regras():
    fig, ax = canvas(9.2, 4.2, (0, 100), (0, 50))
    ax.text(50, 47.5, "Duas regras de atualização", ha="center", fontsize=10.8, fontweight="bold")
    rbox(ax, 2, 12, 46, 30, fc="#EEF2FF", ec=IND, lw=2, r=2.2); rbox(ax, 2, 36, 46, 6, "Label Propagation", fc=IND, tc="white", bold=True, fs=9.2, r=2.2)
    ax.text(25, 28, "F ← P · F", ha="center", fontsize=14, fontweight="bold", family="monospace", color=IND)
    ax.text(25, 20.5, "depois, os rotulados\nvoltam ao valor original\n(\"clamping\" rígido)", ha="center", va="center", fontsize=8.4, linespacing=1.4)
    rbox(ax, 52, 12, 46, 30, fc="#FFFBEB", ec=AMB, lw=2, r=2.2); rbox(ax, 52, 36, 46, 6, "Label Spreading", fc=AMB, tc="white", bold=True, fs=9.2, r=2.2)
    ax.text(75, 28, "F ← α · S · F + (1 − α) · Y", ha="center", fontsize=11.5, fontweight="bold", family="monospace", color="#B45309")
    ax.text(75, 20.5, "os rotulados podem mudar;\nα controla o equilíbrio entre\nvizinhos e rótulo original", ha="center", va="center", fontsize=8.4, linespacing=1.4)
    ax.text(50, 6, "P = D⁻¹W (transição);  S = D⁻½ W D⁻½ (normalizada);  Y = rótulos conhecidos;  F = \"quanto de cada classe\" chega a cada ponto", ha="center", fontsize=7.8, style="italic", color=GRY)
    save(fig, "f8_regras")

def f8_pureza():
    L = out("ch08_grafo").splitlines()[1:]
    k, comp, pur, acc = [], [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; k.append(int(p[0])); comp.append(int(p[1])); pur.append(pct(p[2])); acc.append(pct(p[3]))
    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.plot(k, acc, "-o", c=RED, lw=2.6, label="acurácia do Label Spreading (50 rótulos)"); ax.plot(k, pur, "-s", c=IND, lw=2.2, label="arestas que ligam a mesma classe (%)")
    for kk, a, c in zip(k, acc, comp):
        if c > 1: ax.annotate(f"{c} componentes\nconexos", (kk, a), xytext=(kk + 2.5, a - 3), fontsize=8, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("número de vizinhos k do grafo"); ax.set_ylabel("%"); ax.set_ylim(45, 100); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("O grafo importa: poucos vizinhos desconectam, muitos criam pontes")
    save(fig, "f8_pureza")

def f8_ruido():
    L = out("ch08_ruido").splitlines()[3:]
    r, lp, l2, l8 = [], [], [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; r.append(pct(p[0])); lp.append(pct(p[1])); l2.append(pct(p[2])); l8.append(pct(p[3]))
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.plot(r, lp, "-o", c=IND, lw=2.4, label="Label Propagation (rótulos fixos)"); ax.plot(r, l2, "-s", c=AMB, lw=2.4, label="Label Spreading, α = 0,2"); ax.plot(r, l8, "-^", c=RED, lw=2.4, label="Label Spreading, α = 0,8")
    ax.set_xlabel("proporção de rótulos errados (%)"); ax.set_ylabel("acurácia no teste (%)"); ax.set_title("Rótulos errados: contra a expectativa, o Propagation resistiu melhor"); ax.legend(frameon=False, fontsize=8.5); ax.grid(alpha=0.25); ax.set_xticks(r)
    save(fig, "f8_ruido")

def f8_alpha():
    L = out("ch08_sensibilidade").splitlines()
    cab = [float(x) for x in L[1].split("|")[1:]]; ks, M = [], []
    for l in L[2:]:
        p = l.split("|"); ks.append(int(p[0])); M.append([float(x) for x in p[1:]])
    M = np.array(M)
    fig, ax = plt.subplots(figsize=(6.6, 4.0)); im = ax.imshow(M, cmap="RdYlGn", vmin=86, vmax=95, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(cab))); ax.set_xticklabels([str(c).replace(".", ",") for c in cab]); ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
    ax.set_xlabel("alpha"); ax.set_ylabel("n_neighbors (k)"); ax.set_title("Acurácia (%) do Label Spreading no Digits, 50 rótulos"); ax.spines[:].set_visible(False)
    save(fig, "f8_alpha")

def f8_rbf():
    L = out("ch08_rbf").splitlines()[2:]
    nomes, acc = [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; nomes.append(("knn k=7" if p[0] == "knn" else "rbf " + p[1].replace("gamma = ", "γ=")).replace(".", ",")); acc.append(pct(p[2]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]; ax.bar(range(len(nomes)), acc, color=[GRN] + [RED if a < 20 else IND for a in acc[1:]], width=0.65)
    for i, a in enumerate(acc): ax.text(i, a + 1.5, f"{a:.1f}%".replace(".", ","), ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(nomes))); ax.set_xticklabels(nomes, fontsize=7.8, rotation=20); ax.set_ylabel("acurácia no teste (%)"); ax.set_ylim(0, 108); ax.set_title("Digits, 50 rótulos: grafo k-vizinhos x núcleo RBF")
    ax = axs[1]; d = np.linspace(0, 3000, 300)
    for g, c in [(20, RED), (0.1, IND), (0.005, GRN)]: ax.plot(d, np.exp(-g * d), c=c, lw=2.4, label=f"γ = {g}".replace(".", ","))
    ax.axvline(2392, c=GRY, ls="--", lw=1.2); ax.text(2350, 0.62, "distância² média\nentre dois dígitos\n(≈ 2.392)", ha="right", fontsize=7.8, color=INK)
    ax.set_xlabel("distância² entre dois pontos"); ax.set_ylabel("peso da aresta  exp(−γ·d²)"); ax.set_title("Com γ = 20, todos os pesos viram zero"); ax.legend(frameon=False, fontsize=8.3); ax.grid(alpha=0.25)
    save(fig, "f8_rbf")

def f8_mnist():
    arq = "/home/user/claude/semissupervisionado/dados/mnist.pkl.gz"
    (Xa, ya), _, _ = pickle.load(gzip.open(arq), encoding="latin1")
    L = out("ch08_mnist").splitlines()[3:]
    nomes, M = [], []
    for l in L:
        p = [x.strip() for x in l.split("|")]; nomes.append(p[0]); M.append([pct(p[1]), pct(p[2]), pct(p[3]), pct(p[4])])
    M = np.array(M)
    fig = plt.figure(figsize=(10.6, 4.0)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    ax0 = fig.add_subplot(gs[0]); ax0.axis("off"); ax0.set_title("Os 10 de 10.000 que têm rótulo\n(um exemplo por dígito; 100 no total)", fontsize=9)
    rng = np.random.RandomState(0)
    for d in range(10):
        i = rng.choice(np.where(ya == d)[0]); sub = fig.add_axes([0.01 + (d % 5) * 0.07, 0.50 - (d // 5) * 0.34, 0.065, 0.30]); sub.imshow(Xa[i].reshape(28, 28), cmap="gray_r"); sub.set_xticks([]); sub.set_yticks([]); sub.set_title(str(d), fontsize=8)
    ax = fig.add_subplot(gs[1]); x = np.arange(len(nomes)); w = 0.2
    rots = ["regressão logística (só rotulados)", "1-vizinho (só rotulados)", "Label Spreading (sem rótulo)", "Label Spreading (teste)"]
    for j, (c, r_) in enumerate(zip([GRY, "#CBD5E1", AMB, RED], rots)): ax.bar(x + (j - 1.5) * w, M[:, j], w, color=c, label=r_)
    for i in range(len(nomes)): ax.text(x[i] + 1.5 * w, M[i, 3] + 1, f"{M[i, 3]:.1f}".replace(".", ","), ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=8.5); ax.set_ylim(50, 98); ax.set_ylabel("acurácia (%)"); ax.legend(frameon=False, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2); ax.set_title("MNIST: 10.000 imagens, 100 rótulos", fontsize=9.5)
    save(fig, "f8_mnist")

def f8_gnn():
    fig, ax = canvas(9.2, 4.4, (0, 100), (0, 52))
    ax.text(50, 49.5, "Do Label Propagation às redes neurais de grafos (GNN)", ha="center", fontsize=10.6, fontweight="bold")
    rbox(ax, 2, 10, 44, 32, fc="#EEF2FF", ec=IND, lw=2, r=2.2); rbox(ax, 2, 36, 44, 6, "Label Propagation", fc=IND, tc="white", bold=True, fs=9, r=2.2)
    ax.text(24, 27, "cada nó faz a média dos\nrótulos dos vizinhos;\nnão há parâmetros a aprender\n(só o grafo e os rótulos)", ha="center", va="center", fontsize=8.4, linespacing=1.45)
    rbox(ax, 54, 10, 44, 32, fc="#FFFBEB", ec=AMB, lw=2, r=2.2); rbox(ax, 54, 36, 44, 6, "Rede neural de grafos (GCN)", fc=AMB, tc="white", bold=True, fs=9, r=2.2)
    ax.text(76, 27, "cada nó combina os atributos\ndos vizinhos com pesos\naprendidos; a rede treina só\ncom os nós rotulados\n(Kipf e Welling, 2017)", ha="center", va="center", fontsize=8.4, linespacing=1.45)
    arrow(ax, (46.3, 26), (53.7, 26), INK, lw=2, ms=14)
    ax.text(50, 5, "Mesma ideia: informação escorre pelas arestas. A GNN acrescenta atributos e pesos aprendidos.", ha="center", fontsize=8.2, style="italic", color=GRY)
    save(fig, "f8_gnn")

if __name__ == "__main__":
    for f in [f8_telefone, f8_grafo, f8_propaga, f8_regras, f8_pureza, f8_ruido, f8_alpha, f8_rbf, f8_mnist, f8_gnn]: f()
