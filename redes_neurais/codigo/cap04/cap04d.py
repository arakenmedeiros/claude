"""Cap. 4 (parte D): figuras conceituais (ativações, grafo 2-2-1, fluxo) e moons: nosso MLP x MLPClassifier."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
from rnbook import MLP
from rnbook import otimizadores as O
from rnbook.ativacoes import sigmoide, softmax
from rnbook.estilo import plt, COR, salvar
from rnbook.desenho import caixa, seta
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- f4_ativacoes
z = np.linspace(-5, 5, 400)
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
for nm, y_, c in [("identidade", z, COR["gry"]), ("sigmoide", sigmoide(z), COR["ind"]), ("tanh", np.tanh(z), COR["grn"]), ("ReLU", np.maximum(0, z), COR["red"]), ("leaky ReLU ($0{,}1$)", np.where(z > 0, z, 0.1 * z), COR["amb"])]:
    ax[0].plot(z, y_, label=nm, color=c, lw=1.8)
ax[0].set_ylim(-2, 3); ax[0].legend(fontsize=7.5, loc="upper left"); ax[0].set_title("Funções de ativação $\\phi(z)$"); ax[0].set_xlabel("z")
zz = np.array([[2.0, 1.0, 0.1]]); sm = softmax(zz)[0]
ax[1].bar(["$z_1$", "$z_2$", "$z_3$"], zz[0], color=COR["gry"], alpha=0.5, label="entrada $z$"); ax[1].bar(["$z_1$", "$z_2$", "$z_3$"], sm, color=COR["ind"], width=0.4, label="saída softmax"); ax[1].legend(fontsize=8)
for i, v in enumerate(sm): ax[1].text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=9, color=COR["indD"])
ax[1].set_title("Softmax de $z=(2;\\,1;\\,0{,}1)$")
ds = np.linspace(-6, 6, 300); ax[2].plot(ds, sigmoide(ds) * (1 - sigmoide(ds)), color=COR["ind"], label="$\\sigma'(z)$"); ax[2].plot(ds, 1 - np.tanh(ds) ** 2, color=COR["grn"], label="$\\tanh'(z)$"); ax[2].plot(ds, (ds > 0) * 1.0, color=COR["red"], label="ReLU$'(z)$")
ax[2].fill_between(ds, 0, 0.05, where=(abs(ds) > 4), color=COR["amb"], alpha=0.3); ax[2].text(4.6, 0.3, "saturação:\nderivada $\\approx0$", fontsize=8, ha="center", color=COR["amb"]); ax[2].legend(fontsize=8, loc="upper left"); ax[2].set_title("Derivadas das ativações"); ax[2].set_xlabel("z")
salvar(fig, "f4_ativacoes")
print("softmax(2,1,0.1) =", np.round(sm, 4), "soma", sm.sum())

# ---------------- f4_grafo: grafo computacional 2-2-1 com valores
W1 = np.array([[0.1, 0.3], [-0.2, 0.4]]); b1 = np.array([0.1, -0.1]); W2 = np.array([[0.5, -0.6]]); b2 = np.array([0.2]); x = np.array([1.0, 0.5]); t = 1.0
z1 = W1 @ x + b1; a1 = sigmoide(z1); z2 = W2 @ a1 + b2; y = sigmoide(z2)[0]; r = y - t; d2 = r * y * (1 - y); d1 = (W2.T @ np.array([d2])) * a1 * (1 - a1)
fig, ax = plt.subplots(figsize=(10.4, 4.6)); ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.6); ax.axis("off"); ax.grid(False)
pos = {"x1": (0.7, 3.2), "x2": (0.7, 1.4), "h1": (3.6, 3.2), "h2": (3.6, 1.4), "o": (6.5, 2.3)}
for k, (px, py) in pos.items():
    col = COR["cya"] if k.startswith("x") else COR["ind"] if k.startswith("h") else COR["amb"]
    ax.add_patch(plt.Circle((px, py), 0.45, fc=col, ec="white", lw=1.5, zorder=3)); ax.text(px, py, {"x1": "$x_1$", "x2": "$x_2$", "h1": "$a_1^{(1)}$", "h2": "$a_2^{(1)}$", "o": "$y$"}[k], color="white", ha="center", va="center", fontsize=10, zorder=4)
for i, xi in enumerate(("x1", "x2")):
    for j, hj in enumerate(("h1", "h2")):
        seta(ax, (pos[xi][0] + 0.47, pos[xi][1] - (0.0 if i == j else (0.2 if i < j else -0.2))), (pos[hj][0] - 0.47, pos[hj][1] + (0.0 if i == j else (-0.2 if i < j else 0.2))), cor=COR["gry"], lw=1.1)
for hj in ("h1", "h2"): seta(ax, (pos[hj][0] + 0.47, pos[hj][1] + (-0.2 if hj == "h1" else 0.2)), (pos["o"][0] - 0.47, pos["o"][1] + (0.15 if hj == "h1" else -0.15)), cor=COR["gry"], lw=1.1)
ax.text(0.7, 3.95, "$x=1$", ha="center", fontsize=9); ax.text(0.7, 0.7, "$x=0{,}5$", ha="center", fontsize=9)
ax.text(2.15, 4.3, "$W^{(1)}$: linhas $(0{,}1;\\,0{,}3)$ e $(-0{,}2;\\,0{,}4)$", ha="center", fontsize=8.5, color=COR["ind"])
ax.text(3.6, 3.85, f"$z={z1[0]:.2f}$\n$a={a1[0]:.4f}$", ha="center", fontsize=8.5); ax.text(3.6, 0.65, f"$z={z1[1]:.2f}$\n$a={a1[1]:.4f}$", ha="center", fontsize=8.5)
ax.text(6.5, 3.2, f"$z={z2[0]:.4f}$\n$y={y:.4f}$", ha="center", fontsize=8.5); ax.text(6.5, 1.1, f"$t=1$, $r={r:.4f}$\n$E={0.5*r*r:.5f}$", ha="center", fontsize=8.5, color=COR["red"])
ax.text(8.9, 3.6, "retropropagação", ha="center", fontsize=10, color=COR["red"], fontweight="bold")
ax.text(8.9, 2.85, f"$\\delta^{{(2)}}=r\\,y(1-y)=${d2:.5f}", ha="center", fontsize=9, color=COR["red"])
ax.text(8.9, 2.25, f"$\\delta^{{(1)}}_1=${d1[0]:.5f}", ha="center", fontsize=9, color=COR["red"]); ax.text(8.9, 1.75, f"$\\delta^{{(1)}}_2=${d1[1]:.5f}", ha="center", fontsize=9, color=COR["red"])
ax.text(8.9, 1.05, f"$\\partial E/\\partial W^{{(2)}}=(${d2*a1[0]:.5f}$,\\ ${d2*a1[1]:.5f}$)$", ha="center", fontsize=8.5, color=COR["red"])
ax.set_title("Rede 2–2–1 do exemplo numérico (sigmoide em todos os neurônios)")
salvar(fig, "f4_grafo")

# ---------------- f4_fluxo: fluxograma do treinamento
fig, ax = plt.subplots(figsize=(10, 3.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off"); ax.grid(False)
passos = [("1. Passo direto\n$a^{(l)}=\\phi(W a+b)$", COR["cya"]), ("2. Perda\n$E(\\theta)$", COR["amb"]), ("3. Retropropagação\n$\\delta^{(l)}\\to\\nabla E$", COR["red"]), ("4. Otimizador\n$\\theta\\leftarrow\\theta+\\mathbf{d}$", COR["grn"]), ("5. Parada?\nvalidação, $\\|\\nabla E\\|$", COR["vio"])]
for i, (t_, c) in enumerate(passos):
    caixa(ax, 0.15 + i * 1.97, 1.0, 1.75, 1.3, t_, cor=c, fc="white", fs=8.8, bold=True)
    if i < 4: seta(ax, (0.15 + i * 1.97 + 1.77, 1.65), (0.15 + (i + 1) * 1.97 - 0.02, 1.65), cor=COR["gry"])
seta(ax, (9.0, 1.0), (0.9, 1.0), cor=COR["gry"], rad=0.0, ls="--"); ax.text(5, 0.6, "se não parou: repetir (nova época ou novo lote)", ha="center", fontsize=9, color=COR["gry"])
ax.text(5, 2.9, "O backpropagation é apenas a etapa 3; as etapas 4 e 5 definem o algoritmo de treinamento", ha="center", fontsize=10, style="italic")
salvar(fig, "f4_fluxo")

# ---------------- moons: nosso MLP (SSE, saída sigmoide, RPROP) x MLPClassifier (lbfgs)
X, y = make_moons(400, noise=0.25, random_state=0); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
esc = StandardScaler().fit(Xtr); Xs, Xt = esc.transform(Xtr), esc.transform(Xte)
rede = MLP([2, 10, 1], ["tanh", "sigmoide"], semente=1)
res = O.rprop(rede.theta, lambda th: rede.perda(th, Xs, ytr[:, None].astype(float)), lambda th: rede.grad(th, Xs, ytr[:, None].astype(float)), variante="irprop-", max_iter=500)
acc = lambda m, Xa, ya: np.mean((m.prever(Xa, res.theta).ravel() > 0.5) == ya)
print("moons nosso MLP 2-10-1 (SSE, iRPROP-, 500 épocas): acurácia treino %.4f | teste %.4f | P=%d" % (acc(rede, Xs, ytr), acc(rede, Xt, yte), rede.n_params))
sk = MLPClassifier(hidden_layer_sizes=(10,), activation="tanh", solver="lbfgs", max_iter=2000, random_state=1).fit(Xs, ytr)
print("moons MLPClassifier (tanh, lbfgs, log_loss): treino %.4f | teste %.4f" % (sk.score(Xs, ytr), sk.score(Xt, yte)))
sk2 = MLPClassifier(hidden_layer_sizes=(10,), activation="relu", solver="adam", max_iter=2000, random_state=1).fit(Xs, ytr)
print("moons MLPClassifier (relu, adam padrão): treino %.4f | teste %.4f | n_iter_ %d" % (sk2.score(Xs, ytr), sk2.score(Xt, yte), sk2.n_iter_))
gx, gy = np.meshgrid(np.linspace(-2.6, 2.8, 250), np.linspace(-2.2, 2.4, 250)); G = np.c_[gx.ravel(), gy.ravel()]
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.9))
for a, (nm, f_) in zip(ax, [("nosso MLP 2–10–1 (SSE, iRPROP−)", lambda G_: rede.prever(G_, res.theta).ravel()), ("MLPClassifier (tanh, lbfgs, entropia cruzada)", lambda G_: sk.predict_proba(G_)[:, 1])]):
    Z = f_(G).reshape(gx.shape); a.contourf(gx, gy, Z, levels=np.linspace(0, 1, 11), cmap="RdYlGn", alpha=0.6); a.contour(gx, gy, Z, levels=[0.5], colors="k", linewidths=1.5)
    a.scatter(Xs[:, 0], Xs[:, 1], c=np.where(ytr == 1, COR["grn"], COR["red"]), s=14, edgecolor="white", linewidths=0.3); a.set_title(nm, fontsize=9); a.set_xlabel("$x_1$ (padronizado)")
salvar(fig, "f4_moons")
print("OK cap04d")
