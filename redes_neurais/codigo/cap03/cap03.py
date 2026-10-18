"""Cap. 3 - Perceptron: experimentos e figuras (semente fixa; números citados no texto)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rnbook.estilo import plt, COR, salvar
from rnbook.desenho import caixa, seta, diagrama_rede
from rnbook.perceptron import PerceptronDidatico
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

P4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
PORTAS = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "NAND": [1, 1, 1, 0], "XOR": [0, 1, 1, 0]}

# ----------------------------------------------------------- 3.1 arquitetura e geometria
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
a = ax[0]; a.set_xlim(0, 10); a.set_ylim(0, 6); a.axis("off"); a.grid(False); a.set_aspect("equal")
for i, y in enumerate([4.6, 3.0, 1.4]):
    a.text(0.5, y, f"$x_{i+1}$", fontsize=12, ha="center", va="center"); seta(a, (0.95, y), (4.3, 3.0)); a.text(2.4, 3.0 + (y - 3.0) * 0.5 + 0.3, f"$w_{i+1}$", fontsize=10, color=COR["ind"])
a.add_patch(plt.Circle((5.1, 3), 0.8, fc=COR["ind"], ec="white")); a.text(5.1, 3, "$\\Sigma$", color="white", fontsize=15, ha="center", va="center")
a.text(5.1, 4.45, "$b$", fontsize=12, ha="center", color=COR["amb"]); seta(a, (5.1, 4.3), (5.1, 3.85), cor=COR["amb"])
seta(a, (5.95, 3), (6.5, 3)); a.add_patch(plt.Rectangle((6.5, 2.4), 1.2, 1.2, fc="white", ec=COR["grn"], lw=2)); a.plot([6.65, 7.1, 7.1, 7.55], [2.7, 2.7, 3.3, 3.3], color=COR["grn"], lw=2.5)
seta(a, (7.75, 3), (9.2, 3)); a.text(9.5, 3, "$\\hat y$", fontsize=13, va="center")
a.text(4.4, 1.6, "$z=\\mathbf{w}^T\\mathbf{x}+b$", ha="center", fontsize=10.5); a.text(7.4, 1.6, "degrau:\n$\\hat y=1$ se $z>0$", ha="center", fontsize=9)
a.set_title("Arquitetura do perceptron de uma camada")
b = ax[1]; w_ = np.array([1.0, 1.0]); b_ = -1.5; xs = np.linspace(-0.5, 2.0, 10)
b.plot(xs, -(w_[0] * xs + b_) / w_[1], color=COR["ind"], lw=2); b.fill_between(xs, -(w_[0] * xs + b_) / w_[1], 2.2, color=COR["grn"], alpha=0.12); b.fill_between(xs, -0.7, -(w_[0] * xs + b_) / w_[1], color=COR["red"], alpha=0.10)
p0 = np.array([0.75, 0.75]); b.annotate("", xy=p0 + 0.55 * w_ / np.linalg.norm(w_), xytext=p0, arrowprops=dict(arrowstyle="-|>", color=COR["red"], lw=2)); b.text(1.0, 1.3, "$\\mathbf{w}$ (normal)", color=COR["red"], fontsize=10, ha="right")
b.scatter([1.6], [1.4], color=COR["grn"], s=60, zorder=4); b.text(1.65, 1.5, "$z>0$", color=COR["grn"]); b.scatter([0.3], [0.2], color=COR["red"], s=60, zorder=4); b.text(0.35, 0.3, "$z<0$", color=COR["red"])
b.set_xlim(-0.5, 2.0); b.set_ylim(-0.5, 2.0); b.set_aspect("equal"); b.set_xlabel("$x_1$"); b.set_ylabel("$x_2$"); b.set_title("O perceptron separa o plano por um hiperplano $\\mathbf{w}^T\\mathbf{x}+b=0$")
salvar(fig, "f3_arquitetura")

# ----------------------------------------------------------- 3.2 treinamento: portas
res = {}
for nome, t in PORTAS.items():
    p = PerceptronDidatico(eta=1, max_epocas=50).fit(P4, t); res[nome] = p
    print(nome, "convergiu", p.convergiu, "| épocas", len(p.historico_erros), "| w", p.w, "b", p.b, "| atualizações", p.n_atualizacoes, "| erros/época", p.historico_erros[:9])
# traço detalhado da porta OR, primeira época
p = res["OR"]; print("trajetória OR (w,b) após cada atualização:", [(w.tolist(), b) for w, b in p.trajetoria])
p_and = res["AND"]
fig, ax = plt.subplots(1, 4, figsize=(12, 3.1))
for a, nome in zip(ax, PORTAS):
    p = res[nome]; t = PORTAS[nome]
    for pt, tt in zip(P4, t): a.scatter(*pt, s=170, color=COR["grn"] if tt else COR["red"], edgecolor="white", zorder=3)
    if p.convergiu:
        xl = np.linspace(-0.4, 1.4, 10); a.plot(xl, -(p.w[0] * xl + p.b) / p.w[1], color=COR["ind"], lw=2)
        a.set_title(f"{nome}: w={p.w.astype(int).tolist()}, b={int(p.b)}\n({len(p.historico_erros)-1} épocas com erros + 1 sem)", fontsize=9)
    else:
        a.set_title(f"{nome}: não converge\n(4 erros em toda época)", fontsize=9, color=COR["red"])
    a.set_xlim(-0.4, 1.4); a.set_ylim(-0.4, 1.4); a.set_aspect("equal"); a.set_xlabel("$x_1$")
salvar(fig, "f3_portas")

# evolução da fronteira (AND) após cada atualização
fig, ax = plt.subplots(figsize=(5.4, 4.2))
for pt, tt in zip(P4, PORTAS["AND"]): ax.scatter(*pt, s=170, color=COR["grn"] if tt else COR["red"], edgecolor="white", zorder=3)
cmap = plt.cm.viridis; tr = p_and.trajetoria[1:]
for k, (w, b) in enumerate(tr):
    if abs(w[1]) < 1e-9: continue
    xl = np.linspace(-0.5, 1.5, 10); ax.plot(xl, -(w[0] * xl + b) / w[1], color=cmap(k / len(tr)), lw=1.2 if k < len(tr) - 1 else 2.8, alpha=0.8)
ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5); ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("AND: fronteira após cada atualização (claro → escuro: tempo)")
salvar(fig, "f3_evolucao_and")

# ----------------------------------------------------------- 3.2 iris separável
X, y = load_iris(return_X_y=True); m = y < 2; Xi = X[m][:, [2, 3]]; yi = y[m]
pi = PerceptronDidatico(max_epocas=1000).fit(Xi, yi)
print("iris (comprimento e largura da pétala; setosa x versicolor): convergiu", pi.convergiu, "| atualizações", pi.n_atualizacoes, "| w", pi.w, "b", pi.b)
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.8))
for c, nm, col in [(0, "setosa", COR["ind"]), (1, "versicolor", COR["amb"])]: ax[0].scatter(Xi[yi == c, 0], Xi[yi == c, 1], s=22, color=col, label=nm)
for k, (w, b) in enumerate(pi.trajetoria):
    if abs(w[1]) < 1e-9: continue
    xl = np.linspace(0.5, 5.5, 10); ax[0].plot(xl, -(w[0] * xl + b) / w[1], color=plt.cm.viridis(k / max(1, len(pi.trajetoria) - 1)), lw=1.2 if k < len(pi.trajetoria) - 1 else 2.8)
ax[0].set_xlim(0.5, 5.5); ax[0].set_ylim(-0.2, 2.2); ax[0].set_xlabel("comprimento da pétala (cm)"); ax[0].set_ylabel("largura da pétala (cm)"); ax[0].legend(); ax[0].set_title("Iris: o perceptron converge em 4 atualizações")
# ----------------------------------------------------------- teorema de convergência: limite (R/gamma)^2
rng = np.random.default_rng(0); pts = []
for delta in np.geomspace(0.02, 0.5, 60):
    for rep in range(5):
        Xr = rng.uniform(-1, 1, size=(200, 2)); Xr = Xr[np.abs(Xr[:, 0]) >= delta]; yr = (Xr[:, 0] > 0).astype(int)
        Xa = np.column_stack([Xr, np.ones(len(Xr))]); gam = np.abs(Xr[:, 0]).min()           # w* = (1,0,0): margem = min |x1|
        R = np.linalg.norm(Xa, axis=1).max(); pr = PerceptronDidatico(max_epocas=100000, embaralhar=True, semente=rep).fit(Xr, yr)
        assert pr.convergiu
        pts.append((gam, R, (R / gam) ** 2, pr.n_atualizacoes))
pts = np.array(pts); ok = np.all(pts[:, 3] <= pts[:, 2] + 1e-9)
print("limite (R/gamma)^2 respeitado em %d/%d execuções; razão atualizações/limite: média %.3f, máximo %.3f" % (np.sum(pts[:, 3] <= pts[:, 2]), len(pts), np.mean(pts[:, 3] / pts[:, 2]), np.max(pts[:, 3] / pts[:, 2])))
ax[1].loglog(pts[:, 2], pts[:, 3], "o", ms=3.5, color=COR["ind"], alpha=0.6, label="execuções"); mm = [pts[:, 2].min(), pts[:, 2].max()]; ax[1].loglog(mm, mm, "--", color=COR["red"], label="limite $(R/\\gamma)^2$")
ax[1].set_xlabel("limite teórico $(R/\\gamma)^2$"); ax[1].set_ylabel("nº de atualizações observado"); ax[1].legend(); ax[1].set_title("Teorema de convergência: o limite nunca é violado")
salvar(fig, "f3_iris_teorema")

# ----------------------------------------------------------- não separável: versicolor x virginica (2 atributos) e XOR
m2 = y > 0; Xn = X[m2][:, [2, 3]]; yn = (y[m2] == 2).astype(int)
pn = PerceptronDidatico(max_epocas=200, embaralhar=False).fit(Xn, yn)
print("versicolor x virginica (pétalas): convergiu", pn.convergiu, "| erros por época (primeiras 12):", pn.historico_erros[:12], "| mín. erros em uma época:", min(pn.historico_erros))
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6))
ax[0].plot(res["XOR"].historico_erros, "o-", color=COR["red"], label="XOR (4 pontos)"); ax[0].plot(pn.historico_erros, color=COR["amb"], label="versicolor × virginica")
ax[0].plot(pi.historico_erros, "s-", color=COR["ind"], label="setosa × versicolor"); ax[0].set_xlim(0, 60); ax[0].set_xlabel("época"); ax[0].set_ylabel("erros de classificação na época"); ax[0].legend(); ax[0].set_title("Convergir ou não: depende da separabilidade")
for c, nm, col in [(0, "versicolor", COR["amb"]), (1, "virginica", COR["grn"])]: ax[1].scatter(Xn[yn == c, 0], Xn[yn == c, 1], s=22, color=col, label=nm)
ax[1].set_xlabel("comprimento da pétala (cm)"); ax[1].set_ylabel("largura da pétala (cm)"); ax[1].legend(); ax[1].set_title("Classes que se sobrepõem: nenhuma reta separa tudo")
salvar(fig, "f3_nao_separavel")

# ----------------------------------------------------------- XOR com duas camadas (pesos escolhidos à mão)
step = lambda z: (z > 0).astype(int)
h1 = step(P4 @ np.array([1, 1]) - 0.5); h2 = step(P4 @ np.array([-1, -1]) + 1.5); saida = step(h1 + h2 - 1.5)
print("XOR duas camadas: h1(OR)", h1, "h2(NAND)", h2, "saída", saida)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
t = PORTAS["XOR"]
for pt, tt in zip(P4, t): ax[0].scatter(*pt, s=170, color=COR["grn"] if tt else COR["red"], edgecolor="white", zorder=3)
ax[0].set_xlim(-0.4, 1.4); ax[0].set_ylim(-0.4, 1.4); ax[0].set_aspect("equal"); ax[0].set_title("Espaço original $(x_1,x_2)$:\nXOR não separável"); ax[0].set_xlabel("$x_1$"); ax[0].set_ylabel("$x_2$")
for pt, tt, a_, b_ in zip(P4, t, h1, h2): ax[1].scatter(a_ + 0.0, b_ + 0.0, s=170, color=COR["grn"] if tt else COR["red"], edgecolor="white", zorder=3)
xl = np.linspace(-0.4, 1.4, 10); ax[1].plot(xl, 1.5 - xl, color=COR["ind"], lw=2); ax[1].set_xlim(-0.4, 1.4); ax[1].set_ylim(-0.4, 1.4); ax[1].set_aspect("equal"); ax[1].set_xlabel("$h_1$ = OR"); ax[1].set_ylabel("$h_2$ = NAND"); ax[1].set_title("Espaço oculto $(h_1,h_2)$:\nagora uma reta separa")
diagrama_rede(ax[2], [2, 2, 1], rotulos=[["x₁", "x₂"], ["OR", "NAND"], ["AND"]], titulos=["entrada", "camada oculta", "saída"], passo_x=1.6, raio=0.27, fs=9)
ax[2].set_title("Rede 2–2–1 que resolve o XOR")
salvar(fig, "f3_xor_duas_camadas")

# ----------------------------------------------------------- 3.4 scikit-learn x implementação própria
for nome in ("AND", "OR"):
    pd_ = PerceptronDidatico(eta=1, max_epocas=50).fit(P4, PORTAS[nome])
    sk = Perceptron(penalty=None, eta0=1.0, shuffle=False, tol=None, max_iter=len(pd_.historico_erros), fit_intercept=True, random_state=0).fit(P4, PORTAS[nome])
    print("%s: próprio w=%s b=%s | sklearn w=%s b=%s | iguais: %s" % (nome, pd_.w, pd_.b, sk.coef_[0], sk.intercept_[0], np.allclose(pd_.w, sk.coef_[0]) and np.isclose(pd_.b, sk.intercept_[0])))
pd_ = PerceptronDidatico(max_epocas=1000).fit(Xi, yi); sk = Perceptron(penalty=None, eta0=1.0, shuffle=False, tol=None, max_iter=len(pd_.historico_erros)).fit(Xi, yi)
print("iris: próprio w=%s b=%s | sklearn w=%s b=%s" % (pd_.w, pd_.b, sk.coef_[0], sk.intercept_[0]))
# padrão do scikit-learn (shuffle=True, tol=1e-3) em iris completa, 3 classes, com padronização
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
for nome, mod in [("Perceptron() padrão", Perceptron(random_state=0)), ("Perceptron com padronização", make_pipeline(StandardScaler(), Perceptron(random_state=0)))]:
    mod.fit(Xtr, ytr); print("iris 3 classes, %s: acurácia treino %.4f | teste %.4f" % (nome, mod.score(Xtr, ytr), mod.score(Xte, yte)))
mod = make_pipeline(StandardScaler(), Perceptron(random_state=0)).fit(Xtr, ytr); print("coef_ forma:", mod[-1].coef_.shape, "(uma linha por classe: um-contra-todos)", "n_iter_", mod[-1].n_iter_)
# fronteiras sklearn (2 atributos, 3 classes)
X2 = X[:, [2, 3]]; sk3 = make_pipeline(StandardScaler(), Perceptron(random_state=0)).fit(X2, y); print("iris 3 classes, 2 atributos (treino): acurácia", sk3.score(X2, y))
gx, gy = np.meshgrid(np.linspace(0.5, 7.2, 300), np.linspace(-0.1, 2.8, 300)); Z = sk3.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(5.8, 4.2)); ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=[COR["ind"], COR["amb"], COR["grn"]], alpha=0.18)
for c, nm, col in [(0, "setosa", COR["ind"]), (1, "versicolor", COR["amb"]), (2, "virginica", COR["grn"])]: ax.scatter(X2[y == c, 0], X2[y == c, 1], s=20, color=col, label=nm)
ax.legend(); ax.set_xlabel("comprimento da pétala (cm)"); ax.set_ylabel("largura da pétala (cm)"); ax.set_title("scikit-learn Perceptron: 3 classes (um-contra-todos)")
salvar(fig, "f3_sklearn")
print("OK cap03")
