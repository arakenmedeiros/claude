"""Cap. 4 (parte B): XOR com backpropagation, aproximação universal, padrão a padrão x minilotes x lote."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rnbook import MLP
from rnbook import otimizadores as O
from rnbook.estilo import plt, COR, salvar

# ---------------------------------------------------------------- XOR
P4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); T4 = np.array([[0], [1], [1], [0]], float)
def treina_xor(H, semente, eta=1.0, epocas=6000):
    r = MLP([2, H, 1], ["sigmoide", "sigmoide"], semente=semente, init="pequena")
    res = O.gradiente_descendente(r.theta, lambda th: r.perda(th, P4, T4), lambda th: r.grad(th, P4, T4), taxa=eta, max_iter=epocas)
    return r, res
tab = {}
for H in (2, 3, 4):
    ok = 0; ep = []
    for s in range(30):
        r, res = treina_xor(H, s); acertos = ((r.prever(P4, res.theta) > 0.5).astype(float) == T4).all()
        ok += int(acertos); ep.append(res.historico["perda"][-1])
    tab[H] = (ok, np.median(ep)); print("XOR, H=%d: %d/30 sementes resolvem em 6000 épocas (eta=1,0); perda final mediana %.4f" % (H, ok, np.median(ep)))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
bom = mau = None
for s in range(30):
    r, res = treina_xor(2, s); acert = ((r.prever(P4, res.theta) > 0.5).astype(float) == T4).all()
    cor = COR["grn"] if acert else COR["red"]
    ax[0].semilogy(res.historico["perda"], color=cor, alpha=0.5, lw=1)
    if acert and bom is None: bom = (r, res, s)
    if not acert and mau is None: mau = (r, res, s)
ax[0].set_xlabel("época"); ax[0].set_ylabel("perda $E$ (log)"); ax[0].set_title("XOR com rede 2–2–1: 30 sementes\n(verde: aprende; vermelho: preso)")
gx, gy = np.meshgrid(np.linspace(-0.4, 1.4, 120), np.linspace(-0.4, 1.4, 120)); G = np.c_[gx.ravel(), gy.ravel()]
for a, (nm, tup) in zip(ax[1:], [("aprendeu (semente %d)", bom), ("preso (semente %d)", mau)]):
    if tup is None: a.axis("off"); continue
    r, res, s = tup; Z = r.prever(G, res.theta).reshape(gx.shape)
    a.contourf(gx, gy, Z, levels=np.linspace(0, 1, 11), cmap="RdYlGn", alpha=0.6); a.contour(gx, gy, Z, levels=[0.5], colors="k")
    for p_, t_ in zip(P4, T4[:, 0]): a.scatter(*p_, s=120, color=COR["grn"] if t_ else COR["red"], edgecolor="white", zorder=3)
    a.set_title(nm % s + "\nE=%.4f" % res.historico["perda"][-1]); a.set_aspect("equal"); a.set_xlabel("$x_1$")
salvar(fig, "f4_xor")
if mau: print("exemplo preso: semente", mau[2], "perda final %.4f" % mau[1].historico["perda"][-1], "saídas", np.round(mau[0].prever(P4, mau[1].theta).ravel(), 3))
if bom: print("exemplo bom: semente", bom[2], "épocas até E<0.01:", int(np.argmax(np.array(bom[1].historico["perda"]) < 0.01)), "saídas", np.round(bom[0].prever(P4, bom[1].theta).ravel(), 3))

# ---------------------------------------------------------------- aproximação universal (ilustração; não é demonstração)
xx = np.linspace(0, 1, 200)[:, None]; alvo = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x); tt = alvo(xx)
xt = np.linspace(0, 1, 1000)[:, None]
resultados = {}
fig, axs = plt.subplots(2, 3, figsize=(11, 5.2))
for ax_, H in zip(axs.ravel(), (1, 2, 4, 8, 16, 32)):
    melhor = None; rm = []
    for s in range(6):
        r = MLP([1, H, 1], ["tanh", "identidade"], semente=s, init="xavier")
        res = O.levenberg_marquardt(r.theta, lambda th: r.residuos(th, xx, tt), lambda th: r.jacobiana(th, xx), max_iter=200)
        e = np.sqrt(np.mean((r.prever(xx, res.theta) - tt) ** 2)); rm.append(e)
        if melhor is None or e < melhor[0]: melhor = (e, r, res.theta, s)
    e, r, th, s = melhor; erro_max = np.max(np.abs(r.prever(xt, th) - alvo(xt)))
    resultados[H] = (e, erro_max, min(rm), max(rm), r.n_params)
    ax_.plot(xt, alvo(xt), color=COR["gry"], lw=2, label="alvo"); ax_.plot(xt, r.prever(xt, th), color=COR["ind"], lw=1.6, label="rede")
    ax_.set_title(f"H={H} ({r.n_params} parâmetros)\nRMSE={e:.4f}  erro máx.={erro_max:.3f}", fontsize=9); ax_.set_xlabel("x")
axs[0, 0].legend(fontsize=8)
salvar(fig, "f4_aproximacao")
for H, v in resultados.items(): print("H=%2d (P=%3d): melhor RMSE %.5f | erro máx (1000 pontos) %.4f | RMSE entre 6 sementes: [%.4f, %.4f]" % (H, v[4], v[0], v[1], v[2], v[3]))
# treinabilidade: GD simples com H=16
r = MLP([1, 16, 1], ["tanh", "identidade"], semente=0)
res = O.gradiente_descendente(r.theta, lambda th: r.perda(th, xx, tt) / 200, lambda th: r.grad(th, xx, tt) / 200, taxa=0.1, max_iter=20000)
print("H=16 com gradiente descendente simples (eta=0.1, 20000 passos): RMSE = %.4f (LM: %.5f)" % (np.sqrt(np.mean((r.prever(xx, res.theta) - tt) ** 2)), resultados[16][0]))

# ---------------------------------------------------------------- padrão a padrão x minilote x lote
rng = np.random.default_rng(11); N = 200; X = rng.uniform(0, 1, (N, 1)); T = np.sin(2 * np.pi * X) + rng.normal(0, 0.15, (N, 1))
grade = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
modos = [("padrão a padrão (1)", 1), ("minilote (16)", 16), ("lote (200)", N)]
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6)); linhas = {}; cfg = []
for nome, bs in modos:
    melhor = None
    for eta in grade:
        r = MLP([1, 10, 1], ["tanh", "identidade"], semente=3)
        res = O.gd_estocastico(r.theta, lambda th, idx: r.grad(th, X[idx], T[idx]), N, lambda th: r.perda(th, X, T) / N, taxa=eta, tamanho_lote=bs, epocas=300, semente=0)
        h = np.array(res.historico["perda"]); fin = h[-1] if np.isfinite(h[-1]) else np.inf
        if melhor is None or fin < melhor[0]: melhor = (fin, eta, h, res.tempo)
    fin, eta, h, dt = melhor; cfg.append((nome, bs, eta)); linhas[nome] = (h, dt)
    ax[0].semilogy(h, label=f"{nome}, $\\eta$={eta}")
    print("%-22s melhor eta=%.3f (grade %s): perda(época 10)=%.4f (100)=%.4f (300)=%.4f | tempo 300 épocas %.2f s | atualizações por época: %d" % (nome, eta, grade, h[9], h[99], h[-1], dt, int(np.ceil(N / bs))))
print("piso teórico da perda média E/N com ruído sigma=0.15: %.5f" % (0.5 * 0.15 ** 2))
ax[0].axhline(0.5 * 0.15 ** 2, color=COR["gry"], ls=":", lw=1); ax[0].text(120, 0.0125, "piso do ruído", color=COR["gry"], fontsize=8)
ax[0].set_xlabel("época"); ax[0].set_ylabel("perda média $E/N$"); ax[0].legend(fontsize=8); ax[0].set_title("Convergência por época (melhor $\\eta$ de cada modo)")
ax[1].bar([n for n, _, _ in cfg], [linhas[n][1] for n, _, _ in cfg], color=[COR["red"], COR["amb"], COR["grn"]]); ax[1].set_ylabel("tempo para 300 épocas (s)"); ax[1].set_title("Custo por época (laço em Python)"); ax[1].tick_params(axis="x", labelsize=8)
salvar(fig, "f4_lotes")
# ruído do gradiente: dispersão do gradiente do minilote em torno do gradiente de lote
r = MLP([1, 10, 1], ["tanh", "identidade"], semente=3); gfull = r.grad(r.theta, X, T) / N; rng2 = np.random.default_rng(0)
for bs in (1, 16, 200):
    erros = []
    for _ in range(300):
        idx = rng2.choice(N, bs, replace=False); erros.append(np.linalg.norm(r.grad(r.theta, X[idx], T[idx]) / bs - gfull))
    print("lote %3d: ||g_lote - g_completo|| média %.4f (||g_completo|| = %.4f)" % (bs, np.mean(erros), np.linalg.norm(gfull)))
print("OK cap04b")
