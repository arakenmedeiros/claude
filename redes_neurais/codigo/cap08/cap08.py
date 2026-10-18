"""Cap. 8 - Delta-Delta, Delta-Bar-Delta e EDBD: experimentos e figuras (semente fixa; números do texto)."""
import sys, os, time, warnings, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
from rnbook import MLP
from rnbook import otimizadores as O
from rnbook.estilo import plt, COR, salvar

# ---------------------------------------------------------------- 8.1.1 derivação da delta-delta (verificação numérica)
# E(w) quadrática 1D: w(t+1) = w(t) - alfa(t) g(t). dE(t+1)/dalfa(t) = -g(t+1) g(t).
lam = 4.0; Eq = lambda w: 0.5 * lam * w * w; gq = lambda w: lam * w
w, alfa, h = 1.0, 0.1, 1e-6
w1 = lambda a: w - a * gq(w)
num = (Eq(w1(alfa + h)) - Eq(w1(alfa - h))) / (2 * h); ana = -gq(w1(alfa)) * gq(w)
print("dE(t+1)/dalfa: numérico %.6f | analítico -g(t+1)g(t) = %.6f" % (num, ana))

# ---------------------------------------------------------------- quadráticas diagonais: GD, momento, DD, DBD, EDBD
def quad(lam):
    lam = np.array(lam, float); return (lambda v: 0.5 * float(np.sum(lam * v * v))), (lambda v: lam * v)
def itera(otim, lam, x0, lim=1e-12, **kw):
    f, g = quad(lam); r = otim(np.array(x0, float), f, g, tol_grad=0, **kw); L = np.array(r.historico["perda"]); thr = 0.5 * min(lam) * lim
    return (int(np.argmax(L < thr)) if (L < thr).any() else None), r
configs = [("Gradiente descendente ($\\eta=1/\\lambda_{\\max}$)", O.gradiente_descendente, lambda lm: dict(taxa=1 / max(lm), max_iter=20000), COR["gry"]),
           ("Gradiente descendente ($\\eta^*=2/(\\lambda_{\\min}+\\lambda_{\\max})$)", O.gradiente_descendente, lambda lm: dict(taxa=2 / (1 + max(lm)), max_iter=20000), COR["amb"]),
           ("GD com momento ($\\beta=0{,}9$)", O.gradiente_descendente, lambda lm: dict(taxa=0.5 / max(lm), momento=0.9, max_iter=20000), COR["red"]),
           ("Delta-delta", O.delta_delta, lambda lm: dict(alfa0=0.01, gama=1e-3, max_iter=5000), COR["vio"]),
           ("Delta-Bar-Delta", O.delta_bar_delta, lambda lm: dict(alfa0=0.01, kappa=0.005, phi=0.2, teta=0.7, max_iter=5000), COR["ind"]),
           ("EDBD (didático)", O.delta_bar_delta_estendido, lambda lm: dict(alfa0=0.05, kappa_a=0.05, phi_a=0.3, alfa_max=1.0, kappa_m=0.05, mu_max=0.9, max_iter=5000), COR["grn"])]
for lamv in ([1, 10], [1, 10, 100, 1000]):
    print("quadrática diagonal, autovalores", lamv, "(condição %g): iterações até f < 0.5*lam_min*1e-12" % (max(lamv) / min(lamv)))
    for nome, otim, kw, _ in configs:
        it, r = itera(otim, lamv, np.ones(len(lamv)), **kw(lamv)); print("   %-60s %s" % (nome.replace("$", ""), it))
# trajetórias 2D e taxas
lamv = [1, 10]; f, g = quad(lamv)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9)); gx, gy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-1.6, 1.6, 200)); ax[0].contour(gx, gy, 0.5 * (gx ** 2 + 10 * gy ** 2), levels=[0.5, 2, 5, 10, 16, 25], colors="#CBD5E1", linewidths=0.8)
def traj(otim, kw, x0=(2.5, 1.2), n=60):
    pts = [np.array(x0)]; th = np.array(x0, float)
    # reexecuta passo a passo: usa max_iter=k para k = 1..n (barato aqui)
    for k in range(1, n + 1):
        r = otim(np.array(x0, float), f, g, tol_grad=0, **{**kw, "max_iter": k}); pts.append(r.theta.copy())
    return np.array(pts)
for nome, otim, kw, c in [configs[1], configs[4], configs[5]]:
    P = traj(otim, kw(lamv), n=40); ax[0].plot(P[:, 0], P[:, 1], "o-", ms=2.5, lw=1.1, color=c, label=nome)
ax[0].set_xlim(-3, 3); ax[0].set_ylim(-1.6, 1.6); ax[0].legend(fontsize=7); ax[0].set_title("Trajetórias na calha $\\frac{1}{2}(x^2+10y^2)$"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
_, rd = itera(O.delta_bar_delta, [1, 10], [1, 1], **configs[4][2]([1, 10]))
fd = O.delta_bar_delta(np.array([2.5, 1.2]), f, g, alfa0=0.01, kappa=0.005, phi=0.2, teta=0.7, max_iter=80, tol_grad=0)
# taxas individuais: executar com 2 pesos e guardar alfa por coordenada (reconstrução manual)
th = np.array([2.5, 1.2]); alf = np.full(2, 0.01); gb = np.zeros(2); A = []
for k in range(80):
    gg = g(th); prod = gb * gg; alf = np.where(prod > 0, alf + 0.005, np.where(prod < 0, alf * 0.8, alf)); th = th - alf * gg; gb = 0.3 * gg + 0.7 * gb; A.append(alf.copy())
A = np.array(A); ax[1].plot(A[:, 0], color=COR["ind"], label="$\\alpha_x$ (curvatura 1)"); ax[1].plot(A[:, 1], color=COR["red"], label="$\\alpha_y$ (curvatura 10)"); ax[1].axhline(2 / 11, color=COR["gry"], ls=":"); ax[1].text(45, 0.19, "$\\eta^*=2/11$", color=COR["gry"], fontsize=8)
ax[1].set_xlabel("iteração"); ax[1].set_ylabel("taxa individual $\\alpha_i$"); ax[1].legend(fontsize=8); ax[1].set_title("DBD: cada coordenada encontra sua taxa"); salvar(fig, "f8_quadratica")
print("DBD na calha: taxas após 80 passos: alfa_x = %.3f, alfa_y = %.3f | (1/lambda_x = 1, 1/lambda_y = 0.1)" % (A[-1, 0], A[-1, 1]))

# ---------------------------------------------------------------- delta-delta: sensibilidade a gama (falhas)
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6)); lamv = [1, 10]; f, g = quad(lamv)
for gama, c in [(1e-3, COR["grn"]), (1e-2, COR["amb"]), (1e-1, COR["red"])]:
    th = np.array([2.5, 1.2]); alf = np.full(2, 0.01); ga = np.zeros(2); A = []; F = []
    for k in range(60):
        gg = g(th); alf = alf + gama * gg * ga; th = th - alf * gg; ga = gg; A.append(alf.copy()); F.append(f(th))
    A = np.array(A); F = np.array(F); ax[0].plot(A[:, 1], color=c, label=f"$\\gamma={gama:g}$"); ax[1].semilogy(np.maximum(F, 1e-300), color=c, label=f"$\\gamma={gama:g}$")
    print("delta-delta gama=%g: alfa_y após 60 passos %.3g | f final %.3g | alfa_y mín %.3g máx %.3g" % (gama, A[-1, 1], F[-1], A[:, 1].min(), A[:, 1].max()))
ax[0].set_ylim(-2, 2); ax[0].axhline(0, color=COR["gry"], lw=0.8); ax[0].set_xlabel("iteração"); ax[0].set_ylabel("$\\alpha_y$"); ax[0].legend(fontsize=8); ax[0].set_title("Delta-delta: a taxa pode ficar negativa ou explodir")
ax[1].set_xlabel("iteração"); ax[1].set_ylabel("$f$ (log)"); ax[1].legend(fontsize=8); ax[1].set_title("Convergência ou divergência conforme $\\gamma$"); salvar(fig, "f8_delta_delta")

# ---------------------------------------------------------------- rede 1-8-1: protocolo com sementes de ajuste (0-4) e avaliação (5-14)
N = 100; rng = np.random.default_rng(3); X = rng.uniform(0, 1, (N, 1)); T = np.sin(2 * np.pi * X) + rng.normal(0, 0.15, (N, 1))
def run(otim, seed, it=300, **kw):
    r = MLP([1, 8, 1], ["tanh", "identidade"], semente=seed); t0 = time.perf_counter()
    res = otim(r.theta, lambda th: r.perda(th, X, T) / N, lambda th: r.grad(th, X, T) / N, tol_grad=0, max_iter=it, **kw); return res, time.perf_counter() - t0
def mediana(otim, seeds, **kw):
    v = []
    for s in seeds:
        try: x = run(otim, s, **kw)[0].historico["perda"][-1]
        except Exception: x = np.inf
        v.append(x if np.isfinite(x) else 1e9)
    return float(np.median(v)), float(max(v))
tune, aval = range(5), range(5, 15)
best = {}
best["GD"] = min([dict(taxa=e) for e in (0.2, 0.5, 1.0, 1.5)], key=lambda kw: mediana(O.gradiente_descendente, tune, **kw)[0])
best["GD+momento"] = min([dict(taxa=e, momento=b) for e in (0.05, 0.1, 0.2, 0.3) for b in (0.5, 0.9)], key=lambda kw: mediana(O.gradiente_descendente, tune, **kw)[0])
best["Delta-delta"] = min([dict(alfa0=a, gama=gm) for a in (0.05, 0.1, 0.3) for gm in (1e-3, 1e-2, 1e-1)], key=lambda kw: mediana(O.delta_delta, tune, **kw)[0])
best["DBD"] = min([dict(alfa0=a, kappa=k, phi=p, teta=t) for a, k, p, t in itertools.product((0.1, 0.3), (0.01, 0.05, 0.2), (0.1, 0.3), (0.5, 0.9))], key=lambda kw: mediana(O.delta_bar_delta, tune, **kw)[0])
best["EDBD"] = min([dict(alfa0=a, kappa_a=ka, phi_a=pa, alfa_max=am, kappa_m=km, mu_max=mm) for a, ka, pa, am, km, mm in itertools.product((0.1, 0.3), (0.05, 0.2), (0.1, 0.3), (2.0, 5.0), (0.01, 0.05), (0.5, 0.9))], key=lambda kw: mediana(O.delta_bar_delta_estendido, tune, **kw)[0])
ots = {"GD": O.gradiente_descendente, "GD+momento": O.gradiente_descendente, "Delta-delta": O.delta_delta, "DBD": O.delta_bar_delta, "EDBD": O.delta_bar_delta_estendido}
for k, v in best.items(): print("melhor (sementes 0-4) %-12s %s" % (k, v))
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.8)); cores = {"GD": COR["gry"], "GD+momento": COR["red"], "Delta-delta": COR["vio"], "DBD": COR["ind"], "EDBD": COR["grn"]}; finais = {}
for nome in ots:
    H = []; F = []; tt = []
    for s in aval:
        res, dt = run(ots[nome], s, **best[nome]); h = res.historico["perda"]; H.append(h); F.append(h[-1] if np.isfinite(h[-1]) else 1e9); tt.append(dt)
    def _fix(h, n=None):
        h = np.asarray(h, float); fin = np.isfinite(h)
        ult = h[fin][-1] if fin.any() else 1e9
        h = np.where(fin, h, ult)
        return h
    L = max(len(h) for h in H)
    H = np.array([np.concatenate([_fix(h), np.full(L - len(h), _fix(h)[-1])]) for h in H]); Hm = np.median(H, axis=0); finais[nome] = F
    ax[0].semilogy(Hm, color=cores[nome], label=nome); cruz = int(np.argmax(Hm < 0.03)) if (Hm < 0.03).any() else None
    print("avaliação (sementes 5-14) %-12s perda final mediana %.4f [mín %.4f, máx %.4f] | época em que a mediana cruza 0.03: %s | tempo %.3f s" % (nome, np.median(F), min(F), max(F), cruz, np.mean(tt)))
ax[0].axhline(0.5 * 0.15 ** 2, color=COR["gry"], ls=":"); ax[0].legend(fontsize=8); ax[0].set_xlabel("época"); ax[0].set_ylabel("perda $E/N$ (mediana de 10 sementes)"); ax[0].set_title("Rede 1–8–1 (hiperparâmetros ajustados em outras sementes)")
ax[1].boxplot([finais[n] for n in ots], tick_labels=[n.replace("Delta-delta", "Delta-\ndelta").replace("GD+momento", "GD +\nmomento") for n in ots]); ax[1].set_yscale("log"); ax[1].set_ylabel("perda final $E/N$"); ax[1].set_title("Dispersão da perda final")
salvar(fig, "f8_rede")
# taxas individuais da DBD na rede
res, _ = run(O.delta_bar_delta, 5, **best["DBD"]); print("DBD (semente 5): alfa mín %.3g | mediana %.3g | máx %.3g na última época; alfa0 = %g" % (res.historico["alfa_min"][-1], res.historico["alfa_med"][-1], res.historico["alfa_max"][-1], best["DBD"]["alfa0"]))
res, _ = run(O.delta_bar_delta_estendido, 5, **best["EDBD"]); print("EDBD (semente 5): alfa mín %.3g | mediana %.3g | máx %.3g; momento mediano %.3g" % (res.historico["alfa_min"][-1], res.historico["alfa_med"][-1], res.historico["alfa_max"][-1], res.historico["mom_med"][-1]))
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.5)); r1, _ = run(O.delta_bar_delta, 5, **best["DBD"]); r2, _ = run(O.delta_bar_delta_estendido, 5, **best["EDBD"])
for r_, a_, tit in [(r1, ax[0], "DBD"), (r2, ax[1], "EDBD")]:
    a_.semilogy(r_.historico["alfa_min"], color=COR["gry"], label="$\\min\\alpha_i$"); a_.semilogy(r_.historico["alfa_med"], color=COR["ind"], label="mediana"); a_.semilogy(r_.historico["alfa_max"], color=COR["red"], label="$\\max\\alpha_i$"); a_.set_xlabel("época"); a_.set_ylabel("taxa individual"); a_.legend(fontsize=8); a_.set_title(f"{tit}: taxas individuais (semente 5)")
salvar(fig, "f8_taxas")

# ---------------------------------------------------------------- sensibilidade da DBD (kappa x phi) em sementes de avaliação
kap = (0.005, 0.02, 0.05, 0.2); phs = (0.05, 0.1, 0.3, 0.5); G = np.zeros((len(phs), len(kap)))
for i, p in enumerate(phs):
    for j, k in enumerate(kap): G[i, j] = mediana(O.delta_bar_delta, aval, alfa0=0.1, kappa=k, phi=p, teta=0.9)[0]
print("DBD, sensibilidade (alfa0=0.1, teta=0.9), mediana da perda final em 10 sementes; linhas phi, colunas kappa", kap); [print("  phi=%.2f:" % p, np.round(G[i], 4)) for i, p in enumerate(phs)]
tets = (0.0, 0.3, 0.7, 0.9, 0.99); S = [mediana(O.delta_bar_delta, aval, alfa0=0.1, kappa=0.05, phi=0.1, teta=t)[0] for t in tets]; print("DBD, efeito de teta (alfa0=0.1, kappa=0.05, phi=0.1):", dict(zip(tets, np.round(S, 4))))
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6)); im = ax[0].imshow(np.log10(G), cmap="viridis_r", aspect="auto"); ax[0].set_xticks(range(len(kap))); ax[0].set_xticklabels(kap); ax[0].set_yticks(range(len(phs))); ax[0].set_yticklabels(phs); ax[0].set_xlabel("$\\kappa$ (aumento aditivo)"); ax[0].set_ylabel("$\\phi$ (redução)")
for i in range(len(phs)):
    for j in range(len(kap)): ax[0].text(j, i, "%.3f" % G[i, j] if G[i, j] < 1e3 else "div.", ha="center", va="center", color="white", fontsize=8)
ax[0].set_title("DBD: perda final mediana ($\\log_{10}$ em cor)"); ax[1].semilogy(range(len(tets)), S, "o-", color=COR["ind"]); ax[1].set_xticks(range(len(tets))); ax[1].set_xticklabels(tets); ax[1].set_xlabel("$\\theta$ (peso da memória do gradiente)"); ax[1].set_ylabel("perda final mediana"); ax[1].set_title("DBD: efeito de $\\theta$"); salvar(fig, "f8_sensibilidade")

# ---------------------------------------------------------------- mecanismo: g, g_bar e produto em um peso
r = MLP([1, 8, 1], ["tanh", "identidade"], semente=5); th = r.theta.copy(); alf = np.full_like(th, 0.1); gb = np.zeros_like(th); reg = []
for k in range(120):
    gg = r.grad(th, X, T) / N; prod = gb * gg; alf = np.where(prod > 0, alf + 0.05, np.where(prod < 0, alf * 0.9, alf)); th = th - alf * gg; gb = 0.1 * gg + 0.9 * gb; reg.append((gg[3], gb[3], np.sign(prod[3]), alf[3]))
reg = np.array(reg); fig, ax = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True); ax[0].plot(reg[:, 0], color=COR["gry"], lw=1, label="$g(t)$ (gradiente)"); ax[0].plot(reg[:, 1], color=COR["ind"], lw=2, label="$\\bar g(t)$ (média exponencial)"); ax[0].legend(fontsize=8); ax[0].set_ylabel("derivada do peso 4")
ax[1].plot(reg[:, 3], color=COR["red"]); ax[1].set_ylabel("taxa $\\alpha_4$"); ax[1].set_xlabel("época"); salvar(fig, "f8_mecanismo")
print("OK cap08")
