"""Cap. 7 - Levenberg-Marquardt: experimentos e figuras (semente fixa; números citados no texto)."""
import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import least_squares
from rnbook import MLP, grad_numerico
from rnbook import otimizadores as O
from rnbook.estilo import plt, COR, salvar
from rnbook.desenho import caixa, seta

# ------------------------------------------------------------ 7.1.1 pontos críticos de f = x^3 - 3x + y^2
f = lambda v: v[0] ** 3 - 3 * v[0] + v[1] ** 2; gf = lambda v: np.array([3 * v[0] ** 2 - 3, 2 * v[1]]); Hf = lambda v: np.array([[6 * v[0], 0], [0, 2.0]])
for pc in ([1, 0], [-1, 0]): print("ponto crítico", pc, "grad", gf(np.array(pc, float)), "autovalores da Hessiana", np.linalg.eigvalsh(Hf(np.array(pc, float))))

# ------------------------------------------------------------ 7.1.2 Newton x Newton modificado em uma variável
phi = lambda y: (y ** 2 - 1) ** 2 / 4; dphi = lambda y: y * (y ** 2 - 1); d2phi = lambda y: 3 * y ** 2 - 1
y = 0.1; print("Newton puro: y0=0.1  phi'=%.4f phi''=%.4f -> y1 = %.4f (vai ao máximo local em y=0)" % (dphi(y), d2phi(y), y - dphi(y) / d2phi(y)))
tau = 1.1; ym = [0.1]
for _ in range(5): ym.append(ym[-1] - dphi(ym[-1]) / (d2phi(ym[-1]) + max(0.0, tau if d2phi(ym[-1]) < 0.1 else 0.0)))
print("Newton modificado (H+tau I quando H<=0.1): ", np.round(ym, 4))
yy = np.linspace(-1.6, 1.6, 300); fig, ax = plt.subplots(figsize=(6.4, 3.8)); ax.plot(yy, phi(yy), color=COR["ink"], lw=2, label="$\\varphi(y)=(y^2-1)^2/4$")
y0 = 0.1; ax.scatter([y0], [phi(y0)], color=COR["ind"], zorder=5); ax.text(y0 + 0.05, phi(y0) + 0.02, "$y_0$", color=COR["ind"])
q = lambda y_, c: phi(y0) + dphi(y0) * (y_ - y0) + 0.5 * c * (y_ - y0) ** 2
ax.plot(yy, q(yy, d2phi(y0)), "--", color=COR["red"], label="modelo de Newton ($\\varphi''<0$: parábola para baixo)"); ax.plot(yy, q(yy, d2phi(y0) + tau), "--", color=COR["grn"], label="modelo com $\\varphi''+\\tau$ ($\\tau=1{,}1$)")
ax.set_ylim(-0.1, 0.6); ax.legend(fontsize=7.5); ax.set_xlabel("y"); ax.set_title("Hessiana negativa: Newton busca o máximo; $H+\\tau I$ corrige"); salvar(fig, "f7_newton_modificado")

# ------------------------------------------------------------ Rosenbrock: Newton, GN, LM, GD
res_r = lambda v: np.array([10 * (v[1] - v[0] ** 2), 1 - v[0]]); jac_r = lambda v: np.array([[-20 * v[0], 10.0], [-1.0, 0.0]])
Er = lambda v: 0.5 * float(res_r(v) @ res_r(v)); gr_ = lambda v: jac_r(v).T @ res_r(v)
def hess_exata(v): J = jac_r(v); r = res_r(v); S = r[0] * np.array([[-20.0, 0], [0, 0]]); return J.T @ J + S
v0 = np.array([-1.2, 1.0]); traj = {}
v = v0.copy(); tr = [v.copy()]
for k in range(30):
    g = gr_(v)
    if np.linalg.norm(g) < 1e-10: break
    v = v + np.linalg.solve(hess_exata(v), -g); tr.append(v.copy())
traj["Newton (Hessiana exata)"] = np.array(tr); print("Newton exato: %d iterações; ponto final %s" % (len(tr) - 1, np.round(tr[-1], 8)))
r_gn = O.gauss_newton(v0, res_r, jac_r, max_iter=100, tol_grad=1e-10); print("Gauss-Newton puro: %d iterações; E=%.2e; motivo %s" % (r_gn.n_iter, r_gn.historico["perda"][-1], r_gn.motivo))
def trilha(otim, **kw):
    pts = [v0.copy()]
    r = otim(v0, res_r, jac_r, **kw); return r
r_lm = O.levenberg_marquardt(v0, res_r, jac_r, max_iter=100, tol_grad=1e-10); r_nl = O.levenberg_marquardt_nielsen(v0, res_r, jac_r, max_iter=100, tol_grad=1e-10)
print("LM (Hagan-Menhaj, mu0=1e-3, x10): %d iterações; E=%.2e (%s)" % (r_lm.n_iter, r_lm.historico["perda"][-1], r_lm.motivo))
print("LM (Nielsen, razão de ganho): %d iterações; E=%.2e (%s)" % (r_nl.n_iter, r_nl.historico["perda"][-1], r_nl.motivo))
r_gd = O.gradiente_descendente(v0, Er, gr_, taxa=1e-3, max_iter=2000); print("GD (eta=1e-3, 2000 passos): E=%.3e ponto %s" % (r_gd.historico["perda"][-1], np.round(r_gd.theta, 4)))
# trajetórias para a figura: reexecuta LM guardando theta
def lm_traj(res, jac, th0, mu0=1e-3, it=60):
    th = th0.copy(); mu = mu0; tr = [th.copy()]; r = res(th); E = 0.5 * r @ r
    for _ in range(it):
        J = jac(th); g = J.T @ r
        if np.linalg.norm(g) < 1e-10: break
        for __ in range(30):
            d = np.linalg.solve(J.T @ J + mu * np.eye(len(th)), -g); rn = res(th + d); En = 0.5 * rn @ rn
            if En < E: th, r, E = th + d, rn, En; mu *= 0.1; tr.append(th.copy()); break
            mu *= 10
    return np.array(tr)
traj["Levenberg–Marquardt"] = lm_traj(res_r, jac_r, v0)
gd_tr = [v0.copy()]; v = v0.copy()
for _ in range(300): v = v - 1e-3 * gr_(v); gd_tr.append(v.copy())
traj["Gradiente descendente (300 passos)"] = np.array(gd_tr)
gx, gy = np.meshgrid(np.linspace(-1.6, 1.4, 300), np.linspace(-0.4, 1.8, 300)); Z = 0.5 * (100 * (gy - gx ** 2) ** 2 + (1 - gx) ** 2)
fig, ax = plt.subplots(figsize=(6.6, 4.2)); ax.contour(gx, gy, np.log10(Z + 1e-3), levels=np.linspace(-2, 3, 14), colors="#CBD5E1", linewidths=0.8)
for (nm, T_), c in zip(traj.items(), [COR["grn"], COR["ind"], COR["amb"]]): ax.plot(T_[:, 0], T_[:, 1], "o-", ms=3, lw=1.3, color=c, label=f"{nm}: {len(T_)-1} passos")
ax.scatter([1], [1], marker="*", s=140, color=COR["red"], zorder=6, label="mínimo (1, 1)"); ax.scatter(*v0, color=COR["ink"], zorder=6); ax.legend(fontsize=7.5); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Rosenbrock (mínimos quadrados, $m=n=2$)")
salvar(fig, "f7_rosenbrock")

# ------------------------------------------------------------ Hessiana exata x Gauss-Newton em uma rede (resíduo pequeno x grande)
rng = np.random.default_rng(3); N = 100; X = rng.uniform(0, 1, (N, 1)); T = np.sin(2 * np.pi * X) + rng.normal(0, 0.15, (N, 1))
rede = MLP([1, 8, 1], ["tanh", "identidade"], semente=0); P = rede.n_params
res = lambda th: rede.residuos(th, X, T); jac = lambda th: rede.jacobiana(th, X)
def hess_num(th, h=1e-5):
    H = np.zeros((P, P)); g = lambda t: rede.grad(t, X, T, "sse")
    for i in range(P):
        e = np.zeros(P); e[i] = h; H[:, i] = (g(th + e) - g(th - e)) / (2 * h)
    return 0.5 * (H + H.T)
th0 = rede.theta.copy(); r_ = O.levenberg_marquardt(th0, res, jac, max_iter=60); thf = r_.theta
for nome, th in [("inicial", th0), ("após LM (60 it)", thf)]:
    H = hess_num(th); J = jac(th); GN = J.T @ J; r = res(th)
    print("%-16s E=%.4f | ||r||=%.3f | ||H-JtJ||/||H|| = %.4f | autovalores de H: min %.2e máx %.2e | de JtJ: min %.2e máx %.2e | cond(JtJ)=%.2e" % (nome, 0.5 * r @ r, np.linalg.norm(r), np.linalg.norm(H - GN) / np.linalg.norm(H), np.linalg.eigvalsh(H).min(), np.linalg.eigvalsh(H).max(), np.linalg.eigvalsh(GN).min(), np.linalg.eigvalsh(GN).max(), np.linalg.cond(GN)))
print("posto de J no ponto inicial (N*m x P = %d x %d): %d | posto após LM: %d" % (N, P, np.linalg.matrix_rank(jac(th0)), np.linalg.matrix_rank(jac(thf))))

# ------------------------------------------------------------ efeito de mu: interpolação entre GN e gradiente
th = th0.copy(); J = jac(th); r = res(th); g = J.T @ r; JTJ = J.T @ J
mus = 10.0 ** np.arange(-6, 7); nd, ang = [], []; gd_dir = -g / np.linalg.norm(g)
for mu in mus:
    d = np.linalg.solve(JTJ + mu * np.eye(P), -g); nd.append(np.linalg.norm(d)); ang.append(np.degrees(np.arccos(np.clip(d @ gd_dir / np.linalg.norm(d), -1, 1))))
print("normas do passo:", {f"{m:.0e}": f"{v:.3g}" for m, v in zip(mus, nd)}); print("ângulo com -g (graus):", {f"{m:.0e}": f"{v:.1f}" for m, v in zip(mus, ang)})
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6)); ax[0].loglog(mus, nd, "o-", color=COR["ind"]); ax[0].loglog(mus[-5:], np.linalg.norm(g) / mus[-5:], "--", color=COR["red"], label="$\\|g\\|/\\mu$ (limite: gradiente)"); ax[0].legend(fontsize=8)
ax[0].set_xlabel("amortecimento $\\mu$"); ax[0].set_ylabel("$\\|\\mathbf{d}(\\mu)\\|$"); ax[0].set_title("Comprimento do passo de LM em função de $\\mu$")
ax[1].semilogx(mus, ang, "s-", color=COR["grn"]); ax[1].set_xlabel("amortecimento $\\mu$"); ax[1].set_ylabel("ângulo entre $\\mathbf{d}$ e $-\\mathbf{g}$ (graus)"); ax[1].set_title("De Gauss–Newton ($\\mu\\to0$) ao gradiente ($\\mu\\to\\infty$)")
salvar(fig, "f7_mu")

# ------------------------------------------------------------ algoritmos na rede 1-8-1: iterações até E/N < limiar; comparação
def hist(nome, r_): return np.array(r_.historico["perda"])
runs = {}
runs["Levenberg–Marquardt (HM)"] = O.levenberg_marquardt(th0, res, jac, max_iter=60, tol_grad=1e-10)
runs["LM Nielsen"] = O.levenberg_marquardt_nielsen(th0, res, jac, max_iter=60, tol_grad=1e-10)
runs["iRPROP−"] = O.rprop(th0, lambda t: rede.perda(t, X, T), lambda t: rede.grad(t, X, T), variante="irprop-", max_iter=300, tol_grad=0)
runs["Quickprop"] = O.quickprop(th0, lambda t: rede.perda(t, X, T), lambda t: rede.grad(t, X, T), epsilon=0.05 / N, max_iter=300, tol_grad=0)
runs["Gradiente descendente"] = O.gradiente_descendente(th0, lambda t: rede.perda(t, X, T), lambda t: rede.grad(t, X, T), taxa=0.5 / N, max_iter=300, tol_grad=0)
lim = 1.1 * 0.5 * N * 0.15 ** 2  # 10% acima do piso do ruído
print("limiar de perda (10%% acima do piso do ruído, E soma): %.4f" % lim)
fig, ax = plt.subplots(figsize=(6.6, 3.9)); cores = [COR["ind"], COR["vio"], COR["grn"], COR["amb"], COR["red"]]
for (nm, r_), c in zip(runs.items(), cores):
    h = np.array(r_.historico["perda"]); k = int(np.argmax(h < lim)) if (h < lim).any() else None; ax.semilogy(h, color=c, label=nm)
    print("%-26s iterações até E < limiar: %s | perda final %.4f | tempo %.3f s | iterações executadas %d" % (nm, k, h[-1], r_.tempo, r_.n_iter))
ax.axhline(0.5 * N * 0.15 ** 2, color=COR["gry"], ls=":"); ax.set_xlim(0, 120); ax.set_xlabel("iteração (época)"); ax.set_ylabel("perda $E$ (soma)"); ax.legend(fontsize=7.5); ax.set_title("Rede 1–8–1: iterações de cada algoritmo"); salvar(fig, "f7_comparacao")

# ------------------------------------------------------------ mu e aceitação/rejeição (LM HM): histórico
r_ = runs["Levenberg–Marquardt (HM)"]; mu_h = np.array(r_.historico["mu"]); print("LM: mu inicial %.1e, final %.1e; mu mín %.1e máx %.1e; iterações %d" % (mu_h[0], mu_h[-1], mu_h.min(), mu_h.max(), r_.n_iter))
# ------------------------------------------------------------ custo e memória em função de P
tempos = {}
for Pp in (100, 200, 400, 800, 1600):
    A = np.random.default_rng(0).normal(size=(Pp, Pp)); M = A.T @ A + np.eye(Pp); b = np.ones(Pp); t0 = time.perf_counter()
    for _ in range(3): np.linalg.solve(M, b)
    tempos[Pp] = (time.perf_counter() - t0) / 3
print("tempo de resolução (J^T J + mu I) d = -g (s):", {p: round(t_, 4) for p, t_ in tempos.items()}, "| razões sucessivas:", [round(tempos[b_] / tempos[a_], 1) for a_, b_ in zip(list(tempos)[:-1], list(tempos)[1:])])
for Pp in (25, 1000, 10000, 100000): print("P=%d: J^T J ocupa %.2e bytes (%.2f MB)" % (Pp, 8.0 * Pp * Pp, 8.0 * Pp * Pp / 1e6))
print("J para N=100, m=1, P=25: %d x %d = %d elementos; para N=60000, m=10, P=10000: %.2e elementos (%.1f GB)" % (100, 25, 2500, 60000 * 10 * 10000.0, 8 * 60000 * 10 * 10000.0 / 1e9))

# ------------------------------------------------------------ SciPy: least_squares 'lm' x 'trf' x didático
cont = {"nfev": 0}
def res_c(th): cont["nfev"] += 1; return res(th)
def roda_scipy(metodo, **kw):
    cont["nfev"] = 0; t0 = time.perf_counter(); s = least_squares(res_c, th0, jac=jac, method=metodo, max_nfev=kw.get("max_nfev", 200), xtol=1e-12, ftol=1e-12, gtol=1e-12); return s, cont["nfev"], time.perf_counter() - t0
for metodo in ("lm", "trf"):
    s, nf, dt = roda_scipy(metodo); print("scipy least_squares(method=%r): E=%.6f | status %d | nfev(res) %d, njev %s | tempo %.3f s" % (metodo, 0.5 * s.fun @ s.fun, s.status, nf, s.njev, dt))
t0 = time.perf_counter(); r_hm = O.levenberg_marquardt(th0, res, jac, max_iter=200, tol_grad=1e-12, tol_passo=1e-14); dt = time.perf_counter() - t0
print("LM didático (HM): E=%.6f | iterações %d | motivo %s | tempo %.3f s" % (r_hm.historico["perda"][-1], r_hm.n_iter, r_hm.motivo, dt))
r_nn = O.levenberg_marquardt_nielsen(th0, res, jac, max_iter=200, tol_grad=1e-12, tol_passo=1e-14); print("LM didático (Nielsen): E=%.6f | iterações %d | motivo %s" % (r_nn.historico["perda"][-1], r_nn.n_iter, r_nn.motivo))
# várias sementes: mínimo local
vals = {"HM": [], "Nielsen": [], "scipy lm": []}
for s_ in range(10):
    rd = MLP([1, 8, 1], ["tanh", "identidade"], semente=s_); rs = lambda th: rd.residuos(th, X, T); jc = lambda th: rd.jacobiana(th, X)
    vals["HM"].append(O.levenberg_marquardt(rd.theta, rs, jc, max_iter=200).historico["perda"][-1]); vals["Nielsen"].append(O.levenberg_marquardt_nielsen(rd.theta, rs, jc, max_iter=200).historico["perda"][-1]); vals["scipy lm"].append(0.5 * (lambda s: s.fun @ s.fun)(least_squares(rs, rd.theta, jac=jc, method="lm", max_nfev=400)))
for k, v in vals.items(): print("10 sementes %-9s: E final mediana %.4f [mín %.4f, máx %.4f]" % (k, np.median(v), min(v), max(v)))
print("piso do ruído E = N sigma^2/2 = %.4f" % (0.5 * N * 0.15 ** 2))
print("OK cap07")
