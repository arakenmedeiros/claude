"""Cap. 5 - Quickprop: experimentos e figuras (semente fixa; números citados no texto)."""
import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
from rnbook import MLP
from rnbook import otimizadores as O
from rnbook.estilo import plt, COR, salvar
from rnbook.desenho import caixa, seta

# ------------------------------------------------------------- 5.8 : quickprop 1D = secante na derivada
f = lambda w: np.exp(w) - 2 * w; s_ = lambda w: np.exp(w) - 2
w_prev = 0.0; eps = 0.3; dw = -eps * s_(w_prev); w = w_prev + dw      # 1º passo: gradiente
traj = [w_prev, w]
for _ in range(5):
    S, Sp = s_(traj[-1]), s_(traj[-2]); dwp = traj[-1] - traj[-2]
    traj.append(traj[-1] + S / (Sp - S) * dwp)
print("Quickprop 1D (f=e^w-2w, w0=0, eps=0.3):", [round(v, 6) for v in traj], "| ln2 =", round(np.log(2), 6))
sec = [traj[0], traj[1]]
for _ in range(5): sec.append(sec[-1] - s_(sec[-1]) * (sec[-1] - sec[-2]) / (s_(sec[-1]) - s_(sec[-2])))
print("máxima diferença entre Quickprop e secante em f':", max(abs(a - b) for a, b in zip(traj, sec)))
# quadrática exata: um passo
q = lambda w: 3 * (w - 2) ** 2 + 1; sq = lambda w: 6 * (w - 2)
w0, w1 = 0.0, 0.5; w2 = w1 + sq(w1) / (sq(w0) - sq(w1)) * (w1 - w0); print("quadrática 3(w-2)^2+1: w0=0,w1=0.5 -> w2 =", w2, "(mínimo em 2)")
# curvatura negativa: parábola para cima? (função cúbica com f''<0)
c = lambda w: -(w ** 3) / 3 + w; sc = lambda w: -(w ** 2) + 1
w0, w1 = 0.5, 0.8; w2 = w1 + sc(w1) / (sc(w0) - sc(w1)) * (w1 - w0)
print("curvatura negativa f=-w^3/3+w (f''<0 em w>0): w0=0.5, w1=0.8 -> passo quadrático leva a", round(w2, 4), "(f'' <0: isto é um MÁXIMO local em w=1, não um mínimo)")
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.7)); ww = np.linspace(-0.4, 1.6, 300)
ax[0].plot(ww, f(ww), color=COR["ink"], lw=2, label="$E(w)=e^w-2w$")
a0, a1 = traj[0], traj[1]; Sa0, Sa1 = s_(a0), s_(a1); curv = (Sa1 - Sa0) / (a1 - a0)
ax[0].plot(ww, f(a1) + Sa1 * (ww - a1) + 0.5 * curv * (ww - a1) ** 2, "--", color=COR["amb"], label="parábola local (curvatura da secante)")
ax[0].scatter(traj[:4], [f(v) for v in traj[:4]], color=COR["ind"], zorder=5); [ax[0].text(v, f(v) + 0.12, f"$w_{i}$", ha="center", fontsize=9, color=COR["indD"]) for i, v in enumerate(traj[:4])]
ax[0].set_ylim(0.4, 2.6); ax[0].legend(fontsize=8); ax[0].set_xlabel("w"); ax[0].set_title("Quickprop em uma variável")
ww2 = np.linspace(-0.2, 1.5, 300); ax[1].plot(ww2, sc(ww2), color=COR["ink"], lw=2, label="$E'(w)=1-w^2$ ($E=-w^3/3+w$)"); ax[1].axhline(0, color=COR["gry"], lw=0.8)
ax[1].plot([0.5, 0.8], [sc(0.5), sc(0.8)], "o--", color=COR["red"]); ax[1].scatter([w2], [0], color=COR["red"], marker="x", s=80, zorder=5)
ax[1].text(w2, 0.18, "passo quadrático", fontsize=8, color=COR["red"], ha="center"); ax[1].legend(fontsize=8, loc="lower left"); ax[1].set_xlabel("w"); ax[1].set_title("Falha: curvatura negativa → o passo vai a um máximo")
salvar(fig, "f5_parabola")

# ------------------------------------------------------------- 5.5-5.7: gradientes explícitos x MLP.grad
def grad_explicito(W, b, V, c, X, T, phi, dphi, phio, dphio):
    """rede n-H-M: y_k = phio(sum_j V_kj phi(sum_i W_ji x_i + b_j) + c_k); E = 1/2 sum (y-t)^2. Laços explícitos."""
    N, n = X.shape; H = W.shape[0]; M = V.shape[0]
    gW, gb, gV, gc = np.zeros_like(W), np.zeros_like(b), np.zeros_like(V), np.zeros_like(c)
    for p in range(N):
        zj = np.array([W[j] @ X[p] + b[j] for j in range(H)]); aj = phi(zj)
        zk = np.array([V[k] @ aj + c[k] for k in range(M)]); yk = phio(zk)
        dk = (yk - T[p]) * dphio(zk)                       # delta da saída
        for k in range(M):
            gc[k] += dk[k]
            for j in range(H): gV[k, j] += dk[k] * aj[j]
        for j in range(H):
            dj = sum(dk[k] * V[k, j] for k in range(M)) * dphi(zj[j])        # delta oculto
            gb[j] += dj
            for i in range(n): gW[j, i] += dj * X[p, i]
    return gW, gb, gV, gc
rng = np.random.default_rng(1)
for (n, H, M, nome) in [(1, 4, 1, "5.5: 1 entrada, H ocultos, 1 saída"), (3, 5, 1, "5.6: n entradas, H ocultos, 1 saída"), (3, 5, 2, "5.7: n entradas, H ocultos, M saídas")]:
    X = rng.normal(size=(7, n)); T = rng.normal(size=(7, M)); r = MLP([n, H, M], ["tanh", "sigmoide"], semente=2)
    (Wm, bm), (Vm, cm) = r.desempacotar(r.theta)
    gW, gb, gV, gc = grad_explicito(Wm, bm, Vm, cm, X, T, np.tanh, lambda z: 1 - np.tanh(z) ** 2, lambda z: 1 / (1 + np.exp(-z)), lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z))))
    g = r.grad(r.theta, X, T); ge = np.concatenate([gW.ravel(), gb, gV.ravel(), gc]); print(f"{nome}: diferença máx. explícito x MLP.grad = {np.abs(g-ge).max():.2e}")

# ------------------------------------------------------------- GD x Quickprop em regressão (1-8-1, N=100)
rng = np.random.default_rng(3); N = 100; X = rng.uniform(0, 1, (N, 1)); T = np.sin(2 * np.pi * X) + rng.normal(0, 0.15, (N, 1))
def roda(otim, semente, **kw):
    r = MLP([1, 8, 1], ["tanh", "identidade"], semente=semente)
    fun = lambda th: r.perda(th, X, T) / N; gr = lambda th: r.grad(th, X, T) / N
    t0 = time.perf_counter(); res = otim(r.theta, fun, gr, **kw); return res, time.perf_counter() - t0
gradeG = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
print("GD (lote), perda final E/N após 300 épocas, mediana de 10 sementes:")
resG = {}
for eta in gradeG:
    v = [roda(O.gradiente_descendente, s, taxa=eta, max_iter=300, tol_grad=0)[0].historico["perda"][-1] for s in range(10)]; resG[eta] = np.median(v); print("  eta=%.2f: %.4f" % (eta, np.median(v)))
etaG = 0.5   # maior taxa que não diverge em nenhuma das 10 sementes (eta=1.0 tem mediana menor, porém diverge em uma semente)
gradeQ = [0.05, 0.1, 0.2, 0.5, 1.0]
resQ = {}
print("Quickprop (mu=1.75), perda final após 300 épocas, mediana de 10 sementes:")
for eps_ in gradeQ:
    v = [roda(O.quickprop, s, epsilon=eps_, max_iter=300, tol_grad=0)[0].historico["perda"][-1] for s in range(10)]; resQ[eps_] = np.median(v); print("  epsilon=%.2f: %.4f (mín %.4f, máx %.4f)" % (eps_, np.median(v), np.min(v), np.max(v)))
epsQ = min(resQ, key=resQ.get); print("eta escolhido (GD) =", etaG, "| melhor epsilon (Quickprop) =", epsQ)
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.7)); hG, hQ = [], []
for s in range(10):
    hG.append(roda(O.gradiente_descendente, s, taxa=etaG, max_iter=300, tol_grad=0)[0].historico["perda"]); hQ.append(roda(O.quickprop, s, epsilon=epsQ, max_iter=300, tol_grad=0)[0].historico["perda"])
for h in hG: ax[0].semilogy(h, color=COR["amb"], alpha=0.45, lw=1)
for h in hQ: ax[0].semilogy(h, color=COR["ind"], alpha=0.45, lw=1)
ax[0].semilogy([], [], color=COR["amb"], label=f"gradiente descendente ($\\eta$={etaG})"); ax[0].semilogy([], [], color=COR["ind"], label=f"Quickprop ($\\epsilon$={epsQ}, $\\mu$=1,75)"); ax[0].axhline(0.5 * 0.15 ** 2, color=COR["gry"], ls=":"); ax[0].legend(fontsize=8)
ax[0].set_xlabel("época"); ax[0].set_ylabel("perda $E/N$"); ax[0].set_title("10 sementes, rede 1–8–1, lote")
finG = [h[-1] for h in hG]; finQ = [h[-1] for h in hQ]
print("após 300 épocas: GD mediana %.4f [%.4f, %.4f] | Quickprop mediana %.4f [%.4f, %.4f] | piso do ruído %.5f" % (np.median(finG), min(finG), max(finG), np.median(finQ), min(finQ), max(finQ), 0.5 * 0.15 ** 2))
for lim in (0.03, 0.02):
    eG = [int(np.argmax(np.array(h) < lim)) if (np.array(h) < lim).any() else None for h in hG]; eQ = [int(np.argmax(np.array(h) < lim)) if (np.array(h) < lim).any() else None for h in hQ]
    print("épocas para E/N<%.2f: GD %s | Quickprop %s" % (lim, eG, eQ))
ax[1].boxplot([finG, finQ], tick_labels=["gradiente\ndescendente", "Quickprop"], showfliers=True); ax[1].set_yscale("log"); ax[1].set_ylabel("perda final $E/N$ (300 épocas)"); ax[1].set_title("Dispersão entre 10 sementes")
salvar(fig, "f5_quickprop_gd")

# ------------------------------------------------------------- sensibilidade a mu e epsilon
mus = [1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0]; fig, ax = plt.subplots(figsize=(6.4, 3.8)); tab = {}
for eps_ in (0.05, 0.2, 0.5):
    med = []; 
    for mu in mus:
        v = []
        for s in range(10):
            res = roda(O.quickprop, s, epsilon=eps_, mu=mu, max_iter=300, tol_grad=0)[0]; ff = res.historico["perda"][-1]; v.append(ff if np.isfinite(ff) else np.inf)
        med.append(np.median(v))
    tab[eps_] = med; ax.semilogy(mus, med, "o-", label=f"$\\epsilon$={eps_}")
ax.set_xlabel("fator máximo de crescimento $\\mu$"); ax.set_ylabel("perda final mediana $E/N$"); ax.legend(); ax.set_title("Quickprop: sensibilidade a $\\mu$ e $\\epsilon$ (300 épocas)"); salvar(fig, "f5_sensibilidade")
for eps_, med in tab.items(): print("epsilon=%.2f:" % eps_, {mu: round(float(m), 4) for mu, m in zip(mus, med)})

# ------------------------------------------------------------- deslocamento 0,1 da derivada (Fahlman) no XOR
P4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); T4 = np.array([[0], [1], [1], [0]], float)
def xor_taxa(ativ_saida, otim, H=2, **kw):
    ok = 0; ep = []
    for s in range(30):
        r = MLP([2, H, 1], ["sigmoide", ativ_saida], semente=s, init="pequena")
        res = otim(r.theta, lambda th: r.perda(th, P4, T4), lambda th: r.grad(th, P4, T4), tol_perda=1e-3, **kw)
        acerto = ((r.prever(P4, res.theta) > 0.5).astype(float) == T4).all(); ok += int(acerto)
        if acerto: ep.append(res.n_iter)
    return ok, (int(np.median(ep)) if ep else None)
print("XOR (2-2-1, 30 sementes), até 3000 épocas, parada em E<1e-3:")
print("  GD (eta=1)       sigmoide:", xor_taxa("sigmoide", O.gradiente_descendente, taxa=1.0, max_iter=3000))
print("  Quickprop(eps=.5) sigmoide:", xor_taxa("sigmoide", O.quickprop, epsilon=0.5, max_iter=3000))
print("  Quickprop(eps=.5) sigmoide_fahlman (+0.1):", xor_taxa("sigmoide_fahlman", O.quickprop, epsilon=0.5, max_iter=3000))
print("  GD (eta=1) sigmoide_fahlman (+0.1):", xor_taxa("sigmoide_fahlman", O.gradiente_descendente, taxa=1.0, max_iter=3000))
rF = MLP([2, 2, 1], ["sigmoide", "sigmoide_fahlman"], semente=0); from rnbook import grad_numerico, erro_relativo
print("gradcheck com o artifício +0.1: erro relativo =", "%.2e" % erro_relativo(rF.grad(rF.theta, P4, T4), grad_numerico(lambda th: rF.perda(th, P4, T4), rF.theta)), "(esperado: grande, pois não é o gradiente verdadeiro)")

# ------------------------------------------------------------- denominador pequeno/zero: tratamento
print("denominador zero: s=s_ant com mesmo sinal:", end=" ")
th = np.array([1.0]); out = O.quickprop(th, lambda t: float(0.5 * np.sum(t ** 2)), lambda t: np.array([0.0]) + 0 * t, max_iter=3)
print("rodou sem erro; theta final", out.theta)
print("OK cap05")
