"""Cap. 9 - Do treinamento clássico ao moderno: experimentos e figuras (sementes fixas; números do texto)."""
import sys, os, time, itertools, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
from rnbook import MLP, autodiff as AD, perdas as PL, modernos as MD, otimizadores as O
from rnbook.gradcheck import grad_numerico, erro_relativo
from rnbook.estilo import plt, COR, salvar
from rnbook.desenho import caixa, seta

# ---------------------------------------------------------------- 9.1-9.2 otimizadores de minilote (rede 1-8-1)
N = 100; rng = np.random.default_rng(3); X = rng.uniform(0, 1, (N, 1)); T = np.sin(2 * np.pi * X) + rng.normal(0, 0.15, (N, 1))
tune, aval = range(5), range(5, 15)
def preparar(semente):
    r = MLP([1, 8, 1], ["tanh", "identidade"], semente=semente)
    return r, (lambda th: r.perda(th, X, T) / N), (lambda th, idx: r.grad(th, X[idx], T[idx]))
def rodar(nome, semente, **kw):
    r, fun, gam = preparar(semente)
    f = dict(sgd=lambda **k: MD.sgd_momento(r.theta, gam, N, fun, nesterov=False, **k), nesterov=lambda **k: MD.sgd_momento(r.theta, gam, N, fun, nesterov=True, **k),
             adagrad=lambda **k: MD.adagrad(r.theta, gam, N, fun, **k), rmsprop=lambda **k: MD.rmsprop(r.theta, gam, N, fun, **k), adam=lambda **k: MD.adam(r.theta, gam, N, fun, **k))[nome]
    return f(tamanho_lote=10, epocas=100, semente=semente, **kw)
grades = {"sgd": [dict(taxa=a, momento=m) for a in (0.005, 0.01, 0.02, 0.05) for m in (0.5, 0.9)],
          "nesterov": [dict(taxa=a, momento=m) for a in (0.005, 0.01, 0.02, 0.05) for m in (0.5, 0.9)],
          "adagrad": [dict(taxa=a) for a in (0.05, 0.1, 0.3, 1.0)],
          "rmsprop": [dict(taxa=a) for a in (0.001, 0.003, 0.01, 0.03)],
          "adam": [dict(taxa=a) for a in (0.001, 0.003, 0.01, 0.03)]}
rotulos = {"sgd": "SGD + momento", "nesterov": "Nesterov", "adagrad": "AdaGrad", "rmsprop": "RMSProp", "adam": "Adam"}
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.8)); cores = dict(sgd=COR["gry"], nesterov=COR["red"], adagrad=COR["vio"], rmsprop=COR["amb"], adam=COR["grn"]); finais = {}
print("== minilote 10, 100 épocas (1000 atualizações), E/N; hiperparâmetros nas sementes 0-4; avaliação nas sementes 5-14")
for nome, grade in grades.items():
    sc = []
    for kw in grade:
        v = [rodar(nome, s, **kw).historico["perda"][-1] for s in tune]; sc.append(np.median([x if np.isfinite(x) else 1e9 for x in v]))
    kw = grade[int(np.argmin(sc))]; H = []; F = []; tt = []
    for s in aval:
        r = rodar(nome, s, **kw); h = np.array(r.historico["perda"]); h = np.where(np.isfinite(h), h, 1e9); H.append(h); F.append(h[-1]); tt.append(r.tempo)
    H = np.array(H); finais[nome] = F
    cruz = int(np.argmax(np.median(H, axis=0) < 0.03)) if (np.median(H, axis=0) < 0.03).any() else None
    print("%-9s melhor %-32s final mediana %.4f [%.4f; %.4f] | época em que a mediana cruza 0,03: %s | tempo %.2f s" % (rotulos[nome], kw, np.median(F), min(F), max(F), cruz, np.mean(tt)))
    ax[0].semilogy(np.median(H, axis=0), color=cores[nome], label=rotulos[nome])
ax[0].axhline(0.5 * 0.15 ** 2, color=COR["gry"], ls=":"); ax[0].set_xlabel("época"); ax[0].set_ylabel("perda $E/N$ (mediana de 10 sementes)"); ax[0].legend(fontsize=8); ax[0].set_title("Minilote de tamanho 10 (1–8–1)")
ax[1].boxplot([finais[k] for k in grades], tick_labels=[rotulos[k] for k in grades]); ax[1].set_yscale("log"); ax[1].set_ylabel("perda final"); ax[1].set_title("Dispersão entre sementes"); ax[1].tick_params(axis="x", labelsize=7)
salvar(fig, "f9_otimizadores")

# ---------------------------------------------------------------- invariância à escala da perda (Adam x SGD)
print("== escala da perda: gradiente multiplicado por c")
for c in (1.0, 100.0):
    out = []
    for nome, kw in (("sgd", dict(taxa=0.01, momento=0.9)), ("adam", dict(taxa=0.01))):
        r, fun, gam = preparar(0); gam_c = lambda th, idx, gam=gam: c * gam(th, idx)
        f = MD.sgd_momento if nome == "sgd" else MD.adam
        res = f(r.theta, gam_c, N, fun, tamanho_lote=10, epocas=100, semente=0, **kw)
        out.append("%s E/N final = %s" % (nome, "%.4f" % res.historico["perda"][-1] if np.isfinite(res.historico["perda"][-1]) else "não finito (%d épocas)" % res.n_iter))
    print("  c = %-5g %s" % (c, " | ".join(out)))

# ---------------------------------------------------------------- AdamW x Adam+L2: decaimento desacoplado
print("== decaimento: gradiente de dados constante -s (empurra para fora), s=(10; 0,1), lambda=0,1, eta=0,1, 20000 passos")
s_ = np.array([10.0, 0.1]); lam = 0.1; eta = 0.1
def trajet(modo):
    w = np.zeros(2); m = np.zeros(2); v = np.zeros(2); b1, b2 = 0.9, 0.999; hist = []
    for t in range(1, 20001):
        g = -s_ + (lam * w if modo == "l2" else 0)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        w = w - eta * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8) - (eta * lam * w if modo == "w" else 0)
        hist.append(w.copy())
    return np.array(hist)
Hl, Hw = trajet("l2"), trajet("w")
print("  Adam + L2 : w final = (%.2f; %.2f) | equilíbrio de g=0: s/lambda = (%.1f; %.1f)" % (Hl[-1, 0], Hl[-1, 1], *(s_ / lam)))
print("  AdamW     : w final = (%.2f; %.2f) | equilíbrio previsto 1/lambda = %.1f (independe de s)" % (Hw[-1, 0], Hw[-1, 1], 1 / lam))
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
for a, Hh, tit in ((ax[0], Hl, "Adam com penalidade L2 (acoplada)"), (ax[1], Hw, "AdamW (decaimento desacoplado)")):
    a.plot(Hh[:, 0], color=COR["red"], label="peso com gradiente grande ($s=10$)"); a.plot(Hh[:, 1], color=COR["ind"], label="peso com gradiente pequeno ($s=0{,}1$)"); a.set_title(tit); a.set_xlabel("passo")
ax[0].set_ylabel("valor do peso"); ax[0].legend(fontsize=7); ax[0].axhline(100, color=COR["gry"], ls=":", lw=0.8); ax[0].axhline(1, color=COR["gry"], ls=":", lw=0.8); ax[1].axhline(10, color=COR["gry"], ls=":", lw=0.8)
salvar(fig, "f9_adamw")

# ---------------------------------------------------------------- 9.4 grafos gerais com autodiff
print("== autodiff em grafos gerais (erro relativo contra diferenças finitas, h=1e-6)")
Xa = rng.normal(size=(6, 3)); Ta = rng.normal(size=(6, 2))
def perda_mlp(th):      # MLP 3-4-2
    W1, b1, W2, b2 = (AD.No(th[:12].reshape(4, 3)), AD.No(th[12:16]), AD.No(th[16:24].reshape(2, 4)), AD.No(th[24:26]))
    y = (AD.No(Xa) @ W1.T + b1).tanh() @ W2.T + b2; E = (y - Ta).quadrado().soma() * 0.5; E.retropropagar(); return E, [W1, b1, W2, b2]
def perda_res(th):      # conexão residual: h = tanh(W1 x + b1) + A x  (ramificação do x) ; y = W2 h + b2
    W1, b1, W2, b2, Aq = (AD.No(th[:12].reshape(4, 3)), AD.No(th[12:16]), AD.No(th[16:24].reshape(2, 4)), AD.No(th[24:26]), AD.No(th[26:38].reshape(4, 3)))
    x = AD.No(Xa); h = (x @ W1.T + b1).tanh() + x @ Aq.T; y = h @ W2.T + b2; E = (y - Ta).quadrado().soma() * 0.5; E.retropropagar(); return E, [W1, b1, W2, b2, Aq]
def perda_comp(th):     # compartilhamento: a mesma matriz W usada em duas camadas (4x4); y = V h2
    W, b, V = (AD.No(th[:16].reshape(4, 4)), AD.No(th[16:20]), AD.No(th[20:28].reshape(2, 4)))
    x = AD.No(Xa @ np.eye(3, 4)); h1 = (x @ W.T + b).tanh(); h2 = (h1 @ W.T + b).tanh(); y = h2 @ V.T; E = (y - Ta).quadrado().soma() * 0.5; E.retropropagar(); return E, [W, b, V]
for nome, f, P in (("MLP 3–4–2", perda_mlp, 26), ("bloco residual", perda_res, 38), ("pesos compartilhados", perda_comp, 28)):
    th = rng.normal(0, 0.5, P); E, ns = f(th); g = np.concatenate([n.grad.ravel() for n in ns]); gn = grad_numerico(lambda t: f(t)[0].valor, th, 1e-6)
    print("  %-22s P=%2d  E=%.4f  erro relativo = %.2e" % (nome, P, float(E.valor), erro_relativo(g, gn)))
# contribuições somadas no compartilhamento: gradiente de W = soma das duas camadas
th = rng.normal(0, 0.5, 28); W = AD.No(th[:16].reshape(4, 4)); b = AD.No(th[16:20]); x = AD.No(Xa @ np.eye(3, 4)); V = AD.No(th[20:28].reshape(2, 4))
h1 = (x @ W.T + b).tanh(); h2 = (h1 @ W.T + b).tanh(); y = h2 @ V.T; E = (y - Ta).quadrado().soma() * 0.5; E.retropropagar()
delta2 = (y.valor - Ta) @ V.valor * (1 - h2.valor ** 2); delta1 = (delta2 @ W.valor) * (1 - h1.valor ** 2)
soma = delta2.T @ h1.valor + delta1.T @ x.valor
print("  W compartilhada: ||grad_autodiff - (delta2' h1 + delta1' x)|| = %.2e" % np.linalg.norm(W.grad - soma))
# layer norm
Wl = rng.normal(size=(4, 3)); Xl = rng.normal(size=(5, 4))
def lnf(w):
    Wn = AD.No(w.reshape(4, 3)); z = AD.normalizacao_de_camada(AD.No(Xl) @ Wn); L = (z * np.arange(1, 4)).quadrado().soma(); L.retropropagar(); return L, Wn
L, Wn = lnf(Wl.ravel()); print("  normalização de camada: erro relativo = %.2e" % erro_relativo(Wn.grad.ravel(), grad_numerico(lambda w: lnf(w)[0].valor, Wl.ravel(), 1e-6)))
# dropout invertido
p = 0.3; M = (np.random.default_rng(0).random((200000, 5)) > p) / (1 - p); h = np.array([1.0, -2.0, 0.5, 3.0, 0.0])
print("  dropout invertido (p=0,3): média do vetor mascarado = %s (esperado %s)" % (np.round((M * h).mean(0), 3), h))
mask = (np.random.default_rng(1).random((6, 4)) > p) / (1 - p)
def perda_drop(th):
    W1, W2 = AD.No(th[:12].reshape(4, 3)), AD.No(th[12:20].reshape(2, 4)); h = (AD.No(Xa) @ W1.T).tanh() * mask; y = h @ W2.T; E = (y - Ta).quadrado().soma() * 0.5; E.retropropagar(); return E, [W1, W2]
th = rng.normal(0, 0.5, 20); E, ns = perda_drop(th); g = np.concatenate([n.grad.ravel() for n in ns]); print("  dropout com máscara fixa: erro relativo = %.2e" % erro_relativo(g, grad_numerico(lambda t: perda_drop(t)[0].valor, th, 1e-6)))
# comparação com JAX (opcional)
try:
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    def f_jax(th):
        W1, b1, W2, b2, Aq = th[:12].reshape(4, 3), th[12:16], th[16:24].reshape(2, 4), th[24:26], th[26:38].reshape(4, 3)
        h = jnp.tanh(Xa @ W1.T + b1) + Xa @ Aq.T; y = h @ W2.T + b2; return 0.5 * jnp.sum((y - Ta) ** 2)
    th = rng.normal(0, 0.5, 38); E, ns = perda_res(th); g = np.concatenate([n.grad.ravel() for n in ns]); gj = np.asarray(jax.grad(f_jax)(th))
    print("  JAX %s: erro relativo (nosso autodiff x jax.grad, bloco residual) = %.2e" % (jax.__version__, erro_relativo(g, gj)))
    v = rng.normal(size=38); hv_jax = np.asarray(jax.jvp(jax.grad(f_jax), (th,), (v,))[1]); hv_dif = MD.hvp_dif_finitas(lambda t: np.asarray(jax.grad(f_jax)(t)), th, v)
    print("  JAX: produto Hessiana–vetor exato (jvp de grad) x diferenças finitas: erro relativo = %.2e" % (np.linalg.norm(hv_jax - hv_dif) / np.linalg.norm(hv_jax)))
except Exception as e:
    print("  JAX indisponível:", type(e).__name__)

# ---------------------------------------------------------------- 9.5 perdas estáveis
print("== perdas estáveis")
Z = np.array([[1000.0, 0.0, -1000.0]]); T1 = np.array([[0.0, 1.0, 0.0]])
print("  entropia cruzada com z=(1000;0;-1000), classe 2: ingênua =", PL.ce_ingenua(Z, T1), "| estável =", PL.ce_logits(Z, T1)[0], "(esperado: logsumexp - z_2 = 1000)")
Zb = np.array([[800.0, -800.0]]); Tb = np.array([[0.0, 1.0]]); print("  binária z=(800;-800), t=(0;1): ingênua =", PL.bce_ingenua(Zb, Tb), "| estável =", PL.bce_logits(Zb, Tb)[0], "(esperado 1600)")
zs = np.linspace(0, 60, 241); naive = []; est = []
for z in zs:
    with np.errstate(all="ignore"): naive.append(-np.log(1 - 1 / (1 + np.exp(-z))))     # t = 0: -log(1 - sigmoide(z))
    est.append(np.logaddexp(0, z))
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.5)); ax[0].plot(zs, est, color=COR["grn"], lw=2, label="estável: $\\log(1+e^{z})$ via logaddexp"); naive = np.array(naive); first_inf = zs[np.argmax(~np.isfinite(naive))] if (~np.isfinite(naive)).any() else None; naive[~np.isfinite(naive)] = np.nan
ax[0].plot(zs, naive, color=COR["red"], ls="--", label="ingênua: $-\\log(1-\\sigma(z))$"); ax[0].set_xlabel("logit $z$ (rótulo $t=0$)"); ax[0].set_ylabel("perda"); ax[0].legend(fontsize=7); ax[0].set_title("Perda binária de um exemplo")
print("  perda binária ingênua (t=0) deixa de ser finita a partir de z = %s" % first_inf)
zz = np.linspace(0, 800, 400); ing = [PL.ce_ingenua(np.array([[z, 0.0]]), np.array([[0.0, 1.0]])) if z < 700 else np.nan for z in zz]; ax[1].plot(zz, [PL.ce_logits(np.array([[z, 0.0]]), np.array([[0.0, 1.0]]))[0] for z in zz], color=COR["grn"], lw=2, label="estável (logsumexp)"); ax[1].plot(zz, ing, color=COR["red"], ls="--", label="ingênua"); ax[1].set_xlabel("logit da classe errada, $z_1$ ($z_2=0$, $t=$ classe 2)"); ax[1].legend(fontsize=7); ax[1].set_title("Entropia cruzada com 2 classes")
salvar(fig, "f9_estavel")
# multirrótulo
rngm = np.random.default_rng(5); Zm = rngm.normal(0, 2, (50, 4)); Tm = (rngm.random((50, 4)) > 0.6).astype(float)
gn = grad_numerico(lambda z: PL.bce_logits(z.reshape(50, 4), Tm)[0], Zm.ravel(), 1e-6); print("  multirrótulo: erro relativo do gradiente (sigm(z)-t)/N = %.2e" % erro_relativo(PL.bce_logits(Zm, Tm)[1].ravel(), gn))
Tc = np.eye(4)[rngm.integers(0, 4, 50)]; gn = grad_numerico(lambda z: PL.ce_logits(z.reshape(50, 4), Tc)[0], Zm.ravel(), 1e-6); print("  multiclasse:  erro relativo do gradiente (softmax(z)-t)/N = %.2e" % erro_relativo(PL.ce_logits(Zm, Tc)[1].ravel(), gn))
z0 = np.array([[2.0, 1.0, -1.0, 0.5]]); print("  z = %s: soma das sigmoides = %.4f; soma da softmax = %.4f" % (z0.ravel(), (1 / (1 + np.exp(-z0))).sum(), PL.softmax_estavel(z0).sum()))

# ---------------------------------------------------------------- 9.7 produtos Hessiana-vetor, potência e LM sem matriz
print("== Hessiana–vetor")
r = MLP([1, 8, 1], ["tanh", "identidade"], semente=0); th = r.theta.copy(); gfun = lambda t: r.grad(t, X, T)
Hc = np.array([(gfun(th + 1e-5 * e) - gfun(th - 1e-5 * e)) / 2e-5 for e in np.eye(r.n_params)]); Hc = (Hc + Hc.T) / 2
v = np.random.default_rng(1).normal(size=r.n_params); print("  erro relativo de H v (diferenças finitas do gradiente) contra a Hessiana numérica completa: %.2e" % (np.linalg.norm(MD.hvp_dif_finitas(gfun, th, v) - Hc @ v) / np.linalg.norm(Hc @ v)))
w = np.random.default_rng(2).normal(size=r.n_params)
for k in range(300): w = MD.hvp_dif_finitas(gfun, th, w); w /= np.linalg.norm(w)
lam_max = w @ MD.hvp_dif_finitas(gfun, th, w); ev = np.linalg.eigvalsh(Hc); print("  potência (300 produtos H v): lambda = %.4f | autovalor de maior módulo da Hessiana numérica completa: %.4f" % (lam_max, ev[np.argmax(np.abs(ev))]))
res = lambda t: r.residuos(t, X, T); jvp = lambda t, v: MD.jvp_dif_finitas(res, t, v); jtvp = lambda t, u: r.grad(t, X, r.prever(X, t) - u.reshape(N, 1))
lim = 1.1 * 0.5 * N * 0.15 ** 2
def iter_ate_lim(f, **kw):
    hist = []
    for it in range(1, 61):
        th_i, n_it, cg = f(it)
        E = 0.5 * float(res(th_i) @ res(th_i)); hist.append((E, cg))
        if E < lim: return it, E, cg, hist
    return None, E, cg, hist
it_cg, E_cg, cg_tot, hcg = iter_ate_lim(lambda it: MD.lm_sem_matriz(th, res, jvp, jtvp, mu0=1e-3, max_iter=it, tol_cg=1e-6, max_cg=100))
jac = lambda t: r.jacobiana(t, X); dir_ = O.levenberg_marquardt(th, res, jac, max_iter=60, tol_grad=0, tol_perda=0)
Ed = np.array(dir_.historico["perda"]); it_dir = int(np.argmax(Ed < lim)) if (Ed < lim).any() else None
print("  limiar E < %.4f: LM direto (Cholesky/solve) = %s iterações | LM com gradiente conjugado = %s iterações (%s produtos J^T J v no total); E final CG-LM = %.4f" % (lim, it_dir, it_cg, cg_tot, E_cg))
fig, ax = plt.subplots(figsize=(5.8, 3.6)); ax.semilogy(Ed[:25], "o-", ms=3, color=COR["ind"], label="LM direto"); ax.semilogy([h[0] for h in hcg][:25], "s-", ms=3, color=COR["grn"], label="LM com gradiente conjugado (sem formar $J^TJ$)"); ax.axhline(lim, color=COR["gry"], ls=":"); ax.set_xlabel("iteração"); ax.set_ylabel("$E$"); ax.legend(fontsize=7); ax.set_title("Mesma rede 1–8–1, mesma inicialização")
salvar(fig, "f9_cg")

# ---------------------------------------------------------------- 9.8 diagnóstico
print("== diagnóstico (rede com 10 camadas ocultas de 20 neurônios, 64 amostras aleatórias, semente 0)")
Xd = np.random.default_rng(0).normal(size=(64, 20)); Td = np.random.default_rng(1).normal(size=(64, 1)); linhas = {}
for nome, ativ, init in (("sigmoide, Xavier", "sigmoide", "xavier"), ("tanh, Xavier", "tanh", "xavier"), ("tanh, pesos pequenos (σ=0,5)", "tanh", "pequena")):
    rd = MLP([20] + [20] * 10 + [1], [ativ] * 10 + ["identidade"], semente=0, init=init); d = MD.diagnostico(rd, rd.theta, Xd, Td)
    gl = d["norma_grad_por_camada"]; linhas[nome] = gl
    print("  %-30s ||grad|| camada 1 = %.2e | camada 11 = %.2e | razão = %.1e | saturação média nas ocultas = %.2f | finito: %s" % (nome, gl[0], gl[-1], gl[-1] / gl[0], np.mean(d["saturacao_por_camada"]), d["finito"]))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for (nome, gl), c in zip(linhas.items(), (COR["red"], COR["ind"], COR["amb"])): ax.semilogy(range(1, 12), gl, "o-", ms=4, color=c, label=nome)
ax.set_xlabel("camada (1 = mais próxima da entrada)"); ax.set_ylabel("$\\|\\partial E/\\partial W^{(l)}\\|$"); ax.legend(fontsize=7); ax.set_title("Diagnóstico: norma do gradiente por camada"); salvar(fig, "f9_diagnostico")
# razão de atualização
r8, fun, gam = preparar(0); passo = []
res_ad = MD.adam(r8.theta, gam, N, fun, taxa=0.01, tamanho_lote=10, epocas=1, semente=0); passo = res_ad.theta - r8.theta
dg = MD.diagnostico(r8, res_ad.theta, X, T, passo=passo); print("  razão ||passo||/||pesos|| por camada após 1 época de Adam (taxa 0,01): %s" % np.round(dg["razao_atualizacao_por_camada"], 4))
# diagrama do grafo
fig, ax = plt.subplots(figsize=(8.2, 2.8)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
caixa(ax, 0.2, 1.2, 0.9, 0.6, "$x$", fs=11); caixa(ax, 2.0, 2.0, 1.6, 0.6, "$\\tanh(W_1x+b_1)$", fs=9); caixa(ax, 2.0, 0.4, 1.6, 0.6, "$Ax$", fs=10); caixa(ax, 4.5, 1.2, 0.9, 0.6, "$+$", fs=12, bold=True); caixa(ax, 6.2, 1.2, 1.4, 0.6, "$W_2h+b_2$", fs=9); caixa(ax, 8.3, 1.2, 1.0, 0.6, "$E$", fs=11)
for p, q in (((1.1, 1.6), (2.0, 2.3)), ((1.1, 1.4), (2.0, 0.7)), ((3.6, 2.3), (4.5, 1.7)), ((3.6, 0.7), (4.5, 1.3)), ((5.4, 1.5), (6.2, 1.5)), ((7.6, 1.5), (8.3, 1.5))): seta(ax, p, q)
for p, q in (((8.3, 1.2), (7.6, 1.2)), ((6.2, 1.2), (5.4, 1.2))): seta(ax, p, q, cor=COR["red"], ls="--")
ax.text(7.1, 2.3, "na retropropagação, $\\partial E/\\partial x$ soma as duas contribuições (ramos)", fontsize=8, color=COR["red"], ha="center")
salvar(fig, "f9_grafo")
print("OK cap09")
