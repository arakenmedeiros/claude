"""Cap. 4 (parte A): exemplo numérico 2-2-1, verificação de gradientes, escalar x vetorizado, XOR, gradientes que desaparecem."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rnbook import MLP, grad_numerico, erro_relativo
from rnbook.backprop_escalar import backprop_escalar
from rnbook.ativacoes import sigmoide, d_sigmoide
from rnbook.estilo import plt, COR, salvar
from rnbook.desenho import caixa, seta, diagrama_rede

# ------------------------------------------------ exemplo 2-2-1 (sigmoide em tudo), uma amostra
W1 = np.array([[0.1, 0.3], [-0.2, 0.4]]); b1 = np.array([0.1, -0.1]); W2 = np.array([[0.5, -0.6]]); b2 = np.array([0.2])
x = np.array([1.0, 0.5]); t = np.array([1.0])
z1 = W1 @ x + b1; a1 = sigmoide(z1); z2 = W2 @ a1 + b2; y = sigmoide(z2); r = y - t; E = 0.5 * float(r @ r)
d2 = r * y * (1 - y); gW2 = np.outer(d2, a1); gb2 = d2
d1 = (W2.T @ d2) * a1 * (1 - a1); gW1 = np.outer(d1, x); gb1 = d1
np.set_printoptions(precision=5, suppress=True)
print("z1", z1, "a1", a1, "z2", z2, "y", y, "r", r, "E %.5f" % E)
print("delta2", d2, "gW2", gW2, "gb2", gb2); print("delta1", d1, "gW1\n", gW1, "gb1", gb1)
rede = MLP([2, 2, 1], ["sigmoide", "sigmoide"], semente=0)
theta = np.concatenate([W1.ravel(), b1, W2.ravel(), b2]); rede.theta = theta.copy()
gv = rede.grad(theta, x[None, :], t[None, :]); gesp = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
f = lambda th: rede.perda(th, x[None, :], t[None, :])
gn = grad_numerico(f, theta)
print("à mão x vetorizado: %.2e | vetorizado x numérico: %.2e (erro relativo)" % (np.abs(gv - gesp).max(), erro_relativo(gv, gn)))
E_, dW, db = backprop_escalar([W1, W2], [b1, b2], ["sigmoide", "sigmoide"], x, t)
print("escalar: E=%.5f; dif. máx. vs vetorizado %.2e" % (E_, np.abs(np.concatenate([dW[0].ravel(), db[0], dW[1].ravel(), db[1]]) - gv).max()))
# passo de atualização com eta=0.5
eta = 0.5; th2 = theta - eta * gv; print("E após 1 passo (eta=0.5): %.5f (antes %.5f)" % (rede.perda(th2, x[None, :], t[None, :]), E))

# ------------------------------------------------ verificação por diferenças finitas: efeito de h
rng = np.random.default_rng(3); X = rng.normal(size=(8, 3)); T = rng.normal(size=(8, 2))
rd = MLP([3, 5, 2], ["tanh", "identidade"], semente=2, init="pequena")
ga = rd.grad(rd.theta, X, T); fe = lambda th: rd.perda(th, X, T)
hs = 10.0 ** -np.arange(1, 14); errs = [erro_relativo(ga, grad_numerico(fe, rd.theta, h)) for h in hs]
print("gradcheck: erro relativo por h:", {f"{h:.0e}": f"{e:.1e}" for h, e in zip(hs, errs)})
print("melhor h:", "%.0e" % hs[int(np.argmin(errs))], "erro %.1e" % min(errs))
# ReLU em ponto de não diferenciabilidade
rr = MLP([1, 1], ["relu"], semente=0); rr.theta = np.array([1.0, 0.0]); Xr = np.array([[0.0]]); Tr = np.array([[1.0]])
print("ReLU em z=0: grad analítico", rr.grad(rr.theta, Xr, Tr), "| numérico", grad_numerico(lambda th: rr.perda(th, Xr, Tr), rr.theta))
fig, ax = plt.subplots(figsize=(5.8, 3.6)); ax.loglog(hs, errs, "o-", color=COR["ind"]); ax.set_xlabel("passo $h$ da diferença central"); ax.set_ylabel("erro relativo do gradiente")
ax.axhline(1e-7, color=COR["grn"], ls="--", lw=1); ax.text(1e-13, 2e-7, "limite usual de aceitação ($10^{-7}$)", color=COR["grn"], fontsize=8); ax.set_title("Verificação de gradiente: escolha de $h$"); salvar(fig, "f4_gradcheck")

# ------------------------------------------------ escalar x vetorizado: tempo
for N, camadas in [(200, [2, 20, 1])]:
    rr_ = MLP(camadas, ["tanh", "identidade"], semente=1); Xs = np.random.default_rng(0).normal(size=(N, 2)); Ts = np.random.default_rng(1).normal(size=(N, 1))
    Wl = [w.copy() for w, _ in rr_.desempacotar(rr_.theta)]; bl = [b.copy() for _, b in rr_.desempacotar(rr_.theta)]
    t0 = time.perf_counter()
    for _ in range(3): g_esc = sum(np.concatenate([a.ravel() for pair in zip(*backprop_escalar(Wl, bl, ["tanh", "identidade"], Xs[i], Ts[i])[1:]) for a in pair]) for i in range(N)) if False else None
    # soma dos gradientes por amostra (escalar), 3 repetições
    t0 = time.perf_counter()
    for _ in range(3):
        tot = np.zeros(rr_.n_params)
        for i in range(N):
            E_i, dWi, dbi = backprop_escalar(Wl, bl, ["tanh", "identidade"], Xs[i], Ts[i])
            tot += np.concatenate([np.concatenate([dWi[l].ravel(), dbi[l]]) for l in range(2)])
    t_esc = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(300): gv2 = rr_.grad(rr_.theta, Xs, Ts)
    t_vet = (time.perf_counter() - t0) / 300
    print("N=%d, rede %s: escalar %.4f s | vetorizado %.6f s | razão %.0fx | diferença máx. %.2e" % (N, camadas, t_esc, t_vet, t_esc / t_vet, np.abs(tot - gv2).max()))

# ------------------------------------------------ gradientes que desaparecem: produto de derivadas
L = np.arange(1, 21); print("limite superior do produto das derivadas (sigmoide): 0.25^L para L=5,10,20:", 0.25 ** 5, 0.25 ** 10, 0.25 ** 20)
rngv = np.random.default_rng(0); normas = {}
for nome, ativ in [("sigmoide", "sigmoide"), ("tanh", "tanh"), ("relu", "relu")]:
    r_ = MLP([4] + [20] * 10 + [1], [ativ] * 10 + ["identidade"], semente=5, init="he" if ativ == "relu" else "xavier")
    Xv = rngv.normal(size=(64, 4)); Tv = rngv.normal(size=(64, 1)); g = r_.grad(r_.theta, Xv, Tv)
    sl = r_.fatias(); normas[nome] = [np.linalg.norm(g[sl[l][0][0]:sl[l][0][1]]) for l in range(r_.L)]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for k, (nome, v) in enumerate(normas.items()): ax.semilogy(range(1, 12), v, "o-", label=nome)
ax.set_xlabel("camada $l$ (1 = mais próxima da entrada)"); ax.set_ylabel("$\\|\\partial E/\\partial W^{(l)}\\|$"); ax.legend(); ax.set_title("Rede com 10 camadas ocultas: norma do gradiente por camada")
salvar(fig, "f4_gradiente_camadas")
print("norma do gradiente na camada 1 / camada 11 :", {k: "%.2e / %.2e" % (v[0], v[-1]) for k, v in normas.items()})
