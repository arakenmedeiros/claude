"""Cap. 6 - RPROP e variantes: experimentos e figuras (semente fixa; números citados no texto)."""
import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
from rnbook import MLP
from rnbook import otimizadores as O
from rnbook.estilo import plt, COR, salvar
from rnbook.desenho import caixa, seta

# ------------------------------------------------------------ 6.1 regra de Manhattan x RPROP em uma calha
H = np.diag([1.0, 10.0]); fq = lambda v: 0.5 * v @ H @ v; gq = lambda v: H @ v
def manhattan(v0, passo, n):
    v = np.array(v0, float); tr = [v.copy()]
    for _ in range(n): v = v - passo * np.sign(gq(v)); tr.append(v.copy())
    return np.array(tr)
tr_m = manhattan([2.5, 1.2], 0.1, 80)
res_r = O.rprop(np.array([2.5, 1.2]), lambda v: fq(v), lambda v: gq(v), variante="irprop-", delta0=0.1, max_iter=80, tol_grad=0)
# reconstruir trajetória do RPROP
tr_r = [np.array([2.5, 1.2])]
v = tr_r[0].copy(); delta = np.full(2, 0.1); g_ant = np.zeros(2)
for _ in range(80):
    g = gq(v); prod = g_ant * g
    delta = np.where(prod > 0, np.minimum(delta * 1.2, 50), delta); delta = np.where(prod < 0, np.maximum(delta * 0.5, 1e-6), delta); g = np.where(prod < 0, 0.0, g)
    v = v - np.sign(g) * delta; g_ant = g; tr_r.append(v.copy())
tr_r = np.array(tr_r)
print("Manhattan (passo 0.1 fixo): distância à origem após 80 passos = %.4f; f = %.4f | menor distância ao longo da trajetória = %.4f" % (np.linalg.norm(tr_m[-1]), fq(tr_m[-1]), min(np.linalg.norm(t) for t in tr_m)))
print("iRPROP- (delta0=0.1, 80 passos): distância final = %.2e; f = %.2e" % (np.linalg.norm(tr_r[-1]), fq(tr_r[-1])))
gx, gy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-1.6, 1.6, 200))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9)); ax[0].contour(gx, gy, 0.5 * (gx ** 2 + 10 * gy ** 2), levels=[0.5, 2, 5, 10, 16, 25], colors="#CBD5E1", linewidths=0.8)
ax[0].plot(tr_m[:, 0], tr_m[:, 1], "o-", ms=2.5, lw=1, color=COR["amb"], label="Manhattan (passo fixo 0,1)"); ax[0].plot(tr_r[:, 0], tr_r[:, 1], "o-", ms=2.5, lw=1, color=COR["ind"], label="iRPROP− (passos adaptativos)")
ax[0].set_xlim(-3, 3); ax[0].set_ylim(-1.6, 1.6); ax[0].legend(fontsize=8); ax[0].set_title("Só o sinal do gradiente: passo fixo × adaptativo"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[1].semilogy([np.linalg.norm(t) + 1e-16 for t in tr_m], color=COR["amb"], label="Manhattan"); ax[1].semilogy([np.linalg.norm(t) + 1e-16 for t in tr_r], color=COR["ind"], label="iRPROP−"); ax[1].set_xlabel("iteração"); ax[1].set_ylabel("distância à origem"); ax[1].legend(); ax[1].set_title("O passo fixo oscila ao redor do mínimo")
salvar(fig, "f6_manhattan")

# ------------------------------------------------------------ figura das regras (4 variantes)
fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off"); ax.grid(False)
cols = [("$g_{\\mathrm{ant}}\\,g>0$\n(mesmo sinal)", COR["grn"]), ("$g_{\\mathrm{ant}}\\,g<0$\n(mudou de sinal)", COR["red"]), ("$g_{\\mathrm{ant}}\\,g=0$", COR["gry"])]
for j, (t_, c) in enumerate(cols): caixa(ax, 1.9 + j * 3.0, 3.35, 2.8, 0.75, t_, cor=c, fc="white", fs=9, bold=True)
linhas = [("RPROP+", ["$\\Delta\\leftarrow\\min(\\eta^+\\Delta,\\Delta_{\\max})$\n$\\Delta w=-\\mathrm{sinal}(g)\\Delta$", "$\\Delta\\leftarrow\\max(\\eta^-\\Delta,\\Delta_{\\min})$\nretrocede: $\\Delta w=-\\Delta w_{\\mathrm{ant}}$; $g\\leftarrow0$", "$\\Delta w=-\\mathrm{sinal}(g)\\Delta$"]),
          ("RPROP−", ["$\\Delta\\leftarrow\\min(\\eta^+\\Delta,\\Delta_{\\max})$\n$\\Delta w=-\\mathrm{sinal}(g)\\Delta$", "$\\Delta\\leftarrow\\max(\\eta^-\\Delta,\\Delta_{\\min})$\nsem retrocesso: $\\Delta w=-\\mathrm{sinal}(g)\\Delta$", "$\\Delta w=-\\mathrm{sinal}(g)\\Delta$"]),
          ("iRPROP+", ["$\\Delta\\leftarrow\\min(\\eta^+\\Delta,\\Delta_{\\max})$\n$\\Delta w=-\\mathrm{sinal}(g)\\Delta$", "$\\Delta\\leftarrow\\max(\\eta^-\\Delta,\\Delta_{\\min})$; $g\\leftarrow0$\nse $E_t>E_{t-1}$: retrocede; senão $\\Delta w=0$", "$\\Delta w=-\\mathrm{sinal}(g)\\Delta$"]),
          ("iRPROP−", ["$\\Delta\\leftarrow\\min(\\eta^+\\Delta,\\Delta_{\\max})$\n$\\Delta w=-\\mathrm{sinal}(g)\\Delta$", "$\\Delta\\leftarrow\\max(\\eta^-\\Delta,\\Delta_{\\min})$; $g\\leftarrow0$\n$\\Rightarrow\\Delta w=0$ (sem retrocesso)", "$\\Delta w=-\\mathrm{sinal}(g)\\Delta$"])]
for i, (nm, cel) in enumerate(linhas):
    y0 = 2.55 - i * 0.82; caixa(ax, 0.05, y0, 1.75, 0.7, nm, cor=COR["ind"], fc="#EEF2FF", fs=10, bold=True)
    for j, t_ in enumerate(cel): caixa(ax, 1.9 + j * 3.0, y0, 2.8, 0.7, t_, cor=cols[j][1], fc="white", fs=7.4, lw=1)
salvar(fig, "f6_regras")

# ------------------------------------------------------------ rede 1-8-1 senoide: variantes (10 sementes)
N = 100; rng = np.random.default_rng(3); X = rng.uniform(0, 1, (N, 1)); T = np.sin(2 * np.pi * X) + rng.normal(0, 0.15, (N, 1))
def roda(var, semente, it=300, **kw):
    r = MLP([1, 8, 1], ["tanh", "identidade"], semente=semente); t0 = time.perf_counter()
    res = O.rprop(r.theta, lambda th: r.perda(th, X, T) / N, lambda th: r.grad(th, X, T) / N, variante=var, max_iter=it, tol_grad=0, **kw); return res, r, time.perf_counter() - t0
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.7)); NOME = {"rprop+": "RPROP+", "rprop-": "RPROP−", "irprop+": "iRPROP+", "irprop-": "iRPROP−"}; cores = {"rprop+": COR["amb"], "rprop-": COR["red"], "irprop+": COR["ind"], "irprop-": COR["grn"]}
fin = {}
for var in O.VARIANTES_RPROP:
    H_ = []; F = []; tt = []
    for s in range(10):
        res, r, dt = roda(var, s); H_.append(res.historico["perda"]); F.append(res.historico["perda"][-1]); tt.append(dt)
    H_ = np.array(H_); fin[var] = F
    ax[0].semilogy(np.median(H_, axis=0), color=cores[var], label=NOME[var], lw=1.8)
    print("%-8s perda final mediana %.4f [mín %.4f, máx %.4f] | época em que a mediana cruza 0.03: %s | tempo %.3f s" % (var, np.median(F), min(F), max(F), (int(np.argmax(np.median(H_, axis=0) < 0.03)) if (np.median(H_, axis=0) < 0.03).any() else None), np.mean(tt)))
ax[0].axhline(0.5 * 0.15 ** 2, color=COR["gry"], ls=":"); ax[0].legend(fontsize=8); ax[0].set_xlabel("época"); ax[0].set_ylabel("perda $E/N$ (mediana de 10 sementes)"); ax[0].set_title("Quatro variantes, rede 1–8–1")
ax[1].boxplot([fin[v] for v in O.VARIANTES_RPROP], tick_labels=[NOME[v] for v in O.VARIANTES_RPROP]); ax[1].set_yscale("log"); ax[1].set_ylabel("perda final $E/N$ (300 épocas)"); ax[1].set_title("Dispersão entre sementes")
salvar(fig, "f6_variantes")

# ------------------------------------------------------------ passos individuais e mudanças de sinal (iRPROP-)
r = MLP([1, 8, 1], ["tanh", "identidade"], semente=0); th = r.theta.copy(); delta = np.full_like(th, 0.1); g_ant = np.zeros_like(th); D = []; trocas = []; Ltr = []
for k in range(300):
    g = r.grad(th, X, T) / N; prod = g_ant * g; trocas.append(int((prod < 0).sum()))
    delta = np.where(prod > 0, np.minimum(delta * 1.2, 50.0), delta); delta = np.where(prod < 0, np.maximum(delta * 0.5, 1e-6), delta); g = np.where(prod < 0, 0.0, g)
    th = th - np.sign(g) * delta; g_ant = g; D.append(delta.copy()); Ltr.append(r.perda(th, X, T) / N)
D = np.array(D); print("iRPROP-: passos individuais após 300 épocas: mín %.2e mediana %.2e máx %.2e | trocas de sinal por época (média das últimas 100): %.2f de %d pesos" % (D[-1].min(), np.median(D[-1]), D[-1].max(), np.mean(trocas[-100:]), th.size))
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6))
for j in range(th.size): ax[0].semilogy(D[:, j], lw=0.8, alpha=0.6, color=COR["ind"])
ax[0].set_xlabel("época"); ax[0].set_ylabel("passo individual $\\Delta_i$"); ax[0].set_title("iRPROP−: 25 passos individuais")
ax[1].plot(trocas, color=COR["red"], lw=1); ax[1].set_xlabel("época"); ax[1].set_ylabel("nº de pesos com mudança de sinal"); ax[1].set_title("Mudanças de sinal do gradiente por época")
salvar(fig, "f6_passos")

# ------------------------------------------------------------ frequência de retrocessos (RPROP+ x iRPROP+)
def conta_retro(var, semente=0, it=300):
    r = MLP([1, 8, 1], ["tanh", "identidade"], semente=semente); th = r.theta.copy(); delta = np.full_like(th, 0.1); g_ant = np.zeros_like(th); dw_ant = np.zeros_like(th); E_ant = np.inf; ret = 0; sc = 0
    for k in range(it):
        g = r.grad(th, X, T) / N; E = r.perda(th, X, T) / N; prod = g_ant * g; pos, neg = prod > 0, prod < 0
        delta = np.where(pos, np.minimum(delta * 1.2, 50.0), delta); delta = np.where(neg, np.maximum(delta * 0.5, 1e-6), delta)
        sc += int(neg.sum())
        if var == "rprop+": dw = np.where(neg, -dw_ant, -np.sign(g) * delta); ret += int(neg.sum())
        else:
            if E > E_ant: dw = np.where(neg, -dw_ant, -np.sign(g) * delta); ret += int(neg.sum())
            else: dw = np.where(neg, 0.0, -np.sign(g) * delta)
        g = np.where(neg, 0.0, g); th = th + dw; g_ant, dw_ant, E_ant = g, dw, E
    return ret, sc
for v in ("rprop+", "irprop+"): print("%s: mudanças de sinal %d | retrocessos efetivos %d (semente 0, 300 épocas, 25 pesos)" % (v, conta_retro(v)[1], conta_retro(v)[0]))

# ------------------------------------------------------------ 6.4 parada antecipada (rede grande, poucos dados)
def exp_pa(seed, ntr=30, nva=30, H=100, noise=0.4, its=3000, pac=None):
    rng = np.random.default_rng(100 + seed); n = ntr + nva + 300; xs = rng.uniform(0, 1, (n, 1)); ts = np.sin(2 * np.pi * xs) + rng.normal(0, noise, (n, 1))
    xtr, ttr = xs[:ntr], ts[:ntr]; xva, tva = xs[ntr:ntr + nva], ts[ntr:ntr + nva]; xte, tte = xs[ntr + nva:], ts[ntr + nva:]
    r = MLP([1, H, 1], ["tanh", "identidade"], semente=seed, init="he")
    f_tr = lambda th: r.perda(th, xtr, ttr) / len(xtr); f_va = lambda th: r.perda(th, xva, tva) / len(xva)
    res = O.rprop(r.theta, f_tr, lambda th: r.grad(th, xtr, ttr) / len(xtr), variante="irprop-", max_iter=its, tol_grad=0, val=f_va, paciencia=pac)
    rt = lambda th: np.sqrt(np.mean((r.prever(xte, th) - tte) ** 2))
    return r, res, (xtr, ttr, xva, tva), rt
out = []
for seed in range(10):
    r, res, _, rt = exp_pa(seed); hva = np.array(res.historico["perda_val"]); k = int(np.argmin(hva))
    r2, res2, _, rt2 = exp_pa(seed, pac=100)
    out.append((k + 1, rt(res.theta_melhor), rt(res.theta), res2.n_iter, rt2(res2.theta_melhor)))
o = np.array(out)
print("Parada antecipada (rede 1-100-1, 30 treino, 30 validação, 300 teste, ruído 0.4, iRPROP-, 3000 épocas, 10 sementes):")
print("  melhor época (mediana) %d [mín %d, máx %d]" % (np.median(o[:, 0]), o[:, 0].min(), o[:, 0].max()))
print("  RMSE de teste: melhor época %.3f | após 3000 épocas %.3f | melhor época vence em %d de 10 | ruído verdadeiro 0.40" % (o[:, 1].mean(), o[:, 2].mean(), (o[:, 1] < o[:, 2]).sum()))
print("  com paciência 100: parou (mediana) na época %d [mín %d, máx %d]; RMSE de teste médio %.3f" % (np.median(o[:, 3]), o[:, 3].min(), o[:, 3].max(), o[:, 4].mean()))
r, res, (xtr, ttr, xva, tva), rt = exp_pa(0); htr, hva = np.array(res.historico["perda"]), np.array(res.historico["perda_val"]); kbest = int(np.argmin(hva))
print("  semente 0: melhor época %d (val %.4f, treino %.4f); após 3000 épocas: treino %.4f, validação %.4f; RMSE teste %.3f vs %.3f" % (kbest + 1, hva[kbest], htr[kbest], htr[-1], hva[-1], rt(res.theta_melhor), rt(res.theta)))
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.7)); ax[0].semilogy(htr, color=COR["ind"], label="treino"); ax[0].semilogy(hva, color=COR["amb"], label="validação"); ax[0].axvline(kbest, color=COR["red"], ls="--")
ax[0].text(kbest + 60, hva.max() * 0.6, f"melhor época = {kbest+1}", color=COR["red"], fontsize=9); ax[0].set_xlabel("época"); ax[0].set_ylabel("perda $E/N$"); ax[0].legend(); ax[0].set_title("Parada antecipada: a validação sobe depois do mínimo")
xx = np.linspace(0, 1, 300)[:, None]; ax[1].scatter(xtr, ttr, s=18, color=COR["ind"], label="treino"); ax[1].scatter(xva, tva, s=18, color=COR["amb"], label="validação"); ax[1].plot(xx, np.sin(2 * np.pi * xx), color=COR["gry"], ls=":", label="verdadeira")
ax[1].plot(xx, r.prever(xx, res.theta_melhor), color=COR["grn"], lw=2, label=f"melhor época ({kbest+1})"); ax[1].plot(xx, r.prever(xx, res.theta), color=COR["red"], lw=1.2, label="após 3000 épocas"); ax[1].set_ylim(-2.5, 2.5); ax[1].legend(fontsize=7.5, ncol=2); ax[1].set_xlabel("x"); ax[1].set_title("Rede 1–100–1, 30 pontos de treino")
salvar(fig, "f6_parada")

# ------------------------------------------------------------ comparação adicional: Friedman (R2 de teste) - 5 sementes
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
Xf, yf = make_friedman1(n_samples=400, n_features=10, noise=1.0, random_state=0); Xa, Xte, ya, yte = train_test_split(Xf, yf, test_size=0.25, random_state=0)
esx, esy = StandardScaler().fit(Xa), StandardScaler().fit(ya[:, None]); Xs, Xt, ys = esx.transform(Xa), esx.transform(Xte), esy.transform(ya[:, None]); Nf = len(ys)
for var in O.VARIANTES_RPROP:
    r2 = []; pf = []
    for s in range(5):
        r = MLP([10, 8, 1], ["tanh", "identidade"], semente=s); res = O.rprop(r.theta, lambda th: r.perda(th, Xs, ys) / Nf, lambda th: r.grad(th, Xs, ys) / Nf, variante=var, max_iter=300, tol_grad=0)
        r2.append(r2_score(yte, esy.inverse_transform(r.prever(Xt, res.theta)).ravel())); pf.append(res.historico["perda"][-1])
    print("Friedman #1 %-8s R2 teste mediana %.3f [%.3f, %.3f] | perda treino mediana %.4f" % (var, np.median(r2), min(r2), max(r2), np.median(pf)))
print("OK cap06")

# ------------------------------------------------------------ figura: organização do código
fig, ax = plt.subplots(figsize=(10.4, 3.9)); ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.9); ax.axis("off"); ax.grid(False)
caixa(ax, 0.2, 2.5, 2.8, 1.1, "Dados\n$X_{tr},T_{tr}$; $X_{va},T_{va}$; $X_{te},T_{te}$", cor=COR["cya"], fc="white", fs=9, bold=True)
caixa(ax, 3.4, 2.5, 3.2, 1.1, "Modelo `MLP`\n$\\theta$ (vetor), `forward`, `perda`,\n`grad`, `jacobiana`", cor=COR["ind"], fc="white", fs=9, bold=True)
caixa(ax, 7.0, 2.5, 3.2, 1.1, "Otimizador\n`rprop`, `quickprop`, `levenberg_marquardt`,\n`delta_bar_delta`, ...", cor=COR["grn"], fc="white", fs=9, bold=True)
caixa(ax, 0.2, 0.5, 4.6, 1.2, "Estado por parâmetro (mesma forma de $\\theta$)\n`delta`, `g_ant`, `dw_ant`  (arrays NumPy)", cor=COR["amb"], fc="white", fs=9, bold=True)
caixa(ax, 5.3, 0.5, 4.9, 1.2, "`Resultado`: `theta`, `theta_melhor`, `historico`\n(`perda`, `perda_val`, `norma_grad`, `tempo`), `n_iter`, `motivo`", cor=COR["vio"], fc="white", fs=9, bold=True)
seta(ax, (3.0, 3.05), (3.4, 3.05)); seta(ax, (6.6, 3.05), (7.0, 3.05)); seta(ax, (8.6, 2.5), (8.0, 1.72)); seta(ax, (2.5, 2.5), (2.5, 1.72), ls="--"); ax.text(3.4, 1.95, "funções `fun(θ)`, `grad(θ)`, `val(θ)`", fontsize=8, color=COR["gry"])
salvar(fig, "f6_estrutura")
