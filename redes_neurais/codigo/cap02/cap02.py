"""Cap. 2 - experimentos e figuras: derivada, cadeia, gradiente, GD, Newton, secantes, lógica, MQ.
Todos os números citados no capítulo saem deste script (semente fixa)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rnbook.estilo import plt, COR, salvar
from rnbook.desenho import caixa, seta
from rnbook.ativacoes import sigmoide, d_sigmoide, d_tanh

# ---------------------------------------------------------------- 2.1 derivada
f = np.sin; x0 = 1.0; exata = np.cos(x0)
hs = 10.0 ** -np.arange(1, 16)
err_f = np.abs((f(x0 + hs) - f(x0)) / hs - exata)
err_c = np.abs((f(x0 + hs) - f(x0 - hs)) / (2 * hs) - exata)
print("melhor h (progressiva): %.0e erro %.2e | melhor h (central): %.0e erro %.2e" % (hs[err_f.argmin()], err_f.min(), hs[err_c.argmin()], err_c.min()))
print("h=1e-6: progressiva %.2e central %.2e" % (err_f[5], err_c[5]))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
xx = np.linspace(0, 3, 200); g = lambda x: 0.5 * x**2; xa = 1.0
ax[0].plot(xx, g(xx), color=COR["ink"], lw=2, label="$f(x)=x^2/2$")
for h, c in [(1.5, COR["amb"]), (0.8, COR["grn"]), (0.3, COR["red"])]:
    m = (g(xa + h) - g(xa)) / h; ax[0].plot(xx, g(xa) + m * (xx - xa), color=c, lw=1.4, label=f"secante, h={h} (incl. {m:.2f})")
ax[0].plot(xx, g(xa) + xa * (xx - xa), "--", color=COR["ind"], lw=2, label="tangente (incl. 1,00)")
ax[0].scatter([xa], [g(xa)], color=COR["ind"], zorder=5); ax[0].set_ylim(-0.3, 4); ax[0].legend(fontsize=7.5); ax[0].set_title("A derivada é o limite das inclinações das secantes"); ax[0].set_xlabel("x")
ax[1].loglog(hs, err_f, "o-", color=COR["amb"], label="diferença progressiva"); ax[1].loglog(hs, err_c, "s-", color=COR["ind"], label="diferença central")
ax[1].set_xlabel("passo h"); ax[1].set_ylabel("erro absoluto"); ax[1].legend(); ax[1].set_title("Erro numérico de $\\sin'(1)$: truncamento × arredondamento")
salvar(fig, "f2_derivada")

# ---------------------------------------------------------------- ativações e derivadas
z = np.linspace(-6, 6, 400)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ax[0].plot(z, sigmoide(z), color=COR["ind"], label="$\\sigma(z)$"); ax[0].plot(z, d_sigmoide(z), "--", color=COR["red"], label="$\\sigma'(z)$"); ax[0].set_title("sigmoide")
ax[1].plot(z, np.tanh(z), color=COR["ind"], label="$\\tanh z$"); ax[1].plot(z, d_tanh(z), "--", color=COR["red"], label="$\\tanh'z$"); ax[1].set_title("tangente hiperbólica")
ax[2].plot(z, np.maximum(0, z), color=COR["ind"], label="ReLU"); ax[2].plot(z, (z > 0) * 1.0, "--", color=COR["red"], label="ReLU'"); ax[2].set_title("ReLU")
for a in ax: a.legend(); a.set_xlabel("z")
salvar(fig, "f2_ativ_deriv")
print("sigmoide'(0)=%.4f (máx 1/4) | tanh'(0)=%.4f" % (d_sigmoide(0.0), d_tanh(0.0)))

# ---------------------------------------------------------------- 2.1.1 regra da cadeia: exemplo micro-backprop
x, w, b, t = 1.5, 0.8, -0.2, 1.0
zz = w * x + b; y = float(sigmoide(zz)); E = 0.5 * (y - t) ** 2
dE_dy = y - t; dy_dz = y * (1 - y); dE_dz = dE_dy * dy_dz; dE_dw = dE_dz * x; dE_db = dE_dz
Ef = lambda ww, bb: 0.5 * (float(sigmoide(ww * x + bb)) - t) ** 2
h = 1e-6
print("cadeia: z=%.4f y=%.4f E=%.5f dE/dy=%.4f dy/dz=%.4f dE/dz=%.5f dE/dw=%.5f dE/db=%.5f" % (zz, y, E, dE_dy, dy_dz, dE_dz, dE_dw, dE_db))
print("   numérico: dE/dw=%.5f dE/db=%.5f" % ((Ef(w + h, b) - Ef(w - h, b)) / (2 * h), (Ef(w, b + h) - Ef(w, b - h)) / (2 * h)))
fig, ax = plt.subplots(figsize=(10, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off"); ax.grid(False)
nos = [("$w,\\,x,\\,b$", 0.2, f"w={w}, x={x}\nb={b}"), ("$z=wx+b$", 2.3, f"z={zz:.2f}"), ("$y=\\sigma(z)$", 4.4, f"y={y:.4f}"), ("$r=y-t$", 6.5, f"r={y-t:.4f}"), ("$E=\\frac{1}{2}r^2$", 8.4, f"E={E:.5f}")]
for tt, xp, val in nos:
    caixa(ax, xp, 1.9, 1.5, 1.0, tt, cor=COR["ind"], fc="white", fs=10); ax.text(xp + 0.75, 3.15, val, ha="center", fontsize=8.5, color=COR["ind"])
for i in range(4): seta(ax, (nos[i][1] + 1.52, 2.4), (nos[i + 1][1] - 0.02, 2.4), cor=COR["ind"])
loc = [("$\\partial z/\\partial w=x=1{,}5$", 2.3 + 0.75), ("$\\sigma'(z)=y(1-y)=%.4f$" % dy_dz, 4.4 + 0.75), ("$\\partial r/\\partial y=1$", 6.5 + 0.75), ("$\\partial E/\\partial r=r$", 8.4 + 0.75)]
for tt, xp in loc: ax.text(xp, 1.45, tt, ha="center", fontsize=8.3, color=COR["gry"])
for i in range(4, 0, -1): seta(ax, (nos[i][1] - 0.02, 1.85), (nos[i - 1][1] + 1.52, 1.85), cor=COR["red"], lw=1.2)
ax.text(5, 0.6, f"Retropropagação: $\\partial E/\\partial w = r\\cdot\\sigma'(z)\\cdot x = {dE_dy:.4f}\\cdot{dy_dz:.4f}\\cdot1{{,}}5 = {dE_dw:.5f}$", ha="center", fontsize=10.5, color=COR["red"])
ax.text(5, 0.1, "setas azuis: cálculo direto (passo para frente) · setas vermelhas: derivadas locais multiplicadas de trás para frente", ha="center", fontsize=8.5, color=COR["gry"])
salvar(fig, "f2_cadeia")

# ---------------------------------------------------------------- 2.2-2.6: superfície, gradiente, direcional, Hessiana
fq = lambda X, Y: 0.5 * (X**2 + 10 * Y**2)
gx, gy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-1.6, 1.6, 200))
p = np.array([2.0, 1.0]); grad = np.array([p[0], 10 * p[1]])
print("gradiente em (2,1) = ", grad, "norma %.4f" % np.linalg.norm(grad))
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
cs = ax[0].contour(gx, gy, fq(gx, gy), levels=[0.5, 2, 5, 10, 16, 25], colors=COR["gry"], linewidths=0.9)
ax[0].clabel(cs, fontsize=7); ax[0].scatter(*p, color=COR["ind"], zorder=5)
ax[0].annotate("", xy=p + 0.12 * grad, xytext=p, arrowprops=dict(arrowstyle="-|>", color=COR["red"], lw=2)); ax[0].text(p[0] + 0.1, p[1] + 0.9, "$\\nabla f$", color=COR["red"], fontsize=11)
ax[0].annotate("", xy=p - 0.12 * grad, xytext=p, arrowprops=dict(arrowstyle="-|>", color=COR["grn"], lw=2)); ax[0].text(p[0] - 0.75, p[1] - 1.05, "$-\\nabla f$", color=COR["grn"], fontsize=11)
ax[0].set_title("$f(x,y)=\\frac{1}{2}(x^2+10y^2)$: gradiente é perpendicular às curvas de nível"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].set_aspect("equal"); ax[0].set_xlim(-3, 3); ax[0].set_ylim(-1.6, 1.6)
th = np.linspace(0, 2 * np.pi, 361); D = grad[0] * np.cos(th) + grad[1] * np.sin(th)
ax[1].plot(np.degrees(th), D, color=COR["ind"], lw=2); ax[1].axhline(0, color=COR["gry"], lw=0.8)
ax[1].scatter([np.degrees(np.arctan2(grad[1], grad[0]))], [np.linalg.norm(grad)], color=COR["red"], zorder=5, label="máximo = $\\|\\nabla f\\|$ na direção do gradiente")
ax[1].scatter([np.degrees(np.arctan2(grad[1], grad[0])) + 180], [-np.linalg.norm(grad)], color=COR["grn"], zorder=5, label="mínimo: direção $-\\nabla f$")
ax[1].set_xlabel("ângulo $\\vartheta$ da direção $\\mathbf{u}$ (graus)"); ax[1].set_ylabel("$D_{\\mathbf{u}}f=\\nabla f\\cdot\\mathbf{u}$"); ax[1].legend(fontsize=8); ax[1].set_title("Derivada direcional no ponto (2, 1)")
salvar(fig, "f2_gradiente")
ang = np.degrees(np.arctan2(grad[1], grad[0])); print("ângulo do gradiente %.2f graus; D_u para u=(1,0): %.2f; u=(0,1): %.2f; u=(1,1)/sqrt2: %.4f" % (ang, grad[0], grad[1], grad @ np.array([1, 1]) / np.sqrt(2)))

fig = plt.figure(figsize=(11, 3.6))
Xs, Ys = np.meshgrid(np.linspace(-1.5, 1.5, 60), np.linspace(-1.5, 1.5, 60))
for i, (Z, tit) in enumerate([(Xs**2 + Ys**2, "$H=\\mathrm{diag}(2,\\,2)$: mínimo"), (Xs**2 - Ys**2, "$H=\\mathrm{diag}(2,\\,-2)$: sela"), (-Xs**2 - Ys**2, "$H=\\mathrm{diag}(-2,\\,-2)$: máximo")]):
    a = fig.add_subplot(1, 3, i + 1, projection="3d"); a.plot_surface(Xs, Ys, Z, cmap="viridis", alpha=0.9, linewidth=0); a.set_title(tit, fontsize=9); a.set_xticks([]); a.set_yticks([]); a.set_zticks([])
salvar(fig, "f2_hessiana")

# Jacobiana e Hessiana: exemplos
F = lambda x, y: np.array([x**2 * y, 5 * x + np.sin(y)])
J = lambda x, y: np.array([[2 * x * y, x**2], [5, np.cos(y)]])
print("Jacobiana em (1,2):\n", J(1, 2))
Jn = np.stack([(F(1 + 1e-6, 2) - F(1 - 1e-6, 2)) / 2e-6, (F(1, 2 + 1e-6) - F(1, 2 - 1e-6)) / 2e-6], axis=1); print("numérica:\n", Jn.round(5))
Hf = lambda x, y: np.array([[2 * y, 2 * x], [2 * x, 6 * y]])   # f = x^2 y + y^3
print("Hessiana de x^2 y + y^3 em (1,2):", Hf(1, 2).tolist(), "autovalores", np.linalg.eigvalsh(Hf(1, 2)).round(4))
print("Hessiana de f(x,y)=x^2-y^2 autovalores:", np.linalg.eigvalsh(np.diag([2., -2.])))

# ---------------------------------------------------------------- 2.7 gradiente descendente
H = np.diag([1.0, 10.0]); fgd = lambda v: 0.5 * v @ H @ v; ggd = lambda v: H @ v
def gd(v0, eta, n):
    v = np.array(v0, float); tr = [v.copy()]
    for _ in range(n): v = v - eta * ggd(v); tr.append(v.copy())
    return np.array(tr)
cases = [(0.02, COR["amb"]), (0.18, COR["grn"]), (0.205, COR["red"])]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
cs = ax[0].contour(gx, gy, fq(gx, gy), levels=[0.5, 2, 5, 10, 16, 25], colors="#CBD5E1", linewidths=0.8)
for eta, c in cases:
    n = 40 if eta < 0.2 else 14
    tr = gd([2.5, 1.2], eta, n); ax[0].plot(tr[:, 0], tr[:, 1], "o-", ms=3, lw=1.2, color=c, label=f"$\\eta={eta}$")
    L = np.array([fgd(v) for v in tr]); ax[1].semilogy(L + 1e-16, color=c, label=f"$\\eta={eta}$")
    print("eta=%.3f: f após %d passos = %.3e" % (eta, n, L[-1]))
ax[0].set_xlim(-3, 3); ax[0].set_ylim(-1.6, 1.6); ax[0].legend(fontsize=8); ax[0].set_title("Trajetórias do gradiente descendente"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[1].set_xlabel("iteração"); ax[1].set_ylabel("$f$ (escala log)"); ax[1].legend(fontsize=8); ax[1].set_title("Convergência: lenta, boa, divergente")
salvar(fig, "f2_gd")
kappa = 10; print("fator de contração ótimo (k-1)/(k+1) = %.4f, eta_otimo = 2/(1+k) = %.4f, limite 2/L = %.4f" % ((kappa - 1) / (kappa + 1), 2 / (1 + kappa), 2 / 10))
# iterações para f<1e-6 com eta=0.18
tr = gd([2.5, 1.2], 0.18, 2000); Lv = np.array([fgd(v) for v in tr]); print("eta=0.18: f<1e-6 em %d passos" % np.argmax(Lv < 1e-6))
tr = gd([2.5, 1.2], 0.02, 2000); Lv = np.array([fgd(v) for v in tr]); print("eta=0.02: f<1e-6 em %d passos" % np.argmax(Lv < 1e-6))

# ---------------------------------------------------------------- 2.8 Newton
fn = lambda x: np.exp(x) - 2 * x; dfn = lambda x: np.exp(x) - 2; d2fn = lambda x: np.exp(x)
xs = [0.0]
for _ in range(5): xs.append(xs[-1] - dfn(xs[-1]) / d2fn(xs[-1]))
print("Newton 1D (x0=0):", [round(v, 6) for v in xs], "ln2 =", round(np.log(2), 6))
xg = [0.0]
for _ in range(30): xg.append(xg[-1] - 0.3 * dfn(xg[-1]))
print("GD 1D eta=0.3 após 5 e 10 passos:", round(xg[5], 6), round(xg[10], 6), "erro10 %.2e" % abs(xg[10] - np.log(2)))
xx = np.linspace(-0.4, 1.6, 300)
fig, ax = plt.subplots(1, 2, figsize=(10.3, 3.6))
ax[0].plot(xx, fn(xx), color=COR["ink"], lw=2, label="$f(x)=e^x-2x$")
xk = 0.0; q = lambda x, xk: fn(xk) + dfn(xk) * (x - xk) + 0.5 * d2fn(xk) * (x - xk) ** 2
ax[0].plot(xx, q(xx, xk), "--", color=COR["amb"], label="parábola em $x_0=0$")
x1 = xk - dfn(xk) / d2fn(xk); ax[0].plot(xx, q(xx, x1), "--", color=COR["grn"], label="parábola em $x_1=1$")
ax[0].scatter([0, x1, xs[2]], [fn(0), fn(x1), fn(xs[2])], color=COR["ind"], zorder=5); ax[0].set_ylim(0.4, 2.6); ax[0].legend(fontsize=8); ax[0].set_title("Newton: salta para o mínimo da parábola local"); ax[0].set_xlabel("x")
ax[1].semilogy(range(6), [abs(v - np.log(2)) for v in xs], "o-", color=COR["grn"], label="Newton"); ax[1].semilogy(range(6), [abs(v - np.log(2)) for v in xg[:6]], "s-", color=COR["amb"], label="gradiente ($\\eta=0{,}3$)")
ax[1].set_xlabel("iteração"); ax[1].set_ylabel("erro $|x_k-\\ln2|$"); ax[1].legend(); ax[1].set_title("Convergência quadrática × linear")
salvar(fig, "f2_newton")
# Newton em quadrática 2D: um passo
v = np.array([2.5, 1.2]); v1 = v - np.linalg.solve(H, ggd(v)); print("Newton na quadrática 2D: v1 =", v1)

# ---------------------------------------------------------------- 2.9 secantes
fr = lambda x: x**2 - 2
s = [1.0, 2.0]
for _ in range(7): s.append(s[-1] - fr(s[-1]) * (s[-1] - s[-2]) / (fr(s[-1]) - fr(s[-2])))
er = [abs(v - np.sqrt(2)) for v in s]; print("secante raiz de x^2-2:", [round(v, 8) for v in s[:7]]); print("erros:", ["%.2e" % e for e in er[:8]])
ratios = [np.log(er[k + 1]) / np.log(er[k]) for k in range(1, 5)]; print("razões log(e_{k+1})/log(e_k):", np.round(ratios, 3), "(golden 1.618)")
# secante para minimizar: raiz de f'(x)
sec = [0.0, 1.0]
for _ in range(6):
    a, b2 = sec[-2], sec[-1]; sec.append(b2 - dfn(b2) * (b2 - a) / (dfn(b2) - dfn(a)))
print("secante em f'(x)=e^x-2:", [round(v, 6) for v in sec])
print("curvatura aproximada na 1ª secante: (f'(1)-f'(0))/(1-0)=%.4f vs f''(0.5)=%.4f" % ((dfn(1) - dfn(0)), d2fn(0.5)))
fig, ax = plt.subplots(1, 2, figsize=(10.3, 3.6)); xx = np.linspace(0.8, 2.2, 200)
ax[0].plot(xx, fr(xx), color=COR["ink"], lw=2, label="$f(x)=x^2-2$"); ax[0].axhline(0, color=COR["gry"], lw=0.8)
x_a, x_b = 1.0, 2.0
ax[0].plot(xx, fr(x_b) + (fr(x_b) - fr(x_a)) / (x_b - x_a) * (xx - x_b), "--", color=COR["amb"], label="secante por $x_0,x_1$")
x_c = s[2]; ax[0].plot(xx, fr(x_c) + (fr(x_c) - fr(x_b)) / (x_c - x_b) * (xx - x_c), "--", color=COR["grn"], label="secante por $x_1,x_2$")
ax[0].scatter([x_a, x_b, x_c], [fr(x_a), fr(x_b), fr(x_c)], color=COR["ind"], zorder=5); ax[0].legend(fontsize=8); ax[0].set_ylim(-1.6, 2.5); ax[0].set_title("Secantes para encontrar uma raiz"); ax[0].set_xlabel("x")
xxx = np.linspace(-0.3, 1.6, 200); ax[1].plot(xxx, dfn(xxx), color=COR["ink"], lw=2, label="$f'(x)=e^x-2$"); ax[1].axhline(0, color=COR["gry"], lw=0.8)
ax[1].plot([0, 1], [dfn(0), dfn(1)], "o--", color=COR["red"], label="secante de $f'$: estima $f''\\approx\\Delta f'/\\Delta x$"); ax[1].legend(fontsize=8); ax[1].set_title("Secantes para estimar curvatura (otimização)"); ax[1].set_xlabel("x")
salvar(fig, "f2_secantes")

# ---------------------------------------------------------------- 2.10 lógica: portas e separabilidade
P4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); portas = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "NAND": [1, 1, 1, 0], "XOR": [0, 1, 1, 0]}
retas = {"AND": (1, 1, -1.5), "OR": (1, 1, -0.5), "NAND": (-1, -1, 1.5)}
fig, ax = plt.subplots(1, 4, figsize=(11.5, 3.1))
for a, (nm, tg) in zip(ax, portas.items()):
    for p_, t_ in zip(P4, tg): a.scatter(*p_, s=170, color=COR["grn"] if t_ else COR["red"], edgecolor="white", zorder=3); a.text(p_[0] + 0.07, p_[1] + 0.07, str(t_), fontsize=9)
    if nm in retas:
        w1, w2, bb = retas[nm]; xl = np.linspace(-0.3, 1.3, 10); a.plot(xl, -(w1 * xl + bb) / w2, color=COR["ind"], lw=2)
    else: a.text(0.5, 0.5, "nenhuma reta\nsepararia", ha="center", va="center", fontsize=9, color=COR["red"], bbox=dict(fc="white", ec=COR["red"], boxstyle="round"))
    a.set_xlim(-0.3, 1.3); a.set_ylim(-0.3, 1.3); a.set_title(nm, fontweight="bold"); a.set_xlabel("$x_1$"); a.set_aspect("equal")
salvar(fig, "f2_logica")

# ---------------------------------------------------------------- 2.11 mínimos quadrados
rng = np.random.default_rng(1); xd = np.linspace(0, 10, 12); yd = 1.5 + 0.8 * xd + rng.normal(0, 0.9, xd.size)
A = np.column_stack([np.ones_like(xd), xd])
th_ne = np.linalg.solve(A.T @ A, A.T @ yd); th_ls = np.linalg.lstsq(A, yd, rcond=None)[0]; Q, R = np.linalg.qr(A); th_qr = np.linalg.solve(R, Q.T @ yd)
print("MQ: normais", th_ne.round(5), "lstsq", th_ls.round(5), "QR", th_qr.round(5), "| SSE=%.4f" % (0.5 * np.sum((A @ th_ls - yd) ** 2)))
thg = np.zeros(2)
for _ in range(20000): thg -= 0.002 * (A.T @ (A @ thg - yd))
print("gradiente descendente (20000 passos, eta=0.002):", thg.round(5), "cond(A^T A)=%.1f cond(A)=%.1f" % (np.linalg.cond(A.T @ A), np.linalg.cond(A)))
xx = np.linspace(0, 10, 200)
fig, ax = plt.subplots(1, 2, figsize=(10.3, 3.7))
ax[0].scatter(xd, yd, color=COR["gry"], zorder=3, label="dados"); ax[0].plot(xx, th_ls[0] + th_ls[1] * xx, color=COR["ind"], lw=2, label="ajuste por mínimos quadrados")
for xi, yi in zip(xd, yd): ax[0].plot([xi, xi], [yi, th_ls[0] + th_ls[1] * xi], color=COR["red"], lw=1)
ax[0].legend(); ax[0].set_title("Os resíduos (em vermelho) são minimizados em soma de quadrados"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
for gr, c in [(1, COR["ind"]), (3, COR["grn"]), (9, COR["red"])]:
    cf = np.polyfit(xd, yd, gr); ax[1].plot(xx, np.polyval(cf, xx), color=c, label=f"grau {gr}")
ax[1].scatter(xd, yd, color=COR["gry"], zorder=3); ax[1].set_ylim(-3, 14); ax[1].legend(); ax[1].set_title("Mais parâmetros ajustam melhor o treino... e o ruído"); ax[1].set_xlabel("x")
salvar(fig, "f2_minimos_quadrados")
for gr in (1, 3, 9): cf = np.polyfit(xd, yd, gr); print("grau %d: SSE treino = %.4f" % (gr, 0.5 * np.sum((np.polyval(cf, xd) - yd) ** 2)))
print("OK cap02")
