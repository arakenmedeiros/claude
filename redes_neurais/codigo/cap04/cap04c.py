"""Cap. 4 (parte C): correspondência com o scikit-learn, divisão da amostra, vazamento de informação."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import numpy as np
from rnbook import MLP
from rnbook.estilo import plt, COR, salvar
from rnbook.desenho import caixa, seta
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ---------------------------------------------------- 1) o nosso MLP reproduz o do scikit-learn?
rng = np.random.default_rng(0); X = rng.uniform(0, 1, (120, 1)); t = (np.sin(2 * np.pi * X) + rng.normal(0, 0.15, X.shape)).ravel()
sk = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", alpha=0.0, max_iter=5000, tol=1e-12, random_state=0).fit(X, t)
W1, W2 = sk.coefs_; b1, b2 = sk.intercepts_
print("formas sklearn: coefs_", [w.shape for w in sk.coefs_], "intercepts_", [b.shape for b in sk.intercepts_], "| out_activation_", sk.out_activation_)
rede = MLP([1, 8, 1], ["tanh", "identidade"]); theta = np.concatenate([W1.T.ravel(), b1, W2.T.ravel(), b2]); rede.theta = theta
print("max |predição própria - sklearn| = %.2e" % np.abs(rede.prever(X, theta).ravel() - sk.predict(X)).max())
Tm = t[:, None]; E_sse = rede.perda(theta, X, Tm, "sse")
print("loss_ do sklearn = %.8f | E_sse/(N*m) = %.8f | E_sse/N = %.8f" % (sk.loss_, E_sse / (len(t) * 1), E_sse / len(t)))
g = rede.grad(theta, X, Tm, "sse"); print("norma do nosso gradiente (SSE) no ponto final do lbfgs do sklearn: %.2e | norma do vetor theta: %.2f" % (np.linalg.norm(g), np.linalg.norm(theta)))
# com alpha > 0
sk2 = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", alpha=0.5, max_iter=5000, tol=1e-12, random_state=0).fit(X, t)
Wa, Wb = sk2.coefs_; th2 = np.concatenate([Wa.T.ravel(), sk2.intercepts_[0], Wb.T.ravel(), sk2.intercepts_[1]]); rede.theta = th2
pen = 0.5 * 0.5 * (np.sum(Wa ** 2) + np.sum(Wb ** 2)) / len(t); print("com alpha=0.5: loss_=%.8f | E_sse/N + penal. = %.8f (penalização só nos pesos, não nos vieses; dividida por N)" % (sk2.loss_, rede.perda(th2, X, Tm, "sse") / len(t) + pen))
# inicialização do sklearn: limite Glorot
print("limite de inicialização (Glorot uniforme) camada 1 (fan_in=1, fan_out=8): tanh sqrt(6/9)=%.4f ; logistic sqrt(2/9)=%.4f" % (np.sqrt(6 / 9), np.sqrt(2 / 9)))

# ---------------------------------------------------- 2) MLPClassifier: saída e perda
from sklearn.datasets import load_iris, make_moons
Xi, yi = load_iris(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.3, random_state=0, stratify=yi)
cl = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(5,), max_iter=2000, random_state=0)).fit(Xtr, ytr)
c = cl[-1]; print("MLPClassifier: out_activation_ =", c.out_activation_, "| classes_", c.classes_, "| n_layers_", c.n_layers_, "| n_iter_", c.n_iter_, "| loss_ %.4f" % c.loss_)
Xb, yb = make_moons(300, noise=0.25, random_state=0); cb = MLPClassifier(hidden_layer_sizes=(5,), max_iter=3000, random_state=0).fit(Xb, yb); print("binário (moons): out_activation_ =", cb.out_activation_, "| coefs_ último:", cb.coefs_[-1].shape)

# ---------------------------------------------------- 3) divisão da amostra e curva de validação (nº de neurônios)
rng = np.random.default_rng(5); N = 150; Xs = rng.uniform(0, 1, (N, 1)); ts = (np.sin(2 * np.pi * Xs) + rng.normal(0, 0.2, Xs.shape)).ravel()
Xtmp, Xte2, ttmp, tte2 = train_test_split(Xs, ts, test_size=0.2, random_state=0); Xtr2, Xva2, ttr2, tva2 = train_test_split(Xtmp, ttmp, test_size=0.25, random_state=0)
print("tamanhos treino/validação/teste:", len(ttr2), len(tva2), len(tte2))
Hs = [1, 2, 3, 5, 8, 12, 20, 40, 80]; etr, eva = [], []
for H in Hs:
    tr_, va_ = [], []
    for s in range(4):
        m = MLPRegressor(hidden_layer_sizes=(H,), activation="tanh", solver="lbfgs", alpha=0.0, max_iter=3000, random_state=s).fit(Xtr2, ttr2)
        tr_.append(np.sqrt(np.mean((m.predict(Xtr2) - ttr2) ** 2))); va_.append(np.sqrt(np.mean((m.predict(Xva2) - tva2) ** 2)))
    etr.append(np.mean(tr_)); eva.append(np.mean(va_)); print("H=%3d: RMSE treino %.4f | validação %.4f" % (H, etr[-1], eva[-1]))
Hbest = Hs[int(np.argmin(eva))]; print("H escolhido pela validação:", Hbest)
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.7))
# esquema da divisão
a = ax[0]; a.set_xlim(0, 10); a.set_ylim(0, 4); a.axis("off"); a.grid(False)
caixa(a, 0.2, 2.6, 5.6, 0.8, "Treino (60%): ajusta $\\theta$", cor=COR["ind"], fc="#EEF2FF", fs=9)
caixa(a, 5.9, 2.6, 2.0, 0.8, "Validação (20%)", cor=COR["amb"], fc="#FFFBEB", fs=9); caixa(a, 8.0, 2.6, 1.8, 0.8, "Teste (20%)", cor=COR["red"], fc="#FEF2F2", fs=9)
a.text(3.0, 2.2, "gradientes e atualizações", ha="center", fontsize=8.5, color=COR["ind"]); a.text(6.9, 2.2, "escolhas: $H$, $\\eta$,\nparada antecipada", ha="center", fontsize=8.5, color=COR["amb"]); a.text(8.9, 2.2, "uma única vez,\nno final", ha="center", fontsize=8.5, color=COR["red"])
a.text(5, 1.0, "Pré-processamento (média, desvio): ajustado SOMENTE com o treino\ne depois aplicado à validação e ao teste", ha="center", fontsize=9.5, bbox=dict(fc="white", ec=COR["grn"], boxstyle="round"))
a.set_title("Divisão da amostra")
ax[1].semilogx(Hs, etr, "o-", color=COR["ind"], label="treino"); ax[1].semilogx(Hs, eva, "s-", color=COR["amb"], label="validação"); ax[1].axhline(0.2, color=COR["gry"], ls=":", lw=1); ax[1].text(1.2, 0.205, "ruído verdadeiro (0,20)", fontsize=8, color=COR["gry"])
ax[1].scatter([Hbest], [min(eva)], s=120, facecolors="none", edgecolors=COR["red"], zorder=5); ax[1].set_xlabel("neurônios ocultos $H$"); ax[1].set_ylabel("RMSE"); ax[1].legend(); ax[1].set_title("Treino × validação em função de $H$")
salvar(fig, "f4_divisao")
mfin = MLPRegressor(hidden_layer_sizes=(Hbest,), activation="tanh", solver="lbfgs", alpha=0.0, max_iter=3000, random_state=0).fit(np.vstack([Xtr2, Xva2]), np.concatenate([ttr2, tva2]))
print("teste (usado uma vez): RMSE %.4f, R2 %.4f" % (np.sqrt(np.mean((mfin.predict(Xte2) - tte2) ** 2)), r2_score(tte2, mfin.predict(Xte2))))

# ---------------------------------------------------- 4) vazamento: seleção de atributos fora do pipeline (rótulos aleatórios)
rng = np.random.default_rng(0); Xr = rng.normal(size=(60, 2000)); yr = rng.integers(0, 2, 60)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
Xsel = SelectKBest(f_classif, k=20).fit_transform(Xr, yr)                      # ERRADO: usa todos os rótulos
acc_vaz = cross_val_score(LogisticRegression(max_iter=2000), Xsel, yr, cv=cv).mean()
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), LogisticRegression(max_iter=2000))
acc_ok = cross_val_score(pipe, Xr, yr, cv=cv).mean()
print("rótulos ALEATÓRIOS (60 amostras, 2000 atributos): validação cruzada com seleção FORA do pipeline = %.3f | DENTRO do pipeline = %.3f (esperado ~0,5)" % (acc_vaz, acc_ok))
print("OK cap04c")
