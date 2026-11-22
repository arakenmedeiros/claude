# Capítulo 14 — MLOps, implantação e projeto integrador
# Bloco 6. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# --- blocos anteriores deste capítulo (encadeados) ---
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
import os; os.makedirs("resultados", exist_ok=True)

def gera(n, seed, infl=0.0):
    r = np.random.default_rng(seed)
    g = (r.random(n) < 0.3).astype(int); z = r.normal(0, 1, n)
    renda = np.exp(0.5 * z + 0.35 * r.normal(size=n) - 0.25 * g) * 3.0 * (1 + infl)       # a renda sobe com a inflação
    emprego = np.clip(5 + 2 * z + r.normal(0, 2, n), 0, None); divida = np.clip(0.4 - 0.1 * z + r.normal(0, 0.1, n), 0, 1)
    score = z + r.normal(0, 1, n)
    return np.c_[renda, emprego, divida, score], (z > 0).astype(int)

def psi(ref, novo, k=10):
    cortes = np.quantile(ref, np.linspace(0, 1, k + 1)[1:-1])
    a = np.bincount(np.searchsorted(cortes, ref), minlength=k) / len(ref)
    b = np.bincount(np.searchsorted(cortes, novo), minlength=k) / len(novo)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))

Xtr, ytr = gera(5000, seed=1)
modelo = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
print(f"{'mês':>3s} {'inflação':>9s} {'acurácia':>9s} {'KS (renda)':>11s} {'valor-p':>9s} {'PSI renda':>10s} {'PSI emprego':>12s}")
linhas = []
for mes in range(0, 7):
    infl = 0.10 * mes
    Xn, yn = gera(2000, seed=100 + mes, infl=infl)
    acc = modelo.score(Xn, yn); ks = ks_2samp(Xtr[:, 0], Xn[:, 0]); pr = psi(Xtr[:, 0], Xn[:, 0]); pe = psi(Xtr[:, 1], Xn[:, 1])
    linhas.append({"mes": mes, "acc": acc, "ks": float(ks.statistic), "psi_renda": pr, "psi_emp": pe})
    print(f"{mes:>3d} {infl:>8.0%} {acc:>9.3f} {ks.statistic:>11.3f} {ks.pvalue:>9.1e} {pr:>10.3f} {pe:>12.3f}")
json.dump(linhas, open("resultados/f14_deriva.json", "w"))
# --- fim dos blocos anteriores ---
import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier

Xr, yr = gera(5000, seed=500, infl=0.60)                      # dados recentes, já com rótulos
X_ret, y_ret, X_des, y_des = Xr[:2000], yr[:2000], Xr[2000:], yr[2000:]
desafiante = HistGradientBoostingClassifier(random_state=0).fit(X_ret, y_ret)

acerto_c = modelo.predict(X_des) == y_des                     # acertos por caso, nos MESMOS casos
acerto_d = desafiante.predict(X_des) == y_des
n = len(y_des); dif = acerto_d.astype(int) - acerto_c.astype(int)     # +1: só o desafiante acertou; -1: só o campeão; 0: empate
b, c = int((dif == -1).sum()), int((dif == 1).sum())                  # discordâncias
rng = np.random.default_rng(0)
boot = np.array([dif[rng.integers(0, n, n)].mean() for _ in range(5000)])    # bootstrap pareado
lo, hi = np.percentile(boot, [2.5, 97.5])
p_mcnemar = binomtest(c, b + c, 0.5).pvalue                           # McNemar exato
delta = 0.005                                                         # melhoria mínima de interesse: 0,5 ponto percentual

print(f"Casos do desafio: {n}")
print(f"Acurácia: campeão = {acerto_c.mean():.3f} | desafiante = {acerto_d.mean():.3f} | diferença = {dif.mean():+.4f}")
print(f"Discordâncias: só o campeão acertou = {b} | só o desafiante acertou = {c}")
print(f"IC de 95% da diferença (bootstrap pareado): [{lo:+.4f}, {hi:+.4f}] | McNemar exato: p = {p_mcnemar:.4f}")
erro_ing = np.sqrt(acerto_d.mean() * (1 - acerto_d.mean()) / n)
print(f"Erro padrão de UMA acurácia = {erro_ing:.4f} | da diferença pareada = {dif.std(ddof=1) / np.sqrt(n):.4f}")
print("Decisão pela regra (limite inferior > δ = 0,005):", "desafiante assume" if lo > delta else "mantém o campeão por ora")
