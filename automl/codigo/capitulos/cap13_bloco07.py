# Capítulo 13 — Explicabilidade, ética e limitações do AutoML
# Bloco 7. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# --- blocos anteriores deste capítulo (encadeados) ---
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
from sklearn.model_selection import train_test_split

r = np.random.default_rng(0); n = 6000
g = (r.random(n) < 0.3).astype(int)                          # 1 = grupo minoritário
z = r.normal(0, 1, n)                                        # capacidade real de pagamento (escondida)
renda = np.exp(0.5 * z + 0.35 * r.normal(size=n) - 0.25 * g) * 3.0
emprego = np.clip(5 + 2 * z + r.normal(0, 2, n), 0, None)
divida = np.clip(0.4 - 0.1 * z + r.normal(0, 0.1, n), 0, 1)
cep = 0.8 * g + 0.6 * r.normal(size=n)                       # variável proxy do grupo
score = z + r.normal(0, 1, n)
y_real = (z > 0).astype(int)                                 # quem de fato pagaria (só existe na simulação)
y_hist = ((z - 0.6 * g + r.normal(0, 0.5, n)) > 0).astype(int)   # decisões históricas, enviesadas contra o grupo 1
NOMES = ["renda", "tempo_emprego", "razao_divida", "cep_score", "score_historico"]
X = np.c_[renda, emprego, divida, cep, score]
Xa, Xte, ya, yte, ga, gte, yra, yrte = train_test_split(X, y_hist, g, y_real, test_size=0.3, random_state=0, stratify=y_hist)   # teste: 30%
Xtr, Xva, ytr, yva, gtr, gva, yrtr, yrva = train_test_split(Xa, ya, ga, yra, test_size=0.25, random_state=1, stratify=ya)       # treino e validação
print("Treino: %d | validação (calibração): %d | teste: %d" % (len(ytr), len(yva), len(yte)))

print("Taxa de aprovação NOS DADOS HISTÓRICOS  : grupo 0 = %.2f | grupo 1 = %.2f" % (ytr[gtr == 0].mean(), ytr[gtr == 1].mean()))
print("Taxa de bons pagadores (verdade, só aqui): grupo 0 = %.2f | grupo 1 = %.2f" % (yrtr[gtr == 0].mean(), yrtr[gtr == 1].mean()))
import lightgbm as lgb
import shap
from sklearn.inspection import permutation_importance
import os; os.makedirs("resultados", exist_ok=True)

modelo = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, random_state=0, verbose=-1).fit(Xtr, ytr)
print("Acurácia contra os rótulos históricos: %.3f" % modelo.score(Xte, yte))

pi = permutation_importance(modelo, Xte, yte, n_repeats=10, random_state=0, scoring="accuracy")
explicador = shap.TreeExplainer(modelo); sv = explicador.shap_values(Xte); sv = sv[1] if isinstance(sv, list) else sv
base = float(np.ravel(explicador.expected_value)[-1])        # valor-base: margem média do modelo (em log-odds)
shap_global = np.abs(sv).mean(0)
print("\nImportância dos atributos (SHAP em log-odds):")
print(f"  {'atributo':<16s} {'permutação (queda de acurácia)':>32s} {'SHAP médio |valor|':>20s}")
for i in np.argsort(-shap_global): print(f"  {NOMES[i]:<16s} {pi.importances_mean[i]:>32.3f} {shap_global[i]:>20.3f}")

i = 0; p = modelo.predict_proba(Xte[i:i + 1])[0, 1]
margem = base + sv[i].sum(); p_shap = 1 / (1 + np.exp(-margem))
print(f"\nExplicação local da pessoa #{i} (grupo = {gte[i]}), contribuições em log-odds:")
for j in np.argsort(-np.abs(sv[i])): print(f"  {NOMES[j]:<16s} valor = {Xte[i, j]:7.2f}   contribuição SHAP = {sv[i, j]:+.2f}")
print(f"  valor-base = {base:+.2f} | soma das contribuições = {sv[i].sum():+.2f} | margem = {margem:+.2f}")
print(f"  probabilidade = 1 / (1 + exp(-margem)) = {p_shap:.4f} | predict_proba do modelo = {p:.4f}")
json.dump({"nomes": NOMES, "perm": pi.importances_mean.tolist(), "shap": shap_global.tolist(), "local": sv[i].tolist(), "valores": Xte[i].tolist(), "p": float(p), "base": base, "margem": float(margem)},
          open("resultados/f13_expl.json", "w"))
from sklearn.ensemble import HistGradientBoostingClassifier
res = {}
print(f"{'versão':<10s} {'acurácia':>9s} | {'seleção g0':>10s} {'seleção g1':>10s} | {'TPR real g0':>11s} {'TPR real g1':>11s}")
for nome, cols in (("com CEP", [0, 1, 2, 3, 4]), ("sem CEP", [0, 1, 2, 4])):
    m = HistGradientBoostingClassifier(random_state=0).fit(Xtr[:, cols], ytr); p = m.predict(Xte[:, cols])
    sel = [p[gte == k].mean() for k in (0, 1)]; tpr = [p[(gte == k) & (yrte == 1)].mean() for k in (0, 1)]
    res[nome] = {"sel": sel, "tpr": tpr}
    print(f"{nome:<10s} {(p == yte).mean():>9.3f} | {sel[0]:>10.3f} {sel[1]:>10.3f} | {tpr[0]:>11.3f} {tpr[1]:>11.3f}")
import os; os.makedirs("resultados", exist_ok=True)
modelo = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, random_state=0, verbose=-1).fit(Xtr, ytr)
pva, pte = modelo.predict_proba(Xva)[:, 1], modelo.predict_proba(Xte)[:, 1]

# 2. limiares escolhidos SOMENTE na validação
sel0 = (pva[gva == 0] >= 0.5).mean()
t_paridade = np.quantile(pva[gva == 1], 1 - sel0)                     # iguala a taxa de seleção (paridade demográfica)
alvo = (pva[(gva == 0) & (yva == 1)] >= 0.5).mean()                   # TPR do grupo 0 contra os rótulos históricos
cand = np.linspace(0.05, 0.9, 171)
t_oport = min(cand, key=lambda t: abs((pva[(gva == 1) & (yva == 1)] >= t).mean() - alvo))   # iguala a TPR histórica
t_transd = np.quantile(pte[gte == 1], 1 - (pte[gte == 0] >= 0.5).mean())                    # variante transdutiva (usa o teste!)

# 3. regras congeladas, avaliadas uma vez no teste
def relatorio(nome, t1):
    pred = np.where(gte == 0, pte >= 0.5, pte >= t1).astype(int)
    sel = [pred[gte == k].mean() for k in (0, 1)]
    tprh = [pred[(gte == k) & (yte == 1)].mean() for k in (0, 1)]; tprr = [pred[(gte == k) & (yrte == 1)].mean() for k in (0, 1)]
    print(f"{nome:<24s} {t1:4.2f} {(pred == yte).mean():6.3f} {(pred == yrte).mean():6.3f} | {sel[0]:.2f}/{sel[1]:.2f} | {tprh[0]:.2f}/{tprh[1]:.2f} | {tprr[0]:.2f}/{tprr[1]:.2f}")
    return {"sel": sel, "tpr": tprr}

print(f"{'regra':<24s} {'t(g1)':>4s} {'acc_h':>6s} {'acc_v':>6s} | {'seleção':>9s} | {'TPR hist':>9s} | {'TPR real':>9s}")
base_r = relatorio("limiar único", 0.5)
dp_r = relatorio("paridade demográfica", t_paridade)
eo_r = relatorio("oport. (rótulos hist.)", t_oport)
relatorio("paridade (transdutiva)", t_transd)
json.dump({"base": base_r, "dp": dp_r, "eo": eo_r}, open("resultados/f13_just.json", "w"))
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss

def ece(p, y, faixas=10):
    f = np.minimum((p * faixas).astype(int), faixas - 1); e = 0.0
    for k in range(faixas):
        m = f == k
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

agressivo = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.1, num_leaves=63, random_state=0, verbose=-1).fit(Xtr, ytr)
candidatos = [("modelo do capítulo, bruto", modelo),
              ("agressivo, bruto", agressivo),
              ("agressivo + sigmoide (val.)", CalibratedClassifierCV(FrozenEstimator(agressivo), method="sigmoid").fit(Xva, yva)),
              ("agressivo + isotônica (val.)", CalibratedClassifierCV(FrozenEstimator(agressivo), method="isotonic").fit(Xva, yva))]
print(f"{'modelo':<30s} {'Brier':>7s} {'perda log':>10s} {'ECE':>7s}")
for nome, mdl in candidatos:
    p = mdl.predict_proba(Xte)[:, 1]
    print(f"{nome:<30s} {brier_score_loss(yte, p):7.4f} {log_loss(yte, p):10.4f} {ece(p, yte):7.4f}")
# --- fim dos blocos anteriores ---
alfa = 0.10                                                          # cobertura desejada: 90%
pv, pt = modelo.predict_proba(Xva), modelo.predict_proba(Xte)
escores = 1 - pv[np.arange(len(yva)), yva]                           # não conformidade na calibração (rótulos disponíveis)
n_cal = len(escores)
q = np.quantile(escores, min(1.0, np.ceil((n_cal + 1) * (1 - alfa)) / n_cal), method="higher")
conjunto = pt >= 1 - q                                               # matriz (teste x 2 classes): classe incluída?
coberto = conjunto[np.arange(len(yte)), yte]
tamanho = conjunto.sum(1); unitario = tamanho == 1; pred = pt.argmax(1)

print(f"Cobertura desejada: {1 - alfa:.0%} | cobertura no teste: {coberto.mean():.3f}")
print(f"Cobertura por grupo: grupo 0 = {coberto[gte == 0].mean():.3f} | grupo 1 = {coberto[gte == 1].mean():.3f}")
print(f"Tamanho médio do conjunto: {tamanho.mean():.2f} | conjuntos com 2 classes (abstenção): {(tamanho == 2).mean():.1%}")
print(f"Acurácia nos casos com conjunto unitário: {(pred[unitario] == yte[unitario]).mean():.3f} | acurácia em todos os casos: {(pred == yte).mean():.3f}")
