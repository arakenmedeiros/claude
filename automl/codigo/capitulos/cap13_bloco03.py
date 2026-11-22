# Capítulo 13 — Explicabilidade, ética e limitações do AutoML
# Bloco 3. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
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
# --- fim dos blocos anteriores ---
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
