# Capítulo 10 — Ferramentas de AutoML na prática
# Bloco 10. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# ATENÇÃO: este trecho NÃO foi executado na elaboração do livro. Fixe versões e confira a documentação.
from tabpfn import TabPFNClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import time

for nome, modelo in (("TabPFN", TabPFNClassifier()), ("Boosting padrão", HistGradientBoostingClassifier(random_state=0))):
    t0 = time.perf_counter(); modelo.fit(Xtr, ytr); t_fit = time.perf_counter() - t0   # "treino" (no TabPFN, só guarda o contexto)
    t0 = time.perf_counter(); previsao = modelo.predict(Xte); t_pred = time.perf_counter() - t0   # inferência
    print(f"{nome:<16s} acurácia = {(previsao == yte).mean():.3f} | fit = {t_fit:.2f} s | predict = {t_pred:.2f} s")
