# Capítulo 12 — solução de referência (EX 2)
import numpy as np
import statsmodels.api as sm

y = sm.datasets.co2.load_pandas().data["co2"].ffill().resample("MS").mean().values        # preenchimento causal (sem olhar o futuro)
n, H = len(y), 36
real = y[n - H:]
ing = y[n - H - 1:n - 1]
saz = y[n - H - 12:n - 12]
mm = np.array([y[t - 12:t].mean() for t in range(n - H, n)])
for nome, p in (("ingênua", ing), ("sazonal ingênua", saz), ("média móvel de 12", mm)):
    print(f"{nome:<18s} MAE = {np.abs(p - real).mean():.3f}")
print("Crescimento médio anual da série nos últimos 36 meses: %.2f ppm/ano" % ((y[-1] - y[-37]) / 3))
