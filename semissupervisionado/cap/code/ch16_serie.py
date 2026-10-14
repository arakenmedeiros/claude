import os, urllib.request
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

def baixar(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino): urllib.request.urlretrieve(url, destino)
    return destino

d = pd.read_csv(baixar("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", "dados/daily-min-temperatures.csv"), parse_dates=["Date"])
t, datas = d["Temp"].values, d["Date"]; J = 30
feat = lambda w: [w.mean(), w.std(), w.min(), w.max(), np.polyfit(np.arange(J), w, 1)[0], np.corrcoef(w[:-1], w[1:])[0, 1]]
X, y, ano = [], [], []
for i in range(0, len(t) - J + 1):
    mes = datas[i + J // 2].month
    X.append(feat(t[i:i + J])); y.append(2 if mes in (12, 1, 2) else 0 if mes in (6, 7, 8) else 1); ano.append(datas[i + J // 2].year)
X, y, ano = np.array(X), np.array(y), np.array(ano)
print(f"Temperaturas mínimas diárias (Melbourne, 1981-1990): {len(y)} janelas de {J} dias; classes: 2 = verão (dez-fev), 0 = inverno (jun-ago), 1 = meia-estação (demais meses)")
SORTEIOS = 20
print(f"Acurácia no teste, {SORTEIOS} sorteios, para dois números de rótulos e duas formas de dividir treino e teste")
print("rótulos | divisão do teste                          | reg. logística | 1-vizinho    | Label Spreading (knn, k = 10)")
for por in [10, 50]:
    for nome in ["aleatória (janelas se sobrepõem)", "temporal (treino até 1987, teste de 1988 em diante)"]:
        if nome.startswith("aleat"):
            r0 = np.random.RandomState(0); perm = r0.permutation(len(y)); tr, te = perm[:int(0.7 * len(y))], perm[int(0.7 * len(y)):]
        else:
            tr, te = np.where(ano <= 1987)[0], np.where(ano >= 1988)[0]
        esc = StandardScaler().fit(X[tr]); Xtr, Xte = esc.transform(X[tr]), esc.transform(X[te]); ytr, yte = y[tr], y[te]; res = {"lr": [], "nn": [], "ls": []}
        for s in range(SORTEIOS):
            r = np.random.RandomState(s); idx = np.concatenate([r.choice(np.where(ytr == k)[0], por, replace=False) for k in range(3)])
            yp = np.full(len(ytr), -1); yp[idx] = ytr[idx]
            res["lr"].append(LogisticRegression(max_iter=3000).fit(Xtr[idx], ytr[idx]).score(Xte, yte)); res["nn"].append(KNeighborsClassifier(1).fit(Xtr[idx], ytr[idx]).score(Xte, yte))
            res["ls"].append(LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=200).fit(Xtr, yp).score(Xte, yte))
        print(f"{3 * por:7d} | {nome:49s} | {100 * np.mean(res['lr']):5.1f}% ± {100 * np.std(res['lr']):3.1f} | {100 * np.mean(res['nn']):5.1f}% ± {100 * np.std(res['nn']):3.1f} | {100 * np.mean(res['ls']):5.1f}% ± {100 * np.std(res['ls']):3.1f}")
