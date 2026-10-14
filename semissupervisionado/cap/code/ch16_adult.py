import os, urllib.request
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import f1_score

def baixar(url, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if not os.path.exists(destino): urllib.request.urlretrieve(url, destino)
    return destino

nomes = ["idade", "classe_trab", "fnlwgt", "escolaridade", "anos_estudo", "estado_civil", "ocupacao", "parentesco", "raca", "sexo", "ganho_capital", "perda_capital", "horas", "pais", "renda"]
df = pd.read_csv(baixar("https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv", "dados/adult-all.csv"), header=None, names=nomes)
y = (df["renda"] == ">50K").astype(int).values; df = df.drop(columns=["renda", "escolaridade", "fnlwgt"])
num = ["idade", "anos_estudo", "ganho_capital", "perda_capital", "horas"]; cat = [c for c in df.columns if c not in num]
prep = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat)])
d_tr, d_te, y_tr, y_te = train_test_split(df, y, test_size=0.3, stratify=y, random_state=0)
A = prep.fit_transform(d_tr); B = prep.transform(d_te); A = A.toarray() if hasattr(A, "toarray") else A; B = B.toarray() if hasattr(B, "toarray") else B
print(f"Adult (renda > 50 mil): {len(y)} pessoas, {100 * y.mean():.1f}% na classe positiva. Treino com {len(y_tr)} pessoas; rótulos sorteados ao acaso; 5 sorteios")
print("rótulos | método                                  | F1 da classe positiva | F1 macro")
for n in [100, 500]:
    res = {}
    for s in range(5):
        r = np.random.RandomState(s)
        while True:
            idx = r.choice(len(y_tr), n, replace=False)
            if y_tr[idx].sum() >= 5: break
        yp = np.full(len(y_tr), -1); yp[idx] = y_tr[idx]
        mods = {"reg. logística (pesos balanceados)": LogisticRegression(max_iter=3000, class_weight="balanced").fit(A[idx], y_tr[idx]),
                "gradient boosting (pesos balanceados)": HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(A[idx], y_tr[idx]),
                "gradient boosting + self-training (0,9)": SelfTrainingClassifier(HistGradientBoostingClassifier(random_state=0), threshold=0.9).fit(A, yp)}
        for k, m in mods.items():
            p = m.predict(B); res.setdefault(k, []).append((f1_score(y_te, p), f1_score(y_te, p, average="macro")))
    for k, v in res.items():
        v = np.array(v); print(f"{n:7d} | {k:39s} | {100 * v[:, 0].mean():9.1f}% ± {100 * v[:, 0].std():4.1f}  | {100 * v[:, 1].mean():5.1f}% ± {100 * v[:, 1].std():4.1f}")
