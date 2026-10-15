import numpy as np, torch
from ssl_profundo import *

X, y, X_te, y_te = carregar_fashion()
nomes_cl = ["camiseta", "calça", "pulôver", "vestido", "casaco", "sandália", "camisa", "tênis", "bolsa", "bota"]
moderado = lambda x: aumentar(x, rotacao=10, escala=0.1, deslocamento=0.1)       # aumento "forte" adequado a roupas (Capítulos 9 a 11)
PASSOS, SEMENTES = 2000, 2
metodos = [("só rotulados", "sup"), ("Pseudo-Label", "pseudo"), ("FixMatch (0,95)", "fixmatch"), ("UDA", "uda")]
print(f"Fashion-MNIST, 10.000 imagens sem rótulo, CNN, {PASSOS} passos, {SEMENTES} sorteios. Acerto no teste (média ± desvio)")
print("rótulos (por classe) | " + " | ".join(f"{n:16s}" for n, _ in metodos))
modelos = {}
for por in [10, 30, 100]:
    linha = []
    for nome, m in metodos:
        v = []
        for s in range(SEMENTES):
            idx = sortear_rotulados(y, por, s); u = np.random.RandomState(100 + s).choice(np.setdiff1d(np.arange(len(X)), idx), 10000, replace=False)
            aluno, prof, h = treinar_ssl(m, X[idx], y[idx], X[u], passos=PASSOS, forte=moderado, fraco=aumento_fraco, semente=s, X_teste=X_te, y_teste=y_te, relatar=(PASSOS,))
            v.append(h[0][1])
            if por == 10 and s == 0: modelos[m] = aluno
        linha.append(f"{100 * np.mean(v):5.1f}% ± {100 * np.std(v):4.1f}  ")
    print(f"{10 * por:5d} ({por:3d})         | " + " | ".join(linha))

print("\nAcerto por categoria no teste (100 rótulos, um sorteio)")
print("categoria  | " + " | ".join(f"{n:16s}" for n, m in metodos if m in modelos))
pred = {}
for m, aluno in modelos.items():
    aluno.eval()
    with torch.no_grad(): pred[m] = torch.cat([aluno(X_te[i:i + 2000]).argmax(1) for i in range(0, len(X_te), 2000)])
for d in range(10): print(f"{nomes_cl[d]:10s} | " + " | ".join(f"{100 * (pred[m][y_te == d] == d).float().mean():15.1f}% " for n, m in metodos if m in modelos))
