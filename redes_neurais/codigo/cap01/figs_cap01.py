"""Figuras conceituais do Capítulo 1 (todas geradas por código; nada de imagens prontas)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from rnbook.estilo import plt, COR, salvar
from rnbook.desenho import caixa, seta, diagrama_rede

# ---- f1_camadas: cinco conceitos distintos
fig, ax = plt.subplots(figsize=(10, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.9); ax.axis("off"); ax.grid(False)
itens = [("Neurônio\nartificial", "$z=\\mathbf{w}^T\\mathbf{x}+b$\n$a=\\phi(z)$", COR["cya"]),
         ("Arquitetura\nda rede", "camadas, neurônios,\nconexões", COR["ind"]),
         ("Função\nde perda", "$E(\\theta)$: mede o erro\nda rede", COR["amb"]),
         ("Algoritmo de\ndiferenciação", "backpropagation:\ncalcula $\\nabla E$", COR["red"]),
         ("Método de\notimização", "usa $\\nabla E$ para\natualizar $\\theta$", COR["grn"])]
w = 1.7
for i, (t, d, c) in enumerate(itens):
    x = 0.15 + i * 1.97
    caixa(ax, x, 1.55, w, 1.6, t, cor=c, fc="white", fs=10.5, bold=True)
    ax.text(x + w / 2, 0.95, d, ha="center", va="center", fontsize=9, color=COR["ink"])
    ax.add_patch(plt.Circle((x + w / 2, 3.45), 0.2, color=c)); ax.text(x + w / 2, 3.45, str(i + 1), color="white", ha="center", va="center", fontweight="bold")
    if i < 4: seta(ax, (x + w + 0.02, 2.35), (x + 1.97 - 0.02, 2.35), cor=COR["gry"])
ax.text(5, 0.2, "Cinco ideias diferentes. Misturá-las é uma das maiores fontes de confusão para quem está começando.", ha="center", fontsize=9.5, style="italic", color=COR["gry"])
salvar(fig, "f1_camadas")

# ---- f1_neuronio: analogia biológica x artificial
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
a = ax[0]; a.set_xlim(0, 10); a.set_ylim(0, 6); a.axis("off"); a.grid(False); a.set_title("Neurônio biológico (esquema)"); a.set_aspect("equal")
a.add_patch(plt.Circle((5, 3), 1.0, fc="#FDE68A", ec=COR["amb"], lw=2)); a.text(5, 3, "corpo\ncelular", ha="center", va="center", fontsize=9)
for ang, (dx, dy) in zip(range(4), [(-3.5, 2), (-3.5, 0.5), (-3.5, -0.5), (-3.2, -2)]):
    a.plot([5 - 1, 5 + dx * 0.55 + 0.5], [3, 3 + dy * 0.7], color=COR["gry"], lw=2)
    a.plot([5 + dx * 0.55 + 0.5, 5 + dx * 0.9 + 0.5], [3 + dy * 0.7, 3 + dy * 1.0], color=COR["gry"], lw=1.2)
a.text(1.1, 5.2, "dendritos\n(recebem sinais)", fontsize=9, ha="center"); a.plot([6, 9.4], [3, 3], color=COR["gry"], lw=3); a.text(7.8, 3.5, "axônio\n(envia o sinal)", ha="center", fontsize=9)
b = ax[1]; b.set_xlim(0, 10); b.set_ylim(0, 6); b.axis("off"); b.grid(False); b.set_title("Neurônio artificial"); b.set_aspect("equal")
ys = [4.6, 3.4, 2.2, 1.0]
for i, y in enumerate(ys):
    b.text(0.4, y, f"$x_{i+1}$", ha="center", va="center", fontsize=11); seta(b, (0.8, y), (4.0, 3.0), cor=COR["gry"])
    b.text(2.2, 3.0 + (y - 3.0) * 0.45 + 0.28, f"$w_{i+1}$", fontsize=10, color=COR["ind"])
b.add_patch(plt.Circle((4.8, 3), 0.85, fc=COR["ind"], ec="white")); b.text(4.8, 3.0, "$\\Sigma$ ", color="white", fontsize=15, ha="center", va="center")
b.text(4.8, 4.35, "$b$ (viés)", ha="center", fontsize=10, color=COR["amb"]); seta(b, (4.8, 4.2), (4.8, 3.9), cor=COR["amb"])
b.add_patch(plt.Circle((7.0, 3), 0.7, fc=COR["grn"], ec="white")); b.text(7.0, 3.0, "$\\phi$", color="white", fontsize=15, ha="center", va="center")
seta(b, (5.7, 3), (6.2, 3)); seta(b, (7.75, 3), (9.4, 3)); b.text(9.6, 3, "$a$", fontsize=12, va="center")
b.text(4.8, 1.75, "$z=\\sum w_i x_i + b$", ha="center", fontsize=9.5); b.text(7.0, 1.75, "$a=\\phi(z)$", ha="center", fontsize=9.5)
salvar(fig, "f1_neuronio")

# ---- f1_rede_dim: rede 3-4-2 com dimensões
fig, ax = plt.subplots(figsize=(8.4, 4.0))
diagrama_rede(ax, [3, 4, 2], rotulos=[["x₁", "x₂", "x₃"], ["a₁", "a₂", "a₃", "a₄"], ["y₁", "y₂"]],
              titulos=["entrada\n$n_0=3$", "camada oculta\n$n_1=4$", "saída\n$n_2=2$"], passo_x=2.6, raio=0.17)
ax.text(1.3, -1.55, "$W^{(1)}$: 4×3,  $b^{(1)}$: 4", ha="center", fontsize=10, color=COR["ind"])
ax.text(3.9, -1.55, "$W^{(2)}$: 2×4,  $b^{(2)}$: 2", ha="center", fontsize=10, color=COR["ind"])
ax.text(2.6, -2.05, "total de parâmetros: $(4\\cdot3+4)+(2\\cdot4+2)=26$", ha="center", fontsize=10, color=COR["ink"])
ax.set_ylim(-2.4, 2.2)
salvar(fig, "f1_rede_dim")

# ---- f1_linha_tempo
eventos = [(1943, "McCulloch–Pitts\nneurônio lógico", COR["cya"], 1), (1949, "Hebb\nregra de\naprendizado", COR["gry"], -1), (1958, "Rosenblatt\nperceptron", COR["ind"], 1),
           (1960, "Widrow–Hoff\nADALINE / LMS", COR["grn"], -1), (1969, "Minsky–Papert\nlimites do\nperceptron", COR["red"], 1), (1974, "Werbos\n(tese)", COR["amb"], -1),
           (1986, "Rumelhart,\nHinton, Williams\nbackpropagation", COR["ind"], 1), (1988, "Fahlman: Quickprop\nJacobs: Delta-Bar-Delta", COR["vio"], -1),
           (1989, "Cybenko; Hornik et al.\naproximação universal", COR["grn"], 1), (1993, "Riedmiller–Braun\nRPROP", COR["pnk"], -1),
           (1994, "Hagan–Menhaj\nLM em redes", COR["amb"], 1), (2003, "Igel–Hüsken\niRPROP", COR["cya"], -1)]
fig, ax = plt.subplots(figsize=(11, 4.4)); ax.axhline(0, color=COR["ink"], lw=2); ax.axis("off"); ax.grid(False)
anos = np.array([e[0] for e in eventos]); xs = np.interp(np.arange(len(eventos)), [0, len(eventos) - 1], [0, 1]) * 22
for (ano, txt, c, lado), x in zip(eventos, xs):
    ax.plot([x, x], [0, 0.55 * lado], color=c, lw=1.5); ax.scatter([x], [0], s=60, color=c, zorder=3)
    ax.text(x, 0.62 * lado, txt, ha="center", va="bottom" if lado > 0 else "top", fontsize=7.8, color=COR["ink"])
    ax.text(x, -0.17 * lado if lado > 0 else 0.17, str(ano), ha="center", va="center", fontsize=8.5, fontweight="bold", color=c)
ax.set_xlim(-1.2, 23.2); ax.set_ylim(-1.5, 1.7)
ax.set_title("Alguns marcos da história que levam aos algoritmos deste livro (espaçamento não proporcional ao tempo)", fontsize=10)
salvar(fig, "f1_linha_tempo")

# ---- f1_mapa: mapa do livro
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off"); ax.grid(False)
caixa(ax, 0.1, 3.0, 4.6, 0.9, "PARTE 1 — Fundamentos\nCap. 1 Introdução · Cap. 2 Matemática e Python", cor=COR["ind"], fs=9.5, bold=True)
caixa(ax, 5.3, 3.0, 4.6, 0.9, "PARTE 2 — Treinamento de redes\nPerceptron → Backpropagation → 4 famílias de algoritmos", cor=COR["cya"], fs=9.5, bold=True)
caps = [("3", "Perceptron", COR["ind"]), ("4", "Backprop\ne MLP", COR["red"]), ("5", "QPROP", COR["amb"]), ("6", "RPROP\n+ 3 variantes", COR["grn"]), ("7", "Levenberg–\nMarquardt", COR["vio"]), ("8", "DBD e EDBD", COR["pnk"])]
for i, (n, t, c) in enumerate(caps):
    x = 0.15 + i * 1.64
    caixa(ax, x, 1.1, 1.45, 1.2, f"Cap. {n}\n{t}", cor=c, fc="white", fs=9, bold=True)
    if i < 5: seta(ax, (x + 1.47, 1.7), (x + 1.62, 1.7), cor=COR["gry"], ms=9)
ax.text(5, 0.55, "Do neurônio isolado ao treinamento eficiente: cada capítulo acrescenta uma ideia sobre o anterior", ha="center", fontsize=10, style="italic", color=COR["gry"])
ax.text(2.6, 0.1, "gradiente, derivada, Newton", ha="center", fontsize=8.5, color=COR["gry"]); ax.text(7.8, 0.1, "gradientes locais → passos adaptativos → curvatura", ha="center", fontsize=8.5, color=COR["gry"])
salvar(fig, "f1_mapa")
print("figuras do cap. 1 ok")
