# Capítulo 6 — Métodos multifidelidade: Successive Halving, Hyperband e BOHB
# Bloco 7. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
# --- blocos anteriores deste capítulo (encadeados) ---
import math
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def carregar_dados(semente=0):
    X, y = load_digits(return_X_y=True)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=semente)
    escala = StandardScaler().fit(Xtr)
    return escala.transform(Xtr), escala.transform(Xva), ytr, yva

class Candidato:
    """Uma configuração que pode ser treinada aos poucos (pausar e retomar)."""
    def __init__(self, rng, dados):
        self.hp = {"alpha": 10 ** rng.uniform(-8, 1), "eta0": 10 ** rng.uniform(-6, 1),
                   "loss": str(rng.choice(["hinge", "log_loss", "modified_huber"]))}
        self.modelo = SGDClassifier(loss=self.hp["loss"], alpha=self.hp["alpha"], learning_rate="constant",
                                    eta0=self.hp["eta0"], random_state=int(rng.integers(1e6)))
        self.dados, self.epocas, self.nota, self.curva = dados, 0, 0.0, []

    def treinar_ate(self, epocas):
        Xtr, Xva, ytr, yva = self.dados
        while self.epocas < epocas:
            self.modelo.partial_fit(Xtr, ytr, classes=np.arange(10))      # uma época
            self.epocas += 1
            self.nota = self.modelo.score(Xva, yva)                        # acurácia na validação
            self.curva.append(self.nota)
        return self.nota
def sucessivo(candidatos, r_min, eta=3, r_max=None, verbose=False):
    """Successive Halving: treina todos com r_min épocas, mantém o melhor 1/eta e multiplica o orçamento por eta."""
    r, gasto = r_min, 0
    while True:
        for c in candidatos:
            antes = c.epocas
            c.treinar_ate(int(r))
            gasto += c.epocas - antes                                      # só contamos as épocas NOVAS
        if verbose:
            print(f"  {len(candidatos):2d} candidatos com {int(r):2d} épocas | melhor acurácia = {max(c.nota for c in candidatos):.3f}")
        if len(candidatos) <= 1 or (r_max and r >= r_max):
            break
        candidatos = sorted(candidatos, key=lambda c: -c.nota)[:max(1, len(candidatos) // eta)]
        r *= eta
    return max(candidatos, key=lambda c: c.nota), gasto
dados = carregar_dados()
rng = np.random.default_rng(0)
candidatos = [Candidato(rng, dados) for _ in range(81)]

print("Successive Halving com 81 candidatos (η = 3):")
vencedor, gasto = sucessivo(candidatos, r_min=1, eta=3, r_max=81, verbose=True)
print(f"\nÉpocas de treino gastas: {gasto}")
print(f"Treinar os 81 candidatos por 81 épocas custaria: {81 * 81}")
print(f"Vencedor: {vencedor.hp['loss']}, alpha = {vencedor.hp['alpha']:.2e}, eta0 = {vencedor.hp['eta0']:.2e}")
rng = np.random.default_rng(4)
dados = carregar_dados(0)
configs = [Candidato(rng, dados) for _ in range(40)]
for c in configs:
    c.treinar_ate(81)

final = np.array([c.curva[-1] for c in configs])
apos3 = np.array([c.curva[2] for c in configs])
posto_final = (-final).argsort().argsort() + 1          # 1 = melhor
posto_apos3 = (-apos3).argsort().argsort() + 1

print("As 3 melhores no final (epoch 81) e o posto delas após 3 épocas:")
for i in np.argsort(-final)[:3]:
    print(f"  configuração {i:2d}: posto final = {posto_final[i]} | posto após 3 épocas = {posto_apos3[i]:2d} de 40")
corte = len(configs) // 3
print(f"Se cortássemos para as {corte} melhores após 3 épocas, "
      f"{sum(posto_apos3[i] <= corte for i in np.argsort(-final)[:3])} das 3 campeãs sobreviveriam.")
def hyperband(rng, dados, R=81, eta=3):
    s_max = int(math.log(R) / math.log(eta) + 1e-9)
    B = (s_max + 1) * R
    melhor, gasto, n_config = None, 0, 0
    for s in range(s_max, -1, -1):
        n = int(math.ceil(B / R * eta ** s / (s + 1)))        # nº de configurações do torneio
        r = R * eta ** (-s)                                    # recurso inicial
        candidatos = [Candidato(rng, dados) for _ in range(n)]
        n_config += n
        vencedor, g = sucessivo(candidatos, r, eta, R)
        gasto += g
        if melhor is None or vencedor.nota > melhor.nota:
            melhor = vencedor
    return melhor, gasto, n_config

def busca_aleatoria(rng, dados, orcamento, R=81):
    """Linha de base: treina cada configuração até R épocas, até acabar o orçamento."""
    melhor, gasto = 0.0, 0
    while gasto + R <= orcamento:
        c = Candidato(rng, dados)
        c.treinar_ate(R)
        gasto += R
        melhor = max(melhor, c.nota)
    return melhor, gasto
# --- fim dos blocos anteriores ---
resultados = []
for s in range(12):
    d = carregar_dados(0)
    sh, gasto_sh = sucessivo([Candidato(np.random.default_rng(2000 + s * 1000 + i), d) for i in range(81)], 1, 3, 81)
    aleat_sh, _ = busca_aleatoria(np.random.default_rng(3000 + s), d, gasto_sh)
    hb, gasto_hb, n_cfg = hyperband(np.random.default_rng(s), d)
    aleat_hb, _ = busca_aleatoria(np.random.default_rng(1000 + s), d, gasto_hb)
    resultados.append((sh.nota, aleat_sh, hb.nota, aleat_hb, gasto_sh, gasto_hb, n_cfg))

r = np.array(resultados)
print(f"Orçamento de {int(r[0,4])} épocas:")
print(f"  Successive Halving (81 configurações): {r[:,0].mean():.3f} (±{r[:,0].std():.3f})")
print(f"  Aleatória (só {int(r[0,4] // 81)} configurações)         : {r[:,1].mean():.3f} (±{r[:,1].std():.3f})")
print(f"Orçamento de {int(r[0,5])} épocas:")
print(f"  Hyperband ({int(r[0,6])} configurações)      : {r[:,2].mean():.3f} (±{r[:,2].std():.3f})")
print(f"  Aleatória (só {int(r[0,5] // 81)} configurações)      : {r[:,3].mean():.3f} (±{r[:,3].std():.3f})")
