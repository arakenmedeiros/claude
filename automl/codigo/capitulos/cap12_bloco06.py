# Capítulo 12 — AutoML para séries temporais, texto e imagens
# Bloco 6. Execute a partir da pasta 'codigo/' (os resultados vão para 'resultados/').
import itertools
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

r = np.random.default_rng(0)
TOPICOS = [[f"t{j}p{k}" for k in range(60)] for j in range(12)]; COMUNS = [f"c{k}" for k in range(80)]
docs, fatos = [], []
for d in range(150):                                              # 150 documentos de 400 palavras, com 2 assuntos
    tops = r.choice(12, 2, replace=False); pal = []
    for _ in range(400):
        u = r.random(); pal.append(r.choice(COMUNS) if u < 0.5 else r.choice(TOPICOS[tops[0 if u < 0.8 else 1]]))
    pos = int(r.integers(20, 380)); ent = f"ent{d % 15}x"           # o código de entidade é compartilhado por 10 documentos
    pal[pos:pos + 4] = [ent, "prazo", "é", f"v{int(r.integers(100, 999))}"]; docs.append(pal); fatos.append((d, pos, ent))

def consulta(f):                                                    # código da entidade + UMA pista do assunto do documento
    d, pos, ent = f; assuntos = [t for t in range(12) if sum(w in TOPICOS[t] for w in docs[d]) > 40]
    return " ".join([ent, str(r.choice(TOPICOS[r.choice(assuntos)]))]), f
Q = [consulta(f) for f in fatos for _ in range(2)]
ordem = r.permutation(len(Q)); dev, teste = [Q[i] for i in ordem[:150]], [Q[i] for i in ordem[150:]]

def revocacao(consultas, trecho, k, ngramas, sublinear):
    passo = max(1, int(trecho * 0.8)); pedacos, dono = [], []         # trechos com 20% de sobreposição
    for d, pal in enumerate(docs):
        for s in range(0, max(1, len(pal) - trecho + passo), passo):
            pedacos.append(" ".join(pal[s:s + trecho])); dono.append((d, s, min(s + trecho, len(pal))))
    vec = TfidfVectorizer(ngram_range=ngramas, sublinear_tf=sublinear).fit(pedacos); M = vec.transform(pedacos); acertos = 0
    for q, (d, pos, ent) in consultas:
        sim = (M @ vec.transform([q]).T).toarray().ravel()
        acertos += any(dono[j][0] == d and dono[j][1] <= pos < dono[j][2] for j in np.argsort(-sim)[:k])
    return acertos / len(consultas)

LIMITE = 1200                                                       # palavras de contexto entregues ao modelo (trecho x k)
tabela = [(cfg, revocacao(dev, *cfg), cfg[0] * cfg[1])
          for cfg in itertools.product([50, 100, 200, 400], [1, 2, 3, 5, 8], [(1, 1), (1, 2)], [True, False])]
viaveis = [x for x in tabela if x[2] <= LIMITE]
print(f"Configurações: {len(tabela)} | viáveis (contexto <= {LIMITE} palavras): {len(viaveis)}")
def mostra(rotulo, x):
    (trecho, k, ng, sub), rev, ctx = x
    print(f"{rotulo:<27s} trecho={trecho:<3d} k={k} ng={ng} ctx={ctx:<4d} dev={rev:.3f} teste={revocacao(teste, trecho, k, ng, sub):.3f}")
mostra("melhor viável (dev)", max(viaveis, key=lambda x: x[1]))
mostra("melhor sem restrição (dev)", max(tabela, key=lambda x: x[1]))
print(f"Erro padrão aproximado do teste ({len(teste)} consultas): {np.sqrt(0.7 * 0.3 / len(teste)):.3f}")
