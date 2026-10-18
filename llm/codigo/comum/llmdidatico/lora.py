"""lora.py: ajuste fino, LoRA e quantização por blocos, em NumPy (Capítulo 12).

Tudo aqui é DIDÁTICO: opera sobre o MLPLM do pacote (poucos milhares de parâmetros), com gradientes escritos à mão.
Não há LLM real. Convenção do livro (artigo de LoRA): W' = W + (alfa/r) * B @ A, com W (d_saida x d_entrada),
B (d_saida x r), A (r x d_entrada). Como o MLPLM guarda as matrizes como (entrada x saída) e calcula X @ W,
aqui a camada computa  Y = X @ W + (alfa/r) * (X @ A.T) @ B.T,  que é a mesma coisa transposta.
"""
import copy
import math
import random
import numpy as np
from .modelos import MLPLM, log_softmax
from . import corpus


# ---------------------------------------------------------------- tarefa nova (mudança de domínio) com o mesmo vocabulário
def frases_dominio_novo(n=2000, seed=0):
    """Frases com o MESMO vocabulário do corpus original, mas outra regra: cada sujeito passa a usar os verbos do sujeito
    dois lugares adiante na lista original (ex.: 'o estudante corrige a prova'), e o complemento é quase sempre presente
    e vem de um conjunto menor. O conhecimento antigo e o novo CONFLITAM em parte: bom para medir esquecimento."""
    rnd = random.Random(seed)
    S = corpus._SUJ
    comps = corpus._COMP[:5]
    out = []
    for _ in range(n):
        i = rnd.randrange(len(S))
        suj = S[i][0]
        verbos = S[(i + 2) % len(S)][1]
        v = rnd.choice(verbos)
        o = rnd.choice(corpus._OBJ[v])
        toks = suj.split() + [v] + o.split()
        if rnd.random() < 0.95:
            toks += rnd.choice(comps).split()
        out.append(toks + ["."])
    return out


def vocabulario(*conjuntos):
    v = {"<s>"}
    for c in conjuntos:
        for f in c:
            v.update(f)
    return ["<s>"] + sorted(v - {"<s>"})


# ---------------------------------------------------------------- avaliação (funciona para MLPLM e MLPLMLoRA)
def perda_media(m, frases):
    X, y = m.dados(frases); _, _, z = m.frente(X)
    return float(-log_softmax(z)[np.arange(len(y)), y].mean())


def acertos_token(m, frases):
    """Vetor 0/1: o token mais provável é o correto? (um elemento por posição prevista)"""
    X, y = m.dados(frases); _, _, z = m.frente(X)
    return (z.argmax(1) == y).astype(float)


# ---------------------------------------------------------------- ajuste completo (ou parcial) de um MLPLM
def ajuste_completo(base, frases, epocas=10, lote=64, lr=0.005, seed=0, treinaveis=None, val=None, replay=None, p_replay=0.0):
    """Copia `base` e continua o treino nos dados novos. Adam REINICIADO (estado do otimizador zerado).
    treinaveis: nomes dos parâmetros a atualizar (None = todos). Retorna (modelo, historico dict).
    replay/p_replay: mistura uma fração de frases do domínio antigo (ensaio de mitigação do esquecimento)."""
    m = copy.deepcopy(base)
    m.m = {k: np.zeros_like(v) for k, v in m.P.items()}; m.v = {k: np.zeros_like(v) for k, v in m.P.items()}; m.t = 0
    nomes = list(m.P) if treinaveis is None else list(treinaveis)
    dados = list(frases)
    if replay is not None and p_replay > 0:
        k = int(p_replay * len(frases)); dados += list(replay[:k])
    X, y = m.dados(dados); rng = np.random.default_rng(seed)
    hist = {"treino": [], "val": []}
    for ep in range(epocas):
        ordem = rng.permutation(len(X)); tot = 0.0
        for i in range(0, len(X), lote):
            b = ordem[i:i + lote]; loss, g = m.perda_grad(X[b], y[b])
            g = {k: (g[k] if k in nomes else np.zeros_like(g[k])) for k in g}
            m.passo(g, lr); tot += loss * len(b)
        hist["treino"].append(tot / len(X))
        if val is not None: hist["val"].append(perda_media(m, val))
    return m, hist


# ---------------------------------------------------------------- LoRA com retropropagação manual
class MLPLMLoRA:
    """Envolve um MLPLM CONGELADO e treina apenas adaptadores de posto r nas matrizes `alvos` (W1 e/ou W2).
    Saída da camada: X @ W + (alfa/r) * (X @ A.T) @ B.T ; A ~ N(0, 1/d_entrada), B = 0 (logo, no início, o modelo é idêntico ao base)."""

    def __init__(self, base, r=4, alfa=8.0, alvos=("W1", "W2"), seed=0):
        self.base = base
        self.P0 = {k: v.copy() for k, v in base.P.items()}           # cópia congelada dos pesos base
        self.vocab, self.idx, self.V, self.ctx, self.d, self.h = base.vocab, base.idx, base.V, base.ctx, base.d, base.h
        self.r, self.alfa, self.alvos = r, float(alfa), tuple(alvos)
        self.escala = self.alfa / self.r
        rng = np.random.default_rng(seed)
        self.A, self.B = {}, {}
        for k in self.alvos:
            d_in, d_out = self.P0[k].shape
            self.A[k] = rng.normal(0, 1 / math.sqrt(d_in), (r, d_in))
            self.B[k] = np.zeros((d_out, r))
        self._m = {}; self._v = {}; self._t = 0
        for k in self.alvos:
            for nome, M in (("A", self.A[k]), ("B", self.B[k])):
                self._m[(nome, k)] = np.zeros_like(M); self._v[(nome, k)] = np.zeros_like(M)

    # contagem
    def n_treinaveis(self):
        return int(sum(self.A[k].size + self.B[k].size for k in self.alvos))

    def n_total_base(self):
        return int(sum(v.size for v in self.P0.values()))

    def dados(self, frases):
        return self.base.dados(frases)

    def _lin(self, X, k):
        """Y = X W + escala * (X A^T) B^T ; devolve também Z = X A^T para o gradiente."""
        Z = X @ self.A[k].T
        return X @ self.P0[k] + self.escala * (Z @ self.B[k].T), Z

    def frente(self, X, _cache=False):
        P0 = self.P0; e = P0["E"][X].reshape(len(X), -1)
        pre, Z1 = self._lin(e, "W1") if "W1" in self.alvos else (e @ P0["W1"], None)
        a = np.tanh(pre + P0["b1"])
        z, Z2 = self._lin(a, "W2") if "W2" in self.alvos else (a @ P0["W2"], None)
        z = z + P0["b2"]
        if _cache: return e, a, z, Z1, Z2
        return e, a, z

    def perda_grad(self, X, y):
        e, a, z, Z1, Z2 = self.frente(X, True); N = len(X)
        lp = log_softmax(z); loss = -lp[np.arange(N), y].mean()
        dz = np.exp(lp); dz[np.arange(N), y] -= 1; dz /= N
        g = {}
        s = self.escala
        if "W2" in self.alvos:
            g[("B", "W2")] = s * dz.T @ Z2                                   # (V x r)
            dZ2 = s * dz @ self.B["W2"]                                      # (N x r)
            g[("A", "W2")] = dZ2.T @ a                                       # (r x h)
            da = dz @ self.P0["W2"].T + dZ2 @ self.A["W2"]
        else:
            da = dz @ self.P0["W2"].T
        dpre = da * (1 - a ** 2)
        if "W1" in self.alvos:
            g[("B", "W1")] = s * dpre.T @ Z1
            dZ1 = s * dpre @ self.B["W1"]
            g[("A", "W1")] = dZ1.T @ e
        return loss, g

    def passo(self, g, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for key, grad in g.items():
            nome, k = key
            M = self.A[k] if nome == "A" else self.B[k]
            self._m[key] = b1 * self._m[key] + (1 - b1) * grad; self._v[key] = b2 * self._v[key] + (1 - b2) * grad ** 2
            mh = self._m[key] / (1 - b1 ** self._t); vh = self._v[key] / (1 - b2 ** self._t)
            M -= lr * mh / (np.sqrt(vh) + eps)                               # atualização no próprio array

    def treinar(self, frases, epocas=10, lote=64, lr=0.01, seed=0, val=None):
        X, y = self.dados(frases); rng = np.random.default_rng(seed)
        hist = {"treino": [], "val": []}
        for ep in range(epocas):
            ordem = rng.permutation(len(X)); tot = 0.0
            for i in range(0, len(X), lote):
                b = ordem[i:i + lote]; loss, g = self.perda_grad(X[b], y[b]); self.passo(g, lr); tot += loss * len(b)
            hist["treino"].append(tot / len(X))
            if val is not None: hist["val"].append(perda_media(self, val))
        return hist

    def perplexidade(self, frases):
        return math.exp(perda_media(self, frases))

    # mesclagem: W' = W + (alfa/r) * (B A)^T   (transposto, pela convenção X @ W)
    def pesos_mesclados(self):
        P = {k: v.copy() for k, v in self.P0.items()}
        for k in self.alvos:
            P[k] = P[k] + self.escala * (self.B[k] @ self.A[k]).T
        return P

    def mesclar(self):
        """Devolve um MLPLM comum (sem adaptadores) com os pesos mesclados."""
        m = copy.deepcopy(self.base); m.P = self.pesos_mesclados()
        return m

    def delta_W(self, k):
        return self.escala * (self.B[k] @ self.A[k]).T


# ---------------------------------------------------------------- contagem de parâmetros e memória (estimativas)
def parametros_lora_matriz(d_in, d_out, r):
    return r * (d_in + d_out)


def parametros_lora_transformer(d, n_camadas, r, d_ff=None, alvos=("q", "k", "v", "o"), mlp_gated=True):
    """Contagem para um Transformer denso HIPOTÉTICO (todas as projeções com d x d; MLP com d x d_ff).
    alvos: subconjunto de q,k,v,o,mlp. Ignora variações como atenção por grupos de consultas."""
    d_ff = d_ff or 4 * d
    por_camada = 0
    for a in alvos:
        if a in ("q", "k", "v", "o"): por_camada += parametros_lora_matriz(d, d, r)
        elif a == "mlp": por_camada += (3 if mlp_gated else 2) * parametros_lora_matriz(d, d_ff, r)
    return por_camada * n_camadas


def bits_por_parametro(bits=4, bloco=64, bits_escala=32, dupla=False, bloco2=256, bits_escala2=32):
    """Custo médio em bits por parâmetro (códigos + constantes de escala). Dupla quantização: as escalas viram 8 bits,
    com uma escala de 32 bits por grupo de `bloco2` escalas (contas do artigo de QLoRA para bloco=64: 0,5 -> 0,127 bit)."""
    if not dupla:
        return bits + bits_escala / bloco
    return bits + 8 / bloco + bits_escala2 / (bloco * bloco2)


def estimativa_memoria_gb(N, n_adapt=0, modo="completo", bits_base=16, bloco=64, dupla=False):
    """Memória (GB = 1e9 bytes) de PESOS + GRADIENTES + OTIMIZADOR, sem ativações.
    completo: 16 B/parâmetro (2 pesos + 2 gradiente + 4 mestre + 4 + 4 do Adam).
    lora: base congelada em bits_base (sem gradiente/otimizador) + 16 B por parâmetro de adaptador.
    qlora: base em 4 bits (com constantes) + 16 B por parâmetro de adaptador."""
    if modo == "completo":
        comp = {"pesos+gradientes+otimizador": 16 * N}
    elif modo == "lora":
        comp = {"base congelada": bits_base / 8 * N, "adaptadores (com otimizador)": 16 * n_adapt}
    elif modo == "qlora":
        comp = {"base 4 bits quantizada": bits_por_parametro(4, bloco, dupla=dupla) / 8 * N, "adaptadores (com otimizador)": 16 * n_adapt}
    else:
        raise ValueError(modo)
    return {k: v / 1e9 for k, v in comp.items()}


# ---------------------------------------------------------------- quantização por blocos (absmax)
def quantizar_blocos(w, bits=8, bloco=64):
    """Quantização simétrica por blocos: para cada bloco, escala = max|w| / qmax; q = round(w / escala).
    Retorna dict com os códigos inteiros (int8), as escalas (float32), a forma original e o preenchimento."""
    w = np.asarray(w, dtype=np.float64); forma = w.shape; v = w.ravel()
    pad = (-len(v)) % bloco
    vb = np.concatenate([v, np.zeros(pad)]).reshape(-1, bloco)
    qmax = 2 ** (bits - 1) - 1
    escala = np.abs(vb).max(1) / qmax
    escala = np.where(escala == 0, 1.0, escala).astype(np.float32)
    q = np.clip(np.rint(vb / escala[:, None]), -qmax, qmax).astype(np.int8)
    return {"q": q, "escala": escala, "forma": forma, "pad": pad, "bits": bits, "bloco": bloco}


def desquantizar_blocos(d):
    v = (d["q"].astype(np.float64) * d["escala"].astype(np.float64)[:, None]).ravel()
    if d["pad"]: v = v[:-d["pad"]]
    return v.reshape(d["forma"])


def livro_normal(bits=4, deslocamento=0.9677):
    """Livro de códigos de 2**bits valores inspirado no NormalFloat (NF4) descrito na literatura: quantis de uma normal,
    assimétrico (um lado a mais) para conter o ZERO exato, normalizado para [-1, 1]. É uma construção DIDÁTICA e
    aproximada; não garante coincidir bit a bit com a tabela de nenhuma biblioteca. `deslocamento` evita quantis infinitos."""
    from scipy.stats import norm
    n = 2 ** bits
    pos = norm.ppf(np.linspace(deslocamento, 0.5, n // 2 + 1)[:-1])         # n/2 valores positivos
    neg = -norm.ppf(np.linspace(deslocamento, 0.5, n // 2)[:-1])            # n/2 - 1 valores negativos
    v = np.sort(np.concatenate([pos, [0.0], neg]))
    return v / np.abs(v).max()


def quantizar_livro(w, livro, bloco=64):
    """Normaliza cada bloco por seu absmax e guarda o índice do valor mais próximo do livro de códigos."""
    w = np.asarray(w, dtype=np.float64); forma = w.shape; v = w.ravel()
    pad = (-len(v)) % bloco
    vb = np.concatenate([v, np.zeros(pad)]).reshape(-1, bloco)
    am = np.abs(vb).max(1); am = np.where(am == 0, 1.0, am).astype(np.float32)
    x = vb / am[:, None]
    idx = np.abs(x[..., None] - livro[None, None, :]).argmin(-1).astype(np.uint8)
    return {"idx": idx, "absmax": am, "livro": livro, "forma": forma, "pad": pad, "bloco": bloco}


def desquantizar_livro(d):
    v = (d["livro"][d["idx"]] * d["absmax"].astype(np.float64)[:, None]).ravel()
    if d["pad"]: v = v[:-d["pad"]]
    return v.reshape(d["forma"])


def dupla_quantizacao(escalas, bloco2=256):
    """Quantiza as próprias escalas (positivas) em 8 bits: subtrai a média, aplica absmax por grupos de bloco2.
    Retorna as escalas reconstruídas (para medir o efeito) e o erro relativo máximo."""
    c = np.asarray(escalas, dtype=np.float64); media = c.mean()
    d = quantizar_blocos(c - media, bits=8, bloco=bloco2)
    rec = desquantizar_blocos(d) + media
    return rec, float(np.max(np.abs(rec - c) / np.maximum(np.abs(c), 1e-12)))


def erro_reconstrucao(w, w_hat):
    w = np.asarray(w, float); w_hat = np.asarray(w_hat, float); e = w - w_hat
    rmse = float(np.sqrt((e ** 2).mean())); rel = float(np.linalg.norm(e) / (np.linalg.norm(w) + 1e-30))
    return {"rmse": rmse, "erro_relativo": rel, "erro_max": float(np.abs(e).max()), "sqnr_db": float(-20 * np.log10(rel + 1e-30))}


def quantizar_modelo(m, bits=8, bloco=64, livro=None, matrizes=("W1", "W2", "E")):
    """Cópia do MLPLM com as matrizes indicadas passadas por quantização + desquantização (quantização pós-treino simulada)."""
    q = copy.deepcopy(m)
    for k in matrizes:
        if livro is None: q.P[k] = desquantizar_blocos(quantizar_blocos(m.P[k], bits, bloco))
        else: q.P[k] = desquantizar_livro(quantizar_livro(m.P[k], livro, bloco))
    return q


# ---------------------------------------------------------------- dados de ajuste: formato, divisão, vazamento, privacidade
def registro_instrucao(texto, rotulo, categorias):
    """Um exemplo no formato instrução-resposta (campos em português; ver o Capítulo 12)."""
    return {"instrucao": "Classifique a mensagem do cliente em UMA das categorias: " + ", ".join(categorias) + ".",
            "entrada": texto, "resposta": rotulo}


def normalizar_texto(s):
    import re, unicodedata
    s = unicodedata.normalize("NFKD", s.lower()); s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def taxa_vazamento(treino_textos, teste_textos):
    """Fração dos textos de teste cuja forma normalizada aparece EXATAMENTE no treino (vazamento por duplicata)."""
    t = {normalizar_texto(x) for x in treino_textos}
    return float(np.mean([normalizar_texto(x) in t for x in teste_textos]))


def divisao_por_hash(textos, fracoes=(0.8, 0.1, 0.1), sal="v1"):
    """Divisão determinística treino/val/teste pelo hash do texto NORMALIZADO: duplicatas caem sempre no mesmo conjunto."""
    import hashlib
    cum = np.cumsum(fracoes); out = []
    for t in textos:
        h = int(hashlib.sha1((sal + normalizar_texto(t)).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        out.append(int(np.searchsorted(cum, h, side="right")))
    return np.minimum(np.array(out), len(fracoes) - 1)


def detectar_pii(texto):
    """Detecção ingênua (regex) de dados pessoais óbvios: CPF, e-mail e telefone. NÃO é suficiente em produção."""
    import re
    padroes = {"cpf": r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "telefone": r"\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b"}
    return {k: re.findall(p, texto) for k, p in padroes.items() if re.findall(p, texto)}


def mascarar_pii(texto):
    import re
    texto = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF]", texto)
    texto = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL]", texto)
    return re.sub(r"\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b", "[TELEFONE]", texto)
