"""Ferramentas didáticas (NumPy) do Capítulo 13: patches, perda contrastiva simétrica, STFT, difusão direta, métricas de recuperação.
Tudo é pequeno e sintético; nada aqui é um modelo multimodal real."""
import numpy as np

PALAVRAS = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove"]

# ---------- imagens e patches ----------
def patches(img, p):
    """img (H,W) -> (N, p*p), N=(H/p)*(W/p), em ordem de leitura."""
    H, W = img.shape
    return img.reshape(H // p, p, W // p, p).transpose(0, 2, 1, 3).reshape(-1, p * p)

def posicao_senoidal(n, d):
    """Codificação posicional senoidal (n,d), d par."""
    pos = np.arange(n)[:, None]; i = np.arange(d // 2)[None, :]
    ang = pos / (10000 ** (2 * i / d))
    out = np.zeros((n, d)); out[:, 0::2] = np.sin(ang); out[:, 1::2] = np.cos(ang)
    return out

# ---------- texto sintético ----------
def descricao(digito, img, rng, mediana_tinta):
    """Descrição textual sintética: dígito + atributos calculados da própria imagem (tinta e lado)."""
    tinta = "traço forte" if img.sum() > mediana_tinta else "traço leve"
    cols = np.arange(8)[None, :]
    cx = (img * cols).sum() / max(img.sum(), 1e-9)
    lado = "à esquerda" if cx < 3.4 else ("à direita" if cx > 3.7 else "ao centro")
    molde = [f"dígito {PALAVRAS[digito]}, {tinta}, {lado}", f"número {PALAVRAS[digito]} escrito à mão, {tinta}, {lado}", f"imagem de um {PALAVRAS[digito]}, {tinta}, {lado}"]
    return molde[rng.integers(3)]

def vocabulario(textos):
    v = sorted({w for t in textos for w in t.replace(",", " ").split()})
    return {w: i for i, w in enumerate(v)}

def bow(textos, vocab):
    X = np.zeros((len(textos), len(vocab)))
    for k, t in enumerate(textos):
        for w in t.replace(",", " ").split():
            if w in vocab: X[k, vocab[w]] += 1
    return X

# ---------- perda contrastiva simétrica ----------
def _norm(Z):
    n = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return Z / n, n

def _softmax(S, axis):
    S = S - S.max(axis=axis, keepdims=True); E = np.exp(S); return E / E.sum(axis=axis, keepdims=True)

def infonce_simetrica(Zi, Zt, tau):
    """Retorna (perda, dZi, dZt, S). Linha i de Zi casa com a linha i de Zt. S = cos/tau."""
    n = len(Zi)
    ui, ni = _norm(Zi); ut, nt = _norm(Zt)
    S = ui @ ut.T / tau
    Pr, Pc = _softmax(S, 1), _softmax(S, 0)
    idx = np.arange(n)
    perda = -0.5 * (np.log(Pr[idx, idx]).mean() + np.log(Pc[idx, idx]).mean())
    I = np.eye(n)
    dS = 0.5 * ((Pr - I) + (Pc - I)) / n
    dui = dS @ ut / tau; dut = dS.T @ ui / tau
    dZi = (dui - ui * (ui * dui).sum(1, keepdims=True)) / ni
    dZt = (dut - ut * (ut * dut).sum(1, keepdims=True)) / nt
    return perda, dZi, dZt, S

def treinar_contrastivo(Xi, Xt, d=16, tau=0.1, epocas=150, lote=64, lr=0.05, seed=0):
    """Dois codificadores lineares (Xi@Wi, Xt@Wt) treinados com gradiente manual (Adam). Devolve Wi, Wt, histórico."""
    rng = np.random.default_rng(seed)
    Wi = rng.normal(0, 0.1, (Xi.shape[1], d)); Wt = rng.normal(0, 0.1, (Xt.shape[1], d))
    m = [np.zeros_like(Wi), np.zeros_like(Wt)]; v = [np.zeros_like(Wi), np.zeros_like(Wt)]; t = 0; hist = []
    for ep in range(epocas):
        perm = rng.permutation(len(Xi)); tot = 0; nb = 0
        for s in range(0, len(perm) - lote + 1, lote):
            b = perm[s:s + lote]
            L, dZi, dZt, _ = infonce_simetrica(Xi[b] @ Wi, Xt[b] @ Wt, tau)
            g = [Xi[b].T @ dZi, Xt[b].T @ dZt]; t += 1
            for k, W in enumerate((Wi, Wt)):
                m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                W -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            tot += L; nb += 1
        hist.append(tot / nb)
    return Wi, Wt, hist

def recall_em_k(S, rotulos_linha, rotulos_coluna, ks=(1, 5)):
    """S (n_consultas, n_candidatos). Acerto se algum dos k melhores candidatos tem o mesmo rótulo da consulta."""
    ordem = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        top = rotulos_coluna[ordem[:, :k]]
        out[k] = float((top == rotulos_linha[:, None]).any(1).mean())
    return out

# ---------- áudio ----------
def stft(x, n_fft=256, salto=64):
    """STFT com janela de Hann. Retorna matriz complexa (n_fft//2+1, quadros)."""
    j = np.hanning(n_fft)
    nq = 1 + (len(x) - n_fft) // salto
    Q = np.stack([x[i * salto:i * salto + n_fft] * j for i in range(nq)], axis=1)
    return np.fft.rfft(Q, axis=0)

# ---------- difusão ----------
def agenda_beta(T=1000, b0=1e-4, b1=0.02):
    beta = np.linspace(b0, b1, T); alpha = 1 - beta
    return beta, alpha, np.cumprod(alpha)

def ruido_direto(x0, t, abar, rng):
    """x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps, com t indexando abar (0..T-1)."""
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps, eps

def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
