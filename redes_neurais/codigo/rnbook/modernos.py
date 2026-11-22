"""Otimizadores modernos (minilote), produtos Hessiana–vetor e diagnóstico.

Implementações didáticas do livro (Capítulo 9); não substituem bibliotecas profissionais.

Convenção: grad_amostras(theta, idx) devolve o gradiente da perda SOMADA sobre as amostras idx
(como em gd_estocastico); o gradiente do minilote é a média, g = grad_amostras / len(idx).
"""
import time
import numpy as np
from .otimizadores import Resultado


def _loop_minilotes(theta0, grad_amostras, N, fun, passo_fn, tamanho_lote, epocas, semente):
    """Laço comum: embaralha a cada época e chama passo_fn(theta, g, t) -> theta novo."""
    rng = np.random.default_rng(semente)
    theta = theta0.copy()
    perdas, t, t0 = [float(fun(theta))], 0, time.perf_counter()
    for _ in range(epocas):
        ordem = rng.permutation(N)
        for ini in range(0, N, tamanho_lote):
            idx = ordem[ini:ini + tamanho_lote]
            g = grad_amostras(theta, idx) / len(idx)
            t += 1
            theta = passo_fn(theta, g, t)
        perdas.append(float(fun(theta)))
        if not np.isfinite(perdas[-1]):
            break
    r = Resultado(theta=theta, theta_melhor=theta, historico={"perda": perdas}, n_iter=len(perdas) - 1)
    r.tempo = time.perf_counter() - t0
    r.motivo = "fim" if np.isfinite(perdas[-1]) else "valores_nao_finitos"
    return r


def sgd_momento(theta0, grad_amostras, N, fun, taxa=0.05, momento=0.9, nesterov=False,
                l2=0.0, tamanho_lote=10, epocas=100, semente=0):
    """SGD com momento clássico (Polyak) ou de Nesterov.
    Clássico:  v <- mu v - eta g(theta);          theta <- theta + v
    Nesterov:  v <- mu v - eta g(theta + mu v);   theta <- theta + v
    O Nesterov é implementado na forma equivalente que usa o gradiente no ponto atual (Sutskever et al., 2013):
               v <- mu v - eta g(theta);          theta <- theta + mu v - eta g(theta)
    l2 soma l2*theta ao gradiente (penalidade L2 acoplada)."""
    estado = {"v": np.zeros_like(theta0)}

    def passo(theta, g, t):
        v = estado["v"]
        if nesterov:   # forma equivalente de Sutskever: theta + mu v_novo - eta g
            v_novo = momento * v - taxa * (g + l2 * theta)
            estado["v"] = v_novo
            return theta + momento * v_novo - taxa * (g + l2 * theta)
        v_novo = momento * v - taxa * (g + l2 * theta)
        estado["v"] = v_novo
        return theta + v_novo
    return _loop_minilotes(theta0, grad_amostras, N, fun, passo, tamanho_lote, epocas, semente)


def adagrad(theta0, grad_amostras, N, fun, taxa=0.1, eps=1e-8, tamanho_lote=10, epocas=100, semente=0):
    """AdaGrad (Duchi et al., 2011): G <- G + g^2;  theta <- theta - eta g / (sqrt(G) + eps)."""
    est = {"G": np.zeros_like(theta0)}

    def passo(theta, g, t):
        est["G"] += g * g
        return theta - taxa * g / (np.sqrt(est["G"]) + eps)
    return _loop_minilotes(theta0, grad_amostras, N, fun, passo, tamanho_lote, epocas, semente)


def rmsprop(theta0, grad_amostras, N, fun, taxa=0.01, rho=0.9, eps=1e-8, tamanho_lote=10, epocas=100, semente=0):
    """RMSProp (Tieleman; Hinton, 2012): s <- rho s + (1-rho) g^2;  theta <- theta - eta g / (sqrt(s) + eps)."""
    est = {"s": np.zeros_like(theta0)}

    def passo(theta, g, t):
        est["s"] = rho * est["s"] + (1 - rho) * g * g
        return theta - taxa * g / (np.sqrt(est["s"]) + eps)
    return _loop_minilotes(theta0, grad_amostras, N, fun, passo, tamanho_lote, epocas, semente)


def adam(theta0, grad_amostras, N, fun, taxa=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
         l2=0.0, weight_decay=0.0, tamanho_lote=10, epocas=100, semente=0):
    """Adam (Kingma; Ba, 2015) e AdamW (Loshchilov; Hutter, 2019).
       m <- b1 m + (1-b1) g;  v <- b2 v + (1-b2) g^2
       m_hat = m/(1-b1^t);  v_hat = v/(1-b2^t)
       theta <- theta - eta m_hat/(sqrt(v_hat)+eps) - eta*weight_decay*theta   (decaimento DESACOPLADO)
    l2 > 0 soma l2*theta ao gradiente ANTES da normalização (penalidade L2 acoplada: Adam + L2)."""
    est = {"m": np.zeros_like(theta0), "v": np.zeros_like(theta0)}

    def passo(theta, g, t):
        g = g + l2 * theta
        est["m"] = beta1 * est["m"] + (1 - beta1) * g
        est["v"] = beta2 * est["v"] + (1 - beta2) * g * g
        mh = est["m"] / (1 - beta1 ** t)
        vh = est["v"] / (1 - beta2 ** t)
        return theta - taxa * mh / (np.sqrt(vh) + eps) - taxa * weight_decay * theta
    return _loop_minilotes(theta0, grad_amostras, N, fun, passo, tamanho_lote, epocas, semente)


# ===================================================== produtos Hessiana–vetor e CG
def hvp_dif_finitas(grad, theta, v, eps=1e-5):
    """H v ≈ [g(theta + eps v) - g(theta - eps v)] / (2 eps), sem formar H (2 gradientes)."""
    return (grad(theta + eps * v) - grad(theta - eps * v)) / (2 * eps)


def jvp_dif_finitas(res, theta, v, eps=1e-6):
    """J v ≈ [r(theta + eps v) - r(theta - eps v)] / (2 eps) (produto Jacobiana–vetor)."""
    return (res(theta + eps * v) - res(theta - eps * v)) / (2 * eps)


def gradiente_conjugado(A_op, b, tol=1e-8, max_iter=None):
    """Resolve A x = b (A simétrica definida positiva) usando só o produto A_op(v). Devolve (x, iterações)."""
    n = len(b); max_iter = max_iter or n
    x = np.zeros(n); r = b.copy(); p = r.copy(); rs = r @ r; b2 = np.linalg.norm(b)
    for k in range(1, max_iter + 1):
        Ap = A_op(p)
        alfa = rs / (p @ Ap)
        x += alfa * p; r -= alfa * Ap
        rs_novo = r @ r
        if np.sqrt(rs_novo) <= tol * b2:
            return x, k
        p = r + (rs_novo / rs) * p; rs = rs_novo
    return x, max_iter


def lm_sem_matriz(theta0, res, jac_vp, jac_tvp, mu0=1e-3, mu_inc=10.0, mu_dec=0.1, max_iter=100,
                  tol_grad=1e-8, tol_cg=1e-4, max_cg=200):
    """LM em que (J^T J + mu I) d = -g é resolvido por gradiente conjugado, usando apenas
    produtos J v (jac_vp) e J^T u (jac_tvp): J^T J nunca é formada. Devolve (theta, n_it, cg_total)."""
    theta, mu = theta0.copy(), mu0
    r = res(theta); E = 0.5 * r @ r; cg_total = 0; n = 0
    for n in range(1, max_iter + 1):
        g = jac_tvp(theta, r)
        if np.linalg.norm(g) < tol_grad:
            break
        aceito = False
        for _ in range(20):
            A = lambda v: jac_tvp(theta, jac_vp(theta, v)) + mu * v
            d, k = gradiente_conjugado(A, -g, tol=tol_cg, max_iter=max_cg); cg_total += k
            rn = res(theta + d); En = 0.5 * rn @ rn
            if np.isfinite(En) and En < E:
                theta, r, E, mu, aceito = theta + d, rn, En, max(mu * mu_dec, 1e-20), True
                break
            mu *= mu_inc
        if not aceito:
            break
    return theta, n, cg_total


# ============================================================== diagnóstico
def diagnostico(rede, theta, X, T, passo=None, perda="sse"):
    """Resumo de saúde do treinamento de uma MLP.
    Devolve dict: finito (bool), perda, norma_grad_por_camada, norma_pesos_por_camada,
    razao_atualizacao_por_camada (||passo||/||pesos||, se passo for dado) e fração de saturação
    (|a|>0,95 em tanh/sigmoide centrada) por camada oculta."""
    A, Z = rede.forward(X, theta)
    g = rede.grad(theta, X, T, perda)
    out = {"finito": bool(np.all(np.isfinite(g)) and np.all(np.isfinite(theta)) and np.all(np.isfinite(A[-1]))),
           "perda": rede.perda(theta, X, T, perda), "norma_grad": float(np.linalg.norm(g)),
           "norma_grad_por_camada": [], "norma_pesos_por_camada": [], "razao_atualizacao_por_camada": [],
           "saturacao_por_camada": []}
    for l, ((iw, fw), (ib, fb)) in enumerate(rede.fatias()):
        out["norma_grad_por_camada"].append(float(np.linalg.norm(g[iw:fw])))
        nw = float(np.linalg.norm(theta[iw:fw]))
        out["norma_pesos_por_camada"].append(nw)
        if passo is not None:
            out["razao_atualizacao_por_camada"].append(float(np.linalg.norm(passo[iw:fw]) / max(nw, 1e-30)))
        if l < rede.L - 1:
            a = A[l + 1]
            out["saturacao_por_camada"].append(float(np.mean(np.abs(a) > 0.95)) if rede.nomes_ativ[l] == "tanh"
                                               else float(np.mean((a < 0.05) | (a > 0.95))))
    return out
