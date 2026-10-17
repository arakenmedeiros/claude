"""Otimização bayesiana mínima (didática), usada no Cap. 5 e nas figuras."""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as Ck
def f(x): return 0.6*np.exp(-((x-3)/1.2)**2)+1.0*np.exp(-((x-8.2)/0.35)**2)
GRID=np.linspace(0,10,501)[:,None]
def ajustar(X,Y,seed=0):
    k=Ck(1.0)*Matern(length_scale=1.0,nu=2.5)+WhiteKernel(1e-6,(1e-10,1e-3))
    return GaussianProcessRegressor(k,normalize_y=True,n_restarts_optimizer=2,random_state=seed).fit(np.array(X)[:,None],Y)
def aquisicoes(gp,Y,xi=0.01,kappa=2.0):
    mu,sd=gp.predict(GRID,return_std=True); best=max(Y); sd=np.maximum(sd,1e-9)
    z=(mu-best-xi)/sd; ei=(mu-best-xi)*norm.cdf(z)+sd*norm.pdf(z); pi=norm.cdf(z); ucb=mu+kappa*sd
    return mu,sd,ei,pi,ucb
def bo(seed,n=15,n0=3):
    rng=np.random.default_rng(seed); X=list(rng.uniform(0,10,n0)); Y=[f(x) for x in X]
    for _ in range(n-n0):
        gp=ajustar(X,Y,seed); mu,sd,ei,pi,ucb=aquisicoes(gp,Y)
        x=float(GRID[np.argmax(ei),0]); X.append(x); Y.append(f(x))
    return np.maximum.accumulate(Y)
def aleatoria(seed,n=15):
    rng=np.random.default_rng(1000+seed); return np.maximum.accumulate(f(rng.uniform(0,10,n)))
