"""Testes da infraestrutura (execute: python testes_rnbook.py)."""
import numpy as np
from scipy.optimize import least_squares
from rnbook import MLP, grad_numerico, erro_relativo
from rnbook.backprop_escalar import backprop_escalar
from rnbook import otimizadores as O

rng = np.random.default_rng(0)

def teste_gradiente():
    for ativ in (["tanh","identidade"],["sigmoide","sigmoide"],["tanh","tanh","identidade"]):
        cam=[3,4,2] if len(ativ)==2 else [3,4,3,2]
        rede=MLP(cam,ativ,semente=1,init="pequena")
        X=rng.normal(size=(7,3)); T=rng.normal(size=(7,2))
        for perda in ("sse","mse"):
            f=lambda th: rede.perda(th,X,T,perda)
            ga=rede.grad(rede.theta,X,T,perda); gn=grad_numerico(f,rede.theta)
            e=erro_relativo(ga,gn); print(f"grad {ativ} {perda}: erro rel {e:.2e}")
            assert e<1e-7

def teste_escalar_vs_vetorial():
    rede=MLP([3,4,2],["tanh","sigmoide"],semente=2,init="pequena")
    x=rng.normal(size=3); t=rng.uniform(size=2)
    W=[w.copy() for w,_ in rede.desempacotar(rede.theta)]; b=[bb.copy() for _,bb in rede.desempacotar(rede.theta)]
    E,dW,db=backprop_escalar(W,b,["tanh","sigmoide"],x,t)
    g=rede.grad(rede.theta,x[None,:],t[None,:])
    g2=np.concatenate([np.concatenate([dW[l].ravel(),db[l]]) for l in range(2)])
    print("escalar x vetorizado:",np.abs(g-g2).max()); assert np.allclose(g,g2)
    assert np.isclose(E,rede.perda(rede.theta,x[None,:],t[None,:]))

def teste_jacobiana():
    rede=MLP([3,5,2],["tanh","identidade"],semente=3,init="pequena")
    X=rng.normal(size=(6,3)); T=rng.normal(size=(6,2))
    J=rede.jacobiana(rede.theta,X)
    r=lambda th: rede.residuos(th,X,T)
    Jn=np.stack([ (r(rede.theta+1e-6*e)-r(rede.theta-1e-6*e))/2e-6 for e in np.eye(rede.n_params)],axis=1)
    print("jacobiana erro max:",np.abs(J-Jn).max()); assert np.allclose(J,Jn,atol=1e-7)
    # grad = J^T r  (sse)
    g=rede.grad(rede.theta,X,T,"sse"); assert np.allclose(g,J.T@r(rede.theta))

def problema():
    X=np.linspace(-1,1,40)[:,None]; T=np.sin(3*X)
    rede=MLP([1,6,1],["tanh","identidade"],semente=4,init="xavier")
    return rede,X,T

def teste_otimizadores():
    rede,X,T=problema()
    fun=lambda th: rede.perda(th,X,T); grad=lambda th: rede.grad(th,X,T)
    E0=fun(rede.theta); print("E0",E0)
    res={}
    res["gd"]=O.gradiente_descendente(rede.theta,fun,grad,taxa=0.01,max_iter=2000)
    for v in O.VARIANTES_RPROP: res[v]=O.rprop(rede.theta,fun,grad,variante=v,max_iter=300)
    res["qprop"]=O.quickprop(rede.theta,fun,grad,epsilon=0.01,max_iter=300)
    res["lm"]=O.levenberg_marquardt(rede.theta,lambda th: rede.residuos(th,X,T),lambda th: rede.jacobiana(th,X),max_iter=100)
    res["lm_marq"]=O.levenberg_marquardt(rede.theta,lambda th: rede.residuos(th,X,T),lambda th: rede.jacobiana(th,X),escala="marquardt",max_iter=100)
    res["dd"]=O.delta_delta(rede.theta,fun,grad,max_iter=500,alfa0=0.005,gama=0.001)
    res["dbd"]=O.delta_bar_delta(rede.theta,fun,grad,max_iter=500)
    res["edbd"]=O.delta_bar_delta_estendido(rede.theta,fun,grad,max_iter=500)
    for k,v in res.items():
        print(f"{k:8s} E_final={v.historico['perda'][-1]:.3e} iter={v.n_iter} motivo={v.motivo}")
        assert np.isfinite(v.historico['perda'][-1])
    return res

def teste_lm_vs_scipy():
    rede,X,T=problema()
    r=lambda th: rede.residuos(th,X,T); J=lambda th: rede.jacobiana(th,X)
    a=O.levenberg_marquardt(rede.theta,r,J,max_iter=200)
    b=least_squares(r,rede.theta,jac=J,method="lm",max_nfev=200)
    print("LM didático E=%.3e (it %d) | scipy lm E=%.3e (nfev %d)"%(a.historico['perda'][-1],a.n_iter,0.5*b.fun@b.fun,b.nfev))

if __name__=="__main__":
    teste_gradiente(); teste_escalar_vs_vetorial(); teste_jacobiana(); teste_otimizadores(); teste_lm_vs_scipy(); print("TODOS OS TESTES OK")

def teste_gn_lm_nielsen():
    rede,X,T=problema()
    r=lambda th: rede.residuos(th,X,T); J=lambda th: rede.jacobiana(th,X)
    a=O.levenberg_marquardt_nielsen(rede.theta,r,J,max_iter=100)
    b=O.gauss_newton(rede.theta,r,J,max_iter=50)
    print("Nielsen E=%.3e it=%d (%s) | Gauss-Newton puro E=%.3e it=%d (%s)"%(a.historico['perda'][-1],a.n_iter,a.motivo,b.historico['perda'][-1],b.n_iter,b.motivo))
    assert np.isfinite(a.historico['perda'][-1])
if __name__=="__main__":
    teste_gn_lm_nielsen()
