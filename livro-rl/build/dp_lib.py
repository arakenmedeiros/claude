import numpy as np
def grade(n=4):
    """Gridworld n×n, terminais nos cantos opostos, recompensa -1 por passo, ações: 0↑ 1↓ 2← 3→."""
    nS=n*n; P=np.zeros((nS,4,nS)); R=-np.ones((nS,4)); term=[0,nS-1]
    mov=[(-1,0),(1,0),(0,-1),(0,1)]
    for s in range(nS):
        if s in term: P[s,:,s]=1; R[s,:]=0; continue
        i,j=divmod(s,n)
        for a,(di,dj) in enumerate(mov):
            i2,j2=min(max(i+di,0),n-1),min(max(j+dj,0),n-1); P[s,a,i2*n+j2]=1
    return P,R,term
def avaliacao(P,R,pol,gamma=1.0,theta=1e-8,max_it=10000,hist=None):
    nS=P.shape[0]; V=np.zeros(nS)
    for k in range(max_it):
        Vn=np.einsum("sa,sa->s",pol,R+gamma*np.einsum("sat,t->sa",P,V))
        d=np.abs(Vn-V).max(); V=Vn
        if hist is not None: hist.append(V.copy())
        if d<theta: break
    return V,k+1
def gulosa(P,R,V,gamma=1.0):
    Q=R+gamma*np.einsum("sat,t->sa",P,V); return Q
def iteracao_politica(P,R,gamma=1.0,theta=1e-8):
    nS,nA=R.shape; pol=np.full((nS,nA),1/nA); n_aval=0
    for it in range(1000):
        V,k=avaliacao(P,R,pol,gamma,theta); n_aval+=k
        Q=gulosa(P,R,V,gamma); nova=np.zeros_like(pol); nova[np.arange(nS),Q.argmax(1)]=1
        if np.array_equal(nova,pol): return V,pol,it+1,n_aval
        pol=nova
    return V,pol,it+1,n_aval
def iteracao_valor(P,R,gamma=1.0,theta=1e-8,hist=None):
    nS=R.shape[0]; V=np.zeros(nS)
    for k in range(100000):
        Q=R+gamma*np.einsum("sat,t->sa",P,V); Vn=Q.max(1); d=np.abs(Vn-V).max(); V=Vn
        if hist is not None: hist.append(d)
        if d<theta: break
    Q=R+gamma*np.einsum("sat,t->sa",P,V); pol=np.zeros_like(Q); pol[np.arange(nS),Q.argmax(1)]=1
    return V,pol,k+1
