"""Successive Halving e Hyperband didáticos, com SGDClassifier e partial_fit (recurso = épocas)."""
import numpy as np, math
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def dados(semente=0):
    X,y=load_digits(return_X_y=True); Xtr,Xva,ytr,yva=train_test_split(X,y,test_size=0.3,stratify=y,random_state=semente)
    sc=StandardScaler().fit(Xtr); return sc.transform(Xtr),sc.transform(Xva),ytr,yva

class Candidato:
    """Uma configuração que pode ser treinada aos poucos (pausar e retomar)."""
    def __init__(self,rng,D):
        self.hp={"alpha":10**rng.uniform(-8,1),"eta0":10**rng.uniform(-6,1),"loss":str(rng.choice(["hinge","log_loss","modified_huber"]))}
        self.m=SGDClassifier(loss=self.hp["loss"],alpha=self.hp["alpha"],learning_rate="constant",eta0=self.hp["eta0"],random_state=int(rng.integers(1e6)))
        self.D=D; self.epocas=0; self.nota=0.0; self.curva=[]
    def treinar_ate(self,epocas):
        Xtr,Xva,ytr,yva=self.D
        while self.epocas<epocas:
            self.m.partial_fit(Xtr,ytr,classes=np.arange(10)); self.epocas+=1; self.nota=self.m.score(Xva,yva); self.curva.append(self.nota)
        return self.nota

def sucessivo(cands,r_min,eta=3,r_max=None):
    """Successive Halving: treina todos com r_min épocas, mantém o 1/eta melhor, multiplica o orçamento por eta."""
    r=r_min; gasto=0
    while True:
        for c in cands:
            antes=c.epocas; c.treinar_ate(int(r)); gasto+=c.epocas-antes
        if len(cands)<=1 or (r_max and r>=r_max): break
        cands=sorted(cands,key=lambda c:-c.nota)[:max(1,len(cands)//eta)]; r*=eta
    return max(cands,key=lambda c:c.nota),gasto

def hyperband(rng,D,R=81,eta=3):
    smax=int(math.log(R)/math.log(eta)+1e-9); B=(smax+1)*R; melhor=None; gasto=0; ncfg=0
    for s in range(smax,-1,-1):
        n=int(math.ceil(B/R*eta**s/(s+1))); r=R*eta**(-s)
        cands=[Candidato(rng,D) for _ in range(n)]; ncfg+=n
        c,g=sucessivo(cands,r,eta,R); gasto+=g
        if melhor is None or c.nota>melhor.nota: melhor=c
    return melhor,gasto,ncfg

def aleatoria(rng,D,orcamento,R=81):
    melhor=0.0; gasto=0
    while gasto+R<=orcamento:
        c=Candidato(rng,D); c.treinar_ate(R); gasto+=R; melhor=max(melhor,c.nota)
    return melhor,gasto
