from icons import *
import sys
sys.path.insert(0,"/home/user/claude/livro_af/cap/lib")
from fedlib import *
from sklearn.metrics import roc_auc_score

def f7_ameacas():
    fig,ax=canvas(10,4.6)
    server(ax,50,52,1.1); ax.text(50,36,"Servidor",ha="center",fontsize=9.5,fontweight="bold")
    for x,y,c,f in [(10,80,IND,phone),(10,22,RED,hospital),(90,80,AMB,bank),(90,22,PNK,phone)]:
        f(ax,x,y,0.9,c)
    arrow(ax,(16,76),(42,58),c=GRN,lw=1.8); arrow(ax,(16,26),(42,46),c=GRN,lw=1.8)
    arrow(ax,(84,76),(58,58),c=GRN,lw=1.8); arrow(ax,(84,26),(58,46),c=GRN,lw=1.8)
    rbox(ax,34,78,32,14,"servidor curioso ou malicioso\nvê cada atualização",fc="#FEF2F2",ec=RED,fs=8.5,lw=1.6)
    arrow(ax,(50,78),(50,66),c=RED,lw=1.6)
    bug(ax,30,52,0.55,RED); ax.text(30,44,"observador\nda rede",ha="center",fontsize=8,color=RED,fontweight="bold")
    bug(ax,70,52,0.55,RED); ax.text(70,44,"cliente\nmalicioso",ha="center",fontsize=8,color=RED,fontweight="bold")
    rbox(ax,32,2,36,10,"usuário do modelo final (consulta o resultado)",fc="#FEF2F2",ec=RED,fs=8.5,lw=1.6)
    save(fig,"f7_ameacas")

def f7_taxonomia():
    fig,ax=canvas(10,4.0)
    rbox(ax,34,82,32,12,"Ataques à privacidade",fc=INK,tc="white",bold=True,fs=10.5)
    fam=[("Reconstrução","inversão de\ngradientes",RED),("Pertencimento","o exemplo estava\nno treino?",AMB),("Propriedades","o conjunto tem\numa característica?",VIO),("Memorização","extração de\ntrechos do treino",CYA),("Ativos","servidor manda\nmodelo armadilha",PNK)]
    for i,(n,d,c) in enumerate(fam):
        x=1+i*19.8
        arrow(ax,(50,82),(x+9,60),c=c,lw=1.5,ms=9)
        rbox(ax,x,42,18,18,n,fc=c,tc="white",bold=True,fs=8.8)
        rbox(ax,x,14,18,22,d,fc="white",ec=c,fs=8.5,lw=1.6)
    save(fig,"f7_taxonomia")

def _rec(W,idx,X,y):
    G=gradiente(W,X[idx],y[idx])*len(idx); gb=G[-1]; c=int(np.argmax(np.abs(gb))); return G[:-1,c]/gb[c]

def f7_inversao():
    Xtr,Xte,ytr,yte=carregar_digits(); rng=np.random.default_rng(3)
    W,_=fedavg(particao_iid(ytr,10),Xtr,ytr,Xte,yte,rodadas=10,epocas=2,eta=0.1)
    def img(v): 
        a=v.reshape(8,8); return (a-a.min())/(a.max()-a.min()+1e-9)
    cols=[("lote = 1",1),("lote = 4",4),("lote = 16",16),("lote = 32",32)]
    fig,axs=plt.subplots(2,4,figsize=(8.4,4.4))
    for j,(t,B) in enumerate(cols):
        for _ in range(50):
            idx=rng.choice(len(ytr),B,replace=False); xr=_rec(W,idx,Xtr,ytr); cor=[np.corrcoef(xr,Xtr[i,:-1])[0,1] for i in idx]
            if B==1 or max(cor)>0.5: break
        k=int(np.argmax(cor)); axs[0,j].imshow(img(Xtr[idx[k],:-1]),cmap="gray_r"); axs[1,j].imshow(img(xr),cmap="gray_r")
        axs[0,j].set_title(t,fontsize=10); axs[1,j].set_xlabel(f"correlação = {max(cor):.2f}",fontsize=8.5)
        for a in axs[:,j]: a.set_xticks([]); a.set_yticks([])
    axs[0,0].set_ylabel("dado real\n(mais próximo)",fontsize=9); axs[1,0].set_ylabel("reconstruído\npelo gradiente",fontsize=9)
    save(fig,"f7_inversao")

def _perdas(W,X,y):
    P=softmax(X@W); return -np.log(P[np.arange(len(y)),y]+1e-12)

def f7_membership():
    Xtr,Xte,ytr,yte=carregar_digits(); cfg=[(150,10,2),(150,50,20),(600,50,20),(1347,50,20)]
    A=[]; ex=None
    for n,r,e in cfg:
        aucs=[]
        for s in range(5):
            idx=np.random.default_rng(s).permutation(len(ytr))[:n]; partes=[idx[p] for p in particao_iid(ytr[idx],6,s)]
            W,_=fedavg(partes,Xtr,ytr,Xte,yte,rodadas=r,epocas=e,eta=0.1,seed=s)
            lm,ln=_perdas(W,Xtr[idx],ytr[idx]),_perdas(W,Xte,yte)
            aucs.append(roc_auc_score(np.r_[np.ones(len(lm)),np.zeros(len(ln))],-np.r_[lm,ln]))
            if (n,r)==(150,50) and s==0: ex=(lm,ln)
        A.append(np.mean(aucs))
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.6))
    lm,ln=ex; bins=np.linspace(-6,0.5,30)
    axs[0].hist(np.log10(lm+1e-6),bins=bins,alpha=0.7,color=RED,density=True,label="membros (treino)"); axs[0].hist(np.log10(ln+1e-6),bins=bins,alpha=0.55,color=IND,density=True,label="não membros")
    axs[0].set_xlabel("log₁₀ da perda"); axs[0].set_ylabel("densidade"); axs[0].legend(frameon=False,fontsize=8.5); axs[0].set_title("150 dados, 50 rodadas, 20 épocas",fontsize=9.5)
    labs=["150\n(10 rod.)","150\n(50 rod.)","600","1347"]
    b=axs[1].bar(range(4),A,color=[AMB,RED,CYA,GRN],ec=INK,lw=0.6); axs[1].axhline(0.5,c=GRY,ls="--"); axs[1].text(3.45,0.505,"chute",fontsize=8,color=GRY,ha="right")
    for r,v in zip(b,A): axs[1].text(r.get_x()+r.get_width()/2,v+0.008,f"{v:.2f}",ha="center",fontweight="bold",fontsize=9)
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(labs,fontsize=8.5); axs[1].set_ylim(0.45,0.7); axs[1].set_ylabel("AUC do ataque"); axs[1].set_xlabel("exemplos de treino"); axs[1].set_title("Mais dados, menos vazamento",fontsize=9.5)
    save(fig,"f7_membership")

if __name__=="__main__":
    for f in [f7_ameacas,f7_taxonomia,f7_inversao,f7_membership]: f()
