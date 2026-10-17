from icons import *
import sys
sys.path.insert(0,"/home/user/claude/livro_af/cap/lib")
from fedlib import *
from scaffold import scaffold

def f4_tipos():
    fig,axs=plt.subplots(1,4,figsize=(10,3.0))
    rng=np.random.default_rng(2); cores=[IND,AMB,GRN]
    # rótulos
    ax=axs[0]; props=[[.7,.2,.1],[.1,.2,.7],[.3,.4,.3]]
    for i,p in enumerate(props):
        b=0
        for c,v in zip(cores,p): ax.barh(2-i,v,left=b,color=c,ec="white"); b+=v
        ax.text(-0.04,2-i,f"cliente {i+1}",ha="right",va="center",fontsize=8)
    ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xticks([]); ax.set_title("Rótulos\nP(y) muda",fontsize=9.5)
    for s in ax.spines.values(): s.set_visible(False)
    # atributos
    ax=axs[1]
    for i,(dx,dy,m) in enumerate([(0,0,"o"),(2.2,0.3,"s"),(1.0,-2.0,"^")]):
        for c in range(2):
            P=rng.normal([c*1.1+dx,c*0.4+dy],0.35,(14,2)); ax.scatter(P[:,0],P[:,1],c=cores[c],marker=m,s=22,ec="white",lw=0.4)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Atributos\nP(x) muda (formas = clientes)",fontsize=9.5)
    # quantidade
    ax=axs[2]; ax.bar(["c1","c2","c3","c4"],[1200,300,80,900],color=[IND,AMB,GRN,PNK]); ax.set_yticks([]); ax.set_title("Quantidade\nn_k muda",fontsize=9.5)
    # conceito
    ax=axs[3]
    X=rng.uniform(-1,1,(60,2))
    for off,(m,th) in enumerate([("o",1),("s",-1)]):
        P=X+np.array([off*2.6,0]); y=(th*P[:,0]-th*off*2.6>0).astype(int)
        for c in range(2): ax.scatter(P[y==c,0],P[y==c,1],c=cores[c],marker=m,s=20,ec="white",lw=0.4)
        ax.text(off*2.6,1.25,f"cliente {off+1}",ha="center",fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-1.2,1.6); ax.set_title("Conceito\nP(y|x) muda",fontsize=9.5)
    for a in axs[1:]:
        for s in a.spines.values(): s.set_color(MID)
    save(fig,"f4_tipos")

def f4_dirichlet():
    Xtr,Xte,ytr,yte=carregar_digits()
    fig,axs=plt.subplots(1,3,figsize=(10,3.4))
    for ax,alfa in zip(axs,(100,1.0,0.1)):
        p=particao_dirichlet(ytr,10,alfa,0); M=np.array([np.bincount(ytr[i],minlength=10)/max(len(i),1) for i in p])
        im=ax.imshow(M,cmap="Blues",vmin=0,vmax=1,aspect="auto"); ax.set_title(f"α = {alfa:g}",fontsize=10.5)
        ax.set_xlabel("classe (dígito)"); ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.set_yticklabels(range(1,11),fontsize=7)
    axs[0].set_ylabel("cliente")
    save(fig,"f4_dirichlet")

def f4_deriva():
    A1=np.diag([1.0,5.0]); c1=np.array([2.4,0.0]); A2=np.diag([5.0,1.0]); c2=np.array([-0.4,2.4])
    ws=np.linalg.solve(A1+A2,A1@c1+A2@c2); med=(c1+c2)/2
    fig,ax=plt.subplots(figsize=(6.2,4.4))
    g1,g2=np.meshgrid(np.linspace(-2,3.6,200),np.linspace(-1.5,3.4,200))
    Z=0.5*(0.5*(A1[0,0]*(g1-c1[0])**2+A1[1,1]*(g2-c1[1])**2)+0.5*(A2[0,0]*(g1-c2[0])**2+A2[1,1]*(g2-c2[1])**2))
    ax.contourf(g1,g2,Z,levels=16,cmap="Blues_r",alpha=0.85); ax.contour(g1,g2,Z,levels=16,colors="white",linewidths=0.3)
    w=np.array([-1.6,-1.0]); P=[w.copy()]
    for r in range(25):
        loc=[]
        for A,c in ((A1,c1),(A2,c2)):
            x=w.copy()
            for _ in range(60): x=x-0.1*A@(x-c)
            loc.append(x)
        w=np.mean(loc,axis=0); P.append(w.copy())
    P=np.array(P); ax.plot(P[:,0],P[:,1],"o-",c=AMB,ms=3.5,lw=1.4,mec="white")
    ax.scatter(*ws,marker="*",s=260,c=GRN,ec="white",zorder=6); ax.scatter(*med,marker="D",s=80,c=RED,ec="white",zorder=6)
    ax.scatter(*c1,marker="X",s=90,c=IND,ec="white",zorder=6); ax.scatter(*c2,marker="X",s=90,c=INK,ec="white",zorder=6)
    ax.annotate("mínimo global",ws,(ws[0]+0.45,ws[1]-0.65),fontsize=8.5,color=GRN,fontweight="bold",arrowprops=dict(arrowstyle="-",color=GRN))
    ax.annotate("FedAvg estaciona aqui\n(média dos mínimos locais)",med,(0.9,2.55),fontsize=8.5,color=RED,fontweight="bold",arrowprops=dict(arrowstyle="-",color=RED))
    ax.text(c1[0]+0.05,c1[1]-0.35,"mín. A",fontsize=8,ha="center"); ax.text(c2[0]-0.1,c2[1]+0.2,"mín. B",fontsize=8,ha="center")
    ax.set_xlabel("w₁"); ax.set_ylabel("w₂"); ax.set_title("Muitos passos locais: o objetivo muda")
    save(fig,"f4_deriva")

def _hard(parts,s,mu=0.0): return fedavg(parts,*DATA,seed=s,rodadas=40,epocas=20,eta=0.3,frac=0.1,mu=mu)[1]
def f4_curvas():
    global DATA
    Xtr,Xte,ytr,yte=carregar_digits(); DATA=(Xtr,ytr,Xte,yte)
    cenarios=[("IID",lambda s:particao_iid(ytr,20,s),IND),("Dirichlet α=1",lambda s:particao_dirichlet(ytr,20,1.0,s),GRN),("Dirichlet α=0,1",lambda s:particao_dirichlet(ytr,20,0.1,s),AMB),("Shards: 1 classe por cliente",lambda s:particao_shards(ytr,20,1,s),RED)]
    fig,ax=plt.subplots(figsize=(6.8,3.9))
    for nome,mk,c in cenarios:
        H=np.mean([_hard(mk(s),s) for s in range(3)],axis=0); ax.plot(range(1,41),H,c=c,lw=2,label=nome)
    ax.set_ylim(0.3,1.0); ax.set_xlabel("rodada de comunicação"); ax.set_ylabel("acurácia no teste"); ax.legend(frameon=False,fontsize=8.5,loc="lower right")
    ax.set_title("Quanto mais heterogêneos os dados, mais lento e instável")
    save(fig,"f4_curvas")

def f4_algos():
    Xtr,Xte,ytr,yte=carregar_digits(); cfg=dict(rodadas=40,epocas=20,eta=0.3,frac=0.1)
    R={"FedAvg":[],"FedProx (μ=0,1)":[],"SCAFFOLD":[]}
    for s in range(3):
        p=particao_shards(ytr,20,1,s)
        R["FedAvg"].append(fedavg(p,Xtr,ytr,Xte,yte,seed=s,**cfg)[1]); R["FedProx (μ=0,1)"].append(fedavg(p,Xtr,ytr,Xte,yte,mu=0.1,seed=s,**cfg)[1]); R["SCAFFOLD"].append(scaffold(p,Xtr,ytr,Xte,yte,seed=s,**cfg)[1])
    fig,ax=plt.subplots(figsize=(6.8,3.9))
    for (n,h),c in zip(R.items(),[IND,AMB,GRN]):
        h=np.array(h); m=h.mean(0); ax.plot(range(1,41),m,c=c,lw=2,label=n); ax.fill_between(range(1,41),h.min(0),h.max(0),color=c,alpha=0.15)
    ax.set_ylim(0.3,1.0); ax.set_xlabel("rodada de comunicação"); ax.set_ylabel("acurácia no teste"); ax.legend(frameon=False,fontsize=8.5,loc="lower right")
    ax.set_title("Cada cliente com uma única classe: média e faixa de 3 sementes",fontsize=9.5)
    save(fig,"f4_algos")

if __name__=="__main__":
    for f in [f4_tipos,f4_dirichlet,f4_deriva,f4_curvas,f4_algos]: f()
