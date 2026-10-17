from icons import *
import sys
sys.path.insert(0,"/home/user/claude/livro_af/cap/lib")
from fronteiras import *
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def f11_lora():
    fig,ax=canvas(10,3.8)
    rbox(ax,4,30,28,40,"W₀\n(modelo pré-treinado,\ncongelado)\nd × K",fc="#E5E7EB",ec=GRY,fs=10,bold=True,lw=2)
    ax.text(36,50,"+",fontsize=30,ha="center",va="center",fontweight="bold")
    rbox(ax,41,38,8,24,"A\nd×r",fc=AMB,tc="white",bold=True,fs=9)
    ax.text(51.5,50,"·",fontsize=26,ha="center",va="center",fontweight="bold")
    rbox(ax,54,45,24,10,"B   r × K",fc=IND,tc="white",bold=True,fs=9)
    ax.text(83,50,"=",fontsize=30,ha="center",va="center",fontweight="bold")
    rbox(ax,87,38,11,24,"modelo\nadaptado",fc="#ECFDF5",ec=GRN,fs=8.5,bold=True,lw=2)
    ax.text(50,12,"só A e B (r·(d + K) números) treinam e viajam; r é pequeno",ha="center",fontsize=10,color=INK,fontweight="bold")
    ax.plot([41,78],[72,72],c=AMB,lw=2); ax.text(59.5,77,"correção de baixo posto  ΔW = A·B",ha="center",fontsize=9,color=AMB,fontweight="bold")
    save(fig,"f11_lora")

def f11_lora_res():
    Xtr,Xte,ytr,yte=carregar_digits()
    W0=np.zeros((65,10))
    for _ in range(300): W0-=0.5*gradiente(W0,Xtr,ytr)
    perm=np.arange(10); perm[3],perm[5]=5,3; perm[4],perm[9]=9,4; ytr2,yte2=perm[ytr],perm[yte]
    fig,ax=plt.subplots(figsize=(7,4.0))
    for nome,posto,c in [("completo",None,INK),("LoRA r=1",1,RED),("LoRA r=2",2,GRN),("LoRA r=4",4,AMB),("LoRA r=8",8,IND)]:
        hs=[];
        for s in range(3):
            h,b=fedavg_adaptador(particao_dirichlet(ytr,20,0.5,s),Xtr,ytr2,Xte,yte2,W0,posto,seed=s); hs.append(h)
        h=np.mean(hs,axis=0); ax.scatter(b/1024,h[-1],s=130,c=c,ec="white",zorder=3); ax.annotate(f"{nome}\n({h[-1]:.2f})",(b/1024,h[-1]),{"LoRA r=1":(6,6),"LoRA r=8":(-30,12),"completo":(6,-30)}.get(nome,(6,-22)),textcoords="offset points",fontsize=8.5)
    ax.set_xlabel("tráfego por cliente e rodada (KB)"); ax.set_ylabel("acurácia na tarefa nova (rodada 20)"); ax.set_ylim(0.7,1.0); ax.set_xlim(0,6.5)
    ax.set_title("Quanto tráfego é preciso? Posto 2 já basta",fontsize=10)
    save(fig,"f11_lora_res")

def f11_kmeans():
    X,y=load_wine(return_X_y=True); X=StandardScaler().fit_transform(X); partes=particao_dirichlet(y,5,0.3,0)
    cli=np.zeros(len(X),int)
    for k,p in enumerate(partes): cli[p]=k
    rng=np.random.default_rng(0); C0=X[rng.choice(len(X),3,replace=False)]; Cf=kmeans_federado(partes,X,C0,50)
    lab=((X[:,None,:]-Cf[None])**2).sum(-1).argmin(1)
    pca=PCA(2).fit(X); Z=pca.transform(X); Zc=pca.transform(Cf)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.8),sharex=True,sharey=True)
    cores=[IND,RED,AMB,GRN,VIO]
    for k in range(5):
        if (cli==k).sum(): axs[0].scatter(Z[cli==k,0],Z[cli==k,1],c=cores[k],s=22,ec="white",lw=0.3,label=f"cliente {k+1} (n={ (cli==k).sum() })")
    axs[0].legend(frameon=False,fontsize=7.5,loc="lower right"); axs[0].set_title("quem possui cada ponto",fontsize=9.5)
    for j,c in enumerate([CYA,PNK,"#84CC16"]): axs[1].scatter(Z[lab==j,0],Z[lab==j,1],c=c,s=22,ec="white",lw=0.3)
    axs[1].scatter(Zc[:,0],Zc[:,1],marker="X",s=200,c=INK,ec="white",zorder=5); axs[1].set_title("grupos do K-means federado (✕ = centroides)",fontsize=9.5)
    for a in axs: a.set_xlabel("componente 1"); 
    axs[0].set_ylabel("componente 2")
    save(fig,"f11_kmeans")

def f11_aplicacoes():
    fig,ax=canvas(10,3.6)
    cards=[("Saúde",RED,hospital,"imagens e prontuários\nficam no hospital\n(cross-silo)"),("Celulares",IND,phone,"teclados e assistentes\nmilhões de aparelhos\n(cross-device)"),("IoT e borda",GRN,None,"sensores e veículos\ntreinam perto dos dados"),("Finanças",AMB,bank,"bancos e seguradoras\ndados complementares\n(vertical)")]
    for i,(t,c,ic,d) in enumerate(cards):
        x=2+i*24.5
        rbox(ax,x,6,23,88,fc="white",ec=c,lw=2.2,r=3); rbox(ax,x,78,23,16,t,fc=c,tc="white",bold=True,fs=10.5,r=3)
        if ic: ic(ax,x+11.5,58,1.3,c)
        else: netglyph(ax,x+11.5,58,1.4,c)
        ax.text(x+11.5,28,d,ha="center",va="center",fontsize=8.6,linespacing=1.3)
    save(fig,"f11_aplicacoes")

if __name__=="__main__":
    for f in [f11_lora,f11_lora_res,f11_kmeans,f11_aplicacoes]: f()
