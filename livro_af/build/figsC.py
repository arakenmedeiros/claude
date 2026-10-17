from icons import *
import sys, copy
sys.path.insert(0,"/home/user/claude/livro_af/cap/lib")
from fedlib import *
from matplotlib.colors import LinearSegmentedColormap

def f3_rodada():
    fig,ax=canvas(10,5.4,yl=(-4,100))
    # servidor
    cloud(ax,50,78,1.5,CYA); server(ax,50,78,0.9)
    rbox(ax,36,56,28,9,"modelo global  w_t",fc="#ECFDF5",ec=GRN,fs=9.5,bold=True)
    ic=[(10,phone,IND,True),(26,hospital,RED,True),(42,bank,AMB,False),(58,phone,PNK,True),(74,hospital,VIO,False),(90,bank,"#0EA5E9",True)]
    for i,(x,f,c,sel) in enumerate(ic):
        col=c if sel else MID
        f(ax,x,28,0.95,col)
        ax.text(x,14,f"cliente {i+1}\n{'n = '+str([120,300,80,150,60,220][i])}",ha="center",fontsize=8,color=INK if sel else GRY)
        if sel:
            arrow(ax,(50+(x-50)*0.25,56),(x,38),c=VIO,lw=1.6)
            arrow(ax,(x+3,38),(50+(x-50)*0.25+3,55.5),c=GRN,lw=1.6,rad=-0.1)
        else:
            ax.text(x,38,"fora desta\nrodada",ha="center",fontsize=7.5,color=GRY,style="italic")
    ax.text(2,66,"① o servidor sorteia clientes\n    e envia w_t",fontsize=8.5,color=VIO,fontweight="bold")
    ax.text(72,66,"③ os clientes devolvem\n    w_(t+1)^k",fontsize=8.5,color=GRN,fontweight="bold",ha="left")
    ax.text(50,6,"② cada cliente sorteado treina E épocas, com mini-lotes de tamanho B e passo η",ha="center",fontsize=8.8,color=INK,fontweight="bold")
    ax.text(50,0,"④ o servidor agrega:  w_(t+1) = Σ (n_k / n_S) · w_(t+1)^k",ha="center",fontsize=8.8,color=INK,fontweight="bold")
    save(fig,"f3_rodada")

def f3_fedsgd_avg():
    # dois clientes com mínimos locais diferentes; perda quadrática anisotrópica
    A1=np.diag([1.0,6.0]); c1=np.array([2.0,0.6]); A2=np.diag([4.0,1.0]); c2=np.array([-0.6,2.2])
    def g(w,A,c): return A@(w-c)
    def run(passos,eta=0.07,w0=np.array([-2.0,-1.5])):
        P1=[w0.copy()];P2=[w0.copy()];w1=w0.copy();w2=w0.copy()
        for _ in range(passos):
            w1=w1-eta*g(w1,A1,c1); w2=w2-eta*g(w2,A2,c2); P1.append(w1.copy()); P2.append(w2.copy())
        return np.array(P1),np.array(P2)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.9),sharex=True,sharey=True)
    g1,g2=np.meshgrid(np.linspace(-2.6,3.0,150),np.linspace(-2.0,3.0,150))
    Z=0.5*(0.5*((g1-c1[0])**2*1+(g2-c1[1])**2*6)+0.5*((g1-c2[0])**2*4+(g2-c2[1])**2*1))
    for ax,(titulo,E) in zip(axs,[("1 passo local por rodada (FedSGD)",1),("10 passos locais por rodada (FedAvg)",10)]):
        ax.contourf(g1,g2,Z,levels=14,cmap="Blues_r",alpha=0.8); ax.contour(g1,g2,Z,levels=14,colors="white",linewidths=0.3)
        w=np.array([-2.0,-1.5]); ax.scatter(*w,s=80,c=RED,zorder=6,ec="white")
        for r in range(3 if E==10 else 12):
            P1,P2=run(E,w0=w); ax.plot(P1[:,0],P1[:,1],"-",c=IND,lw=1.6); ax.plot(P2[:,0],P2[:,1],"-",c=AMB,lw=1.6)
            wn=0.5*(P1[-1]+P2[-1]); ax.plot([P1[-1,0],wn[0]],[P1[-1,1],wn[1]],":",c=INK,lw=0.8); ax.plot([P2[-1,0],wn[0]],[P2[-1,1],wn[1]],":",c=INK,lw=0.8)
            ax.scatter(*wn,s=34,c=GRN,zorder=5,ec="white"); w=wn
        ax.scatter(*c1,marker="X",s=80,c=IND,zorder=6,ec="white"); ax.scatter(*c2,marker="X",s=80,c=AMB,zorder=6,ec="white")
        ax.set_title(titulo,fontsize=9.5); ax.set_xlabel("w₁")
    axs[0].set_ylabel("w₂")
    axs[0].text(-2.5,2.7,"✕ mínimo do cliente A (azul)\n✕ mínimo do cliente B (laranja)\n● modelo global (verde)",fontsize=7.2,va="top",color=INK)
    save(fig,"f3_fedsgd_avg")

def f3_curvas():
    Xtr,Xte,ytr,yte=carregar_digits(); p=particao_iid(ytr,10)
    _,h1=fedavg(p,Xtr,ytr,Xte,yte,rodadas=40,epocas=1,eta=0.5,lote=10**6)
    _,h2=fedavg(p,Xtr,ytr,Xte,yte,rodadas=40,epocas=5,eta=0.1,lote=32)
    from sklearn.linear_model import LogisticRegression
    cen=LogisticRegression(max_iter=3000).fit(Xtr[:,:-1],ytr).score(Xte[:,:-1],yte)
    fig,ax=plt.subplots(figsize=(6.8,3.8))
    ax.plot(range(1,41),h1,c=AMB,lw=2,label="FedSGD (1 passo de gradiente por rodada)"); ax.plot(range(1,41),h2,c=IND,lw=2,label="FedAvg (5 épocas locais por rodada)")
    ax.axhline(cen,c=GRN,ls="--",lw=1.6); ax.text(1,cen+0.004,"centralizado",color=GRN,fontsize=8.5,ha="left",va="bottom")
    ax.set_xlabel("rodada de comunicação"); ax.set_ylabel("acurácia no teste"); ax.set_ylim(0.8,1.0); ax.legend(frameon=False,loc="lower right",fontsize=8.5)
    ax.set_title("Mesmo número de rodadas: o FedAvg aprende mais rápido")
    save(fig,"f3_curvas")

def f3_hiper():
    Xtr,Xte,ytr,yte=carregar_digits(); p=particao_iid(ytr,20)
    Es=[1,5,20]; Fs=[0.1,0.3,1.0]; M6=np.zeros((3,3)); M30=np.zeros((3,3))
    for i,E in enumerate(Es):
        for j,f in enumerate(Fs):
            _,h=fedavg(p,Xtr,ytr,Xte,yte,rodadas=30,epocas=E,eta=0.1,frac=f); M6[i,j]=h[5]; M30[i,j]=h[-1]
    fig,axs=plt.subplots(1,2,figsize=(9,3.6))
    cm=LinearSegmentedColormap.from_list("c",["#FEE2E2","#FEF3C7","#D1FAE5"])
    for ax,M,t in zip(axs,[M6,M30],["Acurácia após 6 rodadas","Acurácia após 30 rodadas"]):
        ax.imshow(M,cmap=cm,vmin=0.65,vmax=0.98,aspect="auto")
        for i in range(3):
            for j in range(3): ax.text(j,i,f"{M[i,j]:.3f}",ha="center",va="center",fontsize=10,fontweight="bold",color=INK)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{int(f*100)}%" for f in Fs]); ax.set_yticks(range(3)); ax.set_yticklabels([f"E = {e}" for e in Es])
        ax.set_xlabel("fração de clientes por rodada (C)"); ax.set_title(t,fontsize=10)
        for s in ax.spines.values(): s.set_visible(False)
    save(fig,"f3_hiper")

def f3_init():
    labels=["cliente A","cliente B","média A+B"]
    d=[0.958,0.962,0.893]; s=[0.958,0.971,0.967]
    fig,axs=plt.subplots(1,2,figsize=(8.6,3.4),sharey=True)
    for ax,v,t,c in zip(axs,[d,s],["Inicializações diferentes","Mesma inicialização"],[RED,GRN]):
        b=ax.bar(labels,v,color=[MID,MID,c],ec=INK,lw=0.8)
        for r,x in zip(b,v): ax.text(r.get_x()+r.get_width()/2,x+0.004,f"{x:.3f}",ha="center",fontsize=9,fontweight="bold")
        ax.set_ylim(0.8,1.0); ax.set_title(t,fontsize=10,color=c)
    axs[0].set_ylabel("acurácia no teste")
    save(fig,"f3_init")

if __name__=="__main__":
    for f in [f3_rodada,f3_fedsgd_avg,f3_curvas,f3_hiper,f3_init]: f()
