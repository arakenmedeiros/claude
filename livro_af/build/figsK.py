from icons import *
import sys
sys.path.insert(0,"/home/user/claude/livro_af/cap/lib")
from arquiteturas import *

def tabela(ax,x0,y0,w,h,nr,nc,fill_cells,titulo,cor,rotulos=("linhas","colunas")):
    cw,ch=w/nc,h/nr
    for i in range(nr):
        for j in range(nc):
            c=fill_cells.get((i,j),"#F3F4F6")
            ax.add_patch(Rectangle((x0+j*cw,y0+(nr-1-i)*ch),cw,ch,fc=c,ec="white",lw=1.2))
    ax.text(x0+w/2,y0+h+9,titulo,ha="center",fontsize=9.5,fontweight="bold",color=cor)

def f10_particoes():
    fig,ax=canvas(10,3.8)
    cols=[IND,RED,AMB]
    # horizontal: 6 linhas, 6 colunas; cada cliente tem 2 linhas
    f={(i,j):cols[i//2] for i in range(6) for j in range(6)}
    tabela(ax,3,22,28,50,6,6,f,"Horizontal (HFL)",IND)
    ax.text(17,11,"cada um tem linhas\n(pessoas diferentes)",ha="center",fontsize=8.5)
    for k,(c,t) in enumerate(zip(cols,["A","B","C"])): ax.text(1,60-k*16.5,t,fontsize=9,fontweight="bold",color=c,ha="center")
    # vertical
    f={(i,j):cols[j//2] for i in range(6) for j in range(6)}
    tabela(ax,38,22,28,50,6,6,f,"Vertical (VFL)",RED)
    ax.text(52,11,"cada um tem colunas\n(atributos diferentes)",ha="center",fontsize=8.5)
    for k,(c,t) in enumerate(zip(cols,["A","B","C"])): ax.text(41.5+k*9.3,75,t,fontsize=9,fontweight="bold",color=c,ha="center")
    # transferencia
    f={}
    for i in range(6):
        for j in range(6):
            if i<3 and j<4: f[(i,j)]=IND
            elif i>=3 and j>=2: f[(i,j)]=AMB
            if 2<=i<4 and 2<=j<4: f[(i,j)]=GRN
    tabela(ax,73,22,24,50,6,6,f,"Por transferência (FTL)",AMB)
    ax.text(85,11,"pouco em comum (verde):\ntransfere conhecimento",ha="center",fontsize=8.5)
    save(fig,"f10_particoes")

def f10_vfl():
    fig,ax=canvas(10,4.4)
    cols=[IND,AMB,PNK]; nomes=["Banco\n(tem o rótulo)","Loja","Plano de saúde"]
    for i,(c,n) in enumerate(zip(cols,nomes)):
        y=78-i*28
        rbox(ax,3,y-9,24,18,f"{n}\nX_{i+1}, w_{i+1}",fc="white",ec=c,fs=8.8,lw=2)
        arrow(ax,(27,y+3),(46,52+ (y-50)*0.35+2),c=c,lw=1.8); ax.text(34,y+7+(-2 if i==0 else 0),f"z_{i+1} = X_{i+1}·w_{i+1}",fontsize=8,color=c,fontweight="bold")
    rbox(ax,46,40,22,22,"soma Σ z_k\n→ p → resíduo\np − y\n(só o banco tem y)",fc="#EEF2FF",ec=IND,fs=8.6,bold=True,lw=2)
    for i,c in enumerate(cols):
        y=78-i*28
        rbox(ax,74,y-14 if False else y-9,24,18,"atualiza só\nos próprios pesos",fc="#ECFDF5",ec=GRN,fs=8.5,lw=1.6) if False else None
    arrow(ax,(68,51),(78,51),c=GRN,lw=2.2); ax.text(89,51,"o resíduo volta\na todos; cada um\natualiza só os\npróprios pesos",ha="center",va="center",fontsize=8.6,color=GRN,fontweight="bold")
    save(fig,"f10_vfl")

def f10_split():
    fig,ax=canvas(10,3.8)
    phone(ax,10,60,1.5,IND); ax.text(10,38,"dados brutos\nficam aqui",ha="center",fontsize=8.5,color=IND,fontweight="bold")
    rbox(ax,22,48,18,24,"camadas\niniciais\n(cliente)",fc="#EEF2FF",ec=IND,fs=9,bold=True,lw=2)
    ax.add_patch(Rectangle((48,40),1.0,40,fc=RED,ec=RED,zorder=1)); ax.text(48.5,84,"camada de corte",ha="center",fontsize=8.5,color=RED,fontweight="bold")
    rbox(ax,58,46,36,28,"camadas finais\ne perda\n(servidor, com rótulos)",fc="#ECFDF5",ec=GRN,fs=9,bold=True,lw=2)
    arrow(ax,(40,64),(57,64),c=AMB,lw=2.4); ax.text(48.5,69.5,"ativações",ha="center",fontsize=8.5,color=AMB,fontweight="bold",bbox=dict(fc="white",ec="none",pad=1.5),zorder=5)
    arrow(ax,(57,54),(40,54),c=VIO,lw=2.4); ax.text(48.5,48.5,"gradiente",ha="center",fontsize=8.5,color=VIO,fontweight="bold",bbox=dict(fc="white",ec="none",pad=1.5),zorder=5)
    ax.text(50,22,"passagem direta →  ativações no corte  |  ← gradiente no corte ← retropropagação",ha="center",fontsize=8.8,color=INK)
    save(fig,"f10_split")

def f10_topologias():
    n=10; fig=plt.figure(figsize=(10,3.6)); gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,1.7])
    th=np.linspace(0,2*np.pi,n,endpoint=False); P=np.c_[np.cos(th),np.sin(th)]
    for k,(t,nome) in enumerate([("anel","Anel"),("estrela","Estrela"),("completa","Completa")]):
        ax=fig.add_subplot(gs[k]); A=matriz_mistura(t,n)
        pos=P.copy()
        if t=="estrela": pos[0]=[0,0]
        for i in range(n):
            for j in range(i+1,n):
                if A[i,j]>0 and not (t=="completa" and abs(i-j)%n not in (1,2,n-1,n-2,5)): ax.plot(*zip(pos[i],pos[j]),c=GRY,lw=0.8 if t=="completa" else 1.3,alpha=0.5 if t=="completa" else 0.9)
        ax.scatter(pos[:,0],pos[:,1],s=90,c=[AMB if (t=="estrela" and i==0) else IND for i in range(n)],ec="white",zorder=3); ax.set_title(nome,fontsize=10); ax.axis("off"); ax.set_aspect("equal")
    ax=fig.add_subplot(gs[3]); rng=np.random.default_rng(0); v=rng.normal(0,1,(n,5))
    for t,c in [("completa",GRN),("anel",IND),("estrela",AMB)]:
        e=consenso(v,matriz_mistura(t,n),30); ax.semilogy(np.maximum(e,1e-6),c=c,lw=2,label=t)
    ax.set_xlabel("rodada"); ax.set_ylabel("erro até a média global (log)"); ax.legend(frameon=False,fontsize=8.5); ax.set_title("Consenso",fontsize=10)
    save(fig,"f10_topologias")

if __name__=="__main__":
    for f in [f10_particoes,f10_vfl,f10_split,f10_topologias]: f()
