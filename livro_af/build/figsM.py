from icons import *
import sys
sys.path.insert(0,"/home/user/claude/livro_af/cap/lib")
from projeto import *

def f12_etapas():
    fig,ax=canvas(10,4.6)
    et=[("1","Diagnóstico","heterogeneidade",IND),("2","Linha de base","FedAvg",IND),("3","Ameaça","backdoor, ASR",RED),("4","Robustez","mediana, aparada",RED),("5","Comunicação","8 e 4 bits",CYA),("6","Privacidade","DP, ε",VIO),("7","Personalização","ajuste fino",GRN),("8","Síntese","tabela, critério",AMB),("9","Relatório","parecer",INK)]
    pos=[(14,78),(50,78),(86,78),(86,50),(50,50),(14,50),(14,22),(50,22),(86,22)]
    for (n,t,d,c),(x,y) in zip(et,pos):
        rbox(ax,x-13,y-9,26,18,"",fc="white",ec=c,lw=2.2,r=2)
        ax.add_patch(Circle((x-9,y+4.5),3.0,fc=c,ec="white",lw=1.2,zorder=4)); ax.text(x-9,y+4.5,n,ha="center",va="center",fontsize=9.5,fontweight="bold",color="white",zorder=5)
        ax.text(x+2,y+3.5,t,ha="center",va="center",fontsize=9.3,fontweight="bold",color=INK); ax.text(x,y-4,d,ha="center",va="center",fontsize=8.2,color=GRY)
    for (x1,y1),(x2,y2) in zip(pos[:-1],pos[1:]):
        if y1==y2: arrow(ax,(x1+(13.4 if x2>x1 else -13.4),y1),(x2-(13.4 if x2>x1 else -13.4),y2),c=INK,lw=1.5,ms=9)
        else: arrow(ax,(x1,y1-9.4),(x2,y2+9.4),c=INK,lw=1.5,ms=9)
    save(fig,"f12_etapas")

def f12_resultados():
    Xtr,Xte,ytr,yte=carregar_digits(); K=20
    cfg=[("base\n(média)",dict(agregador="media"),False),("mediana",dict(agregador="mediana"),False),("mediana\n8 bits",dict(agregador="mediana",bits=8),False),("mediana\n4 bits",dict(agregador="mediana",bits=4),False),("DP\n(z=1)",dict(dp_z=1.0),False),("mediana\n4 bits +\najuste fino",dict(agregador="mediana",bits=4),True)]
    R=[]
    for n,kw,aj in cfg:
        L=[]
        for s in range(3):
            partes=particao_dirichlet(ytr,K,0.3,s); tr,te=dividir_local(partes,0.3,s)
            W,acc,asr,by=fedavg_integrado(tr,Xtr,ytr,Xte,yte,maliciosos=(0,1),seed=s,**kw)
            ok=[i for i in range(K) if len(te[i])>=5]
            ms=[treino_local(W,Xtr[tr[i]],ytr[tr[i]],3,0.1,32,0.0,i) if aj else W for i in ok]
            pc=acuracias_locais(ms,[te[i] for i in ok],Xtr,ytr); L.append((acc,asr,by/1024,pc.mean()))
        R.append(np.mean(L,0))
    R=np.array(R); nomes=[c[0] for c in cfg]; x=np.arange(len(cfg))
    fig,axs=plt.subplots(1,3,figsize=(10,3.6))
    cols=[RED,GRN,GRN,GRN,AMB,GRN]
    axs[0].bar(x,R[:,1],color=cols,ec=INK,lw=0.5); axs[0].axhline(0.10,c=GRY,ls="--"); axs[0].set_title("ASR do backdoor (menor é melhor)",fontsize=9.5)
    axs[1].bar(x,R[:,2],color=[IND,IND,CYA,CYA,IND,CYA],ec=INK,lw=0.5); axs[1].set_title("Tráfego de subida (KB)",fontsize=9.5)
    axs[2].bar(x,R[:,3],color=[IND,PNK,PNK,PNK,PNK,GRN],ec=INK,lw=0.5); axs[2].set_ylim(0.8,0.97); axs[2].set_title("Acurácia média por cliente",fontsize=9.5)
    for a in axs: a.set_xticks(x); a.set_xticklabels([str(i+1) for i in range(len(nomes))],fontsize=9); a.set_xlabel("configuração")
    save(fig,"f12_resultados")

if __name__=="__main__":
    f12_etapas(); f12_resultados()
