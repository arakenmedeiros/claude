from icons import *
import sys
sys.path.insert(0,"/home/user/claude/livro_af/cap/lib")
from personalizacao import *

def f6_conceito_ideia():
    fig,axs=plt.subplots(1,3,figsize=(10,3.1),sharex=True,sharey=True)
    rng=np.random.default_rng(4); X=rng.uniform(-1,1,(80,2)); 
    for ax,(t,flip,boundary) in zip(axs,[("Grupo A",1,"A"),("Grupo B (rótulos opostos)",-1,"B"),("Modelo global tenta servir a ambos",0,"G")]):
        if flip!=0:
            y=(flip*X[:,0]>0).astype(int)
            for c,col in enumerate([IND,AMB]): ax.scatter(X[y==c,0],X[y==c,1],c=col,s=26,ec="white",lw=0.4)
            ax.axvline(0,c=INK,ls="--",lw=1.4)
        else:
            y=(X[:,0]>0).astype(int); y2=(-X[:,0]>0).astype(int)
            for c,col in enumerate([IND,AMB]): ax.scatter(X[:,0][(y==c)],X[:,1][(y==c)],c=col,s=26,ec="white",lw=0.4,alpha=0.35)
            ax.scatter(X[:,0],X[:,1],c="none",s=26,ec=GRY,lw=0.4)
            ax.text(0,0,"?",ha="center",va="center",fontsize=44,color=RED,fontweight="bold",alpha=0.85)
        ax.set_title(t,fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])
    save(fig,"f6_conceito_ideia")

def f6_taxonomia():
    fig,ax=canvas(10,4.4)
    rbox(ax,36,84,28,11,"Personalização em\naprendizado federado",fc=INK,tc="white",bold=True,fs=10)
    fam=[("Ajuste fino\nlocal",IND,"FedAvg +\nfine-tuning"),("Regularização",CYA,"Ditto\npFedMe"),("Partição do\nmodelo",GRN,"FedRep\nFedPer"),("Agrupamento",AMB,"IFCA\nCFL"),("Multitarefa",PNK,"MOCHA"),("Meta-\naprendizado",VIO,"Per-FedAvg")]
    for i,(n,c,ex) in enumerate(fam):
        x=1+i*16.5
        arrow(ax,(50,84),(x+7.2,63),c=c,lw=1.5,ms=9)
        rbox(ax,x,45,14.5,18,n,fc=c,tc="white",bold=True,fs=8)
        rbox(ax,x,22,14.5,18,ex,fc="white",ec=c,fs=8.5,lw=1.6)
    ax.text(50,10,"família (em cima) e exemplos de algoritmos (embaixo)",ha="center",fontsize=9,style="italic",color=GRY)
    save(fig,"f6_taxonomia")

def f6_partilha():
    fig,ax=canvas(10,3.9)
    rbox(ax,4,36,40,30,"corpo compartilhado\n(treinado em conjunto,\nenviado ao servidor)",fc="#ECFDF5",ec=GRN,fs=10,bold=True,lw=2.2)
    netglyph(ax,24,76,1.4,GRN)
    arrow(ax,(44,51),(54,51),c=INK,lw=2)
    cores=[IND,RED,AMB]
    for i,c in enumerate(cores):
        y=70-i*22
        rbox(ax,54,y-7,18,14,f"cabeça\ndo cliente {i+1}",fc=c,tc="white",fs=8.8,bold=True)
        arrow(ax,(44,51),(54,y),c=c,lw=1.2,ms=8)
        arrow(ax,(72,y),(82,y),c=c,lw=1.4,ms=8); ax.text(84,y,"previsão\npessoal",fontsize=8.5,va="center",color=c,fontweight="bold")
    ax.text(26,22,"o conhecimento geral é de todos",ha="center",fontsize=9,color=GRN,fontweight="bold")
    ax.text(63,8,"a decisão final é de cada um (e não sai do cliente)",ha="center",fontsize=9,color=INK,fontweight="bold")
    save(fig,"f6_partilha")

def f6_ifca():
    fig,ax=canvas(10,4.0)
    gcol=[IND,RED,AMB]
    for g in range(3):
        for k in range(4):
            x=6+(g*4+k)*7.2; y=70
            ax.add_patch(Circle((x,y),3.2,fc=gcol[g],ec="white",lw=1.5,zorder=3))
            ax.text(x,y-8,f"c{g*4+k+1}",ha="center",fontsize=7.5,color=GRY)
    ax.text(50,86,"clientes (a cor é o grupo verdadeiro, desconhecido do sistema)",ha="center",fontsize=9,color=INK,fontweight="bold")
    for j,c in enumerate(gcol):
        cx=18+j*32
        rbox(ax,cx-11,8,22,14,f"modelo {j+1}",fc="white",ec=c,fs=10,bold=True,lw=2.2)
        for k in range(4):
            gi=j*4+k  # cada grupo termina em um modelo
            x=6+gi*7.2
            arrow(ax,(x,66),(cx+(k-1.5)*3,22.5),c=c,lw=1.1,ms=8)
    ax.text(50,34,"cada cliente escolhe o modelo com menor perda nos seus dados",ha="center",fontsize=9,color=INK,fontweight="bold",bbox=dict(fc="white",ec="none",alpha=0.85,pad=1.5))
    save(fig,"f6_ifca")

def f6_rotulos():
    Xtr,Xte,ytr,yte=carregar_digits()
    R={"só local":[],"global":[],"global +\najuste fino":[],"Ditto":[]}
    for s in range(5):
        partes=particao_dirichlet(ytr,10,0.1,s); tr,te=dividir_local(partes,0.3,s)
        ok=[i for i in range(10) if len(te[i])>=5]; tr=[tr[i] for i in ok]; te=[te[i] for i in ok]
        Wg,_=fedavg(tr,Xtr,ytr,Xte,yte,rodadas=30,epocas=5,eta=0.1,seed=s); z=np.zeros_like(Wg)
        R["só local"].append(acuracias_locais([treino_local(z,Xtr[p],ytr[p],30,0.1,32,0.0,s+i) for i,p in enumerate(tr)],te,Xtr,ytr))
        R["global"].append(acuracias_locais([Wg]*len(tr),te,Xtr,ytr))
        R["global +\najuste fino"].append(acuracias_locais([treino_local(Wg,Xtr[p],ytr[p],5,0.1,32,0.0,s+i) for i,p in enumerate(tr)],te,Xtr,ytr))
        R["Ditto"].append(acuracias_locais(ditto(tr,Xtr,ytr,Wg,0.5,seed=s),te,Xtr,ytr))
    fig,ax=plt.subplots(figsize=(7.4,4.0)); cols=[GRY,IND,GRN,AMB]
    data=[np.concatenate(v) for v in R.values()]
    parts=ax.violinplot(data,showmeans=False,showextrema=False,widths=0.8)
    for pc,c in zip(parts["bodies"],cols): pc.set_facecolor(c); pc.set_alpha(0.35); pc.set_edgecolor(c)
    for i,(d,c) in enumerate(zip(data,cols)):
        ax.scatter(np.random.default_rng(i).normal(i+1,0.05,len(d)),d,s=12,c=c,alpha=0.8,ec="white",lw=0.3)
        ax.scatter(i+1,d.mean(),marker="D",s=70,c="white",ec=INK,zorder=5); ax.text(i+1.12,d.mean(),f"{d.mean():.3f}",fontsize=8.5,va="center",fontweight="bold")
    ax.set_xticks(range(1,5)); ax.set_xticklabels(list(R.keys())); ax.set_ylabel("acurácia no teste local do cliente"); ax.set_ylim(0.1,1.03)
    ax.set_title("Cada ponto é um cliente (5 sementes); losango = média",fontsize=9.5)
    save(fig,"f6_rotulos")

def f6_conceito():
    Xtr,Xte,ytr0,yte0=carregar_digits(); A={"global":[],"só local":[],"ajuste fino\n(5 épocas)":[],"IFCA\n(K = 3)":[]}
    for s in range(3):
        partes,y,grupo=particao_conceitos(ytr0,12,3,s); tr,te=dividir_local(partes,0.3,s)
        Wg,_=fedavg(tr,Xtr,y,Xte,yte0,rodadas=30,epocas=5,eta=0.1,seed=s)
        loc=[treino_local(np.zeros_like(Wg),Xtr[p],y[p],30,0.1,32,0.0,i) for i,p in enumerate(tr)]
        ft=[treino_local(Wg,Xtr[p],y[p],5,0.1,32,0.0,i) for i,p in enumerate(tr)]
        M,esc=ifca(tr,Xtr,y,3,seed=s); cl=[M[j] for j in esc]
        for k,Ws in zip(A.keys(),[[Wg]*12,loc,ft,cl]): A[k].append(acuracias_locais(Ws,te,Xtr,y).mean())
    fig,ax=plt.subplots(figsize=(6.8,3.9)); cols=[IND,GRY,AMB,GRN]
    for i,((k,v),c) in enumerate(zip(A.items(),cols)):
        ax.bar(i,np.mean(v),color=c,yerr=np.std(v),capsize=4,ec=INK,lw=0.6); ax.text(i,np.mean(v)+0.04,f"{np.mean(v):.2f}",ha="center",fontweight="bold",fontsize=9.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(list(A.keys())); ax.set_ylabel("acurácia média por cliente"); ax.set_ylim(0,1.08)
    ax.set_title("Mudança de conceito (3 grupos, 12 clientes, 3 sementes)",fontsize=9.5)
    save(fig,"f6_conceito")

if __name__=="__main__":
    for f in [f6_conceito_ideia,f6_taxonomia,f6_partilha,f6_ifca,f6_rotulos,f6_conceito]: f()
