from fc import *
import os
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

ZF="/home/user/claude/automl/build/z_svc.npz"
def landscape():
    if os.path.exists(ZF):
        d=np.load(ZF); return d["Cs"],d["gs"],d["Z"]
    X,y=load_digits(return_X_y=True); X=X/16.0; cv=StratifiedKFold(5,shuffle=True,random_state=0)
    Cs=np.logspace(-2,3,13); gs=np.logspace(-5,0,13)
    Z=np.array([[cross_val_score(SVC(C=c,gamma=g),X,y,cv=cv).mean() for g in gs] for c in Cs])
    np.savez(ZF,Cs=Cs,gs=gs,Z=Z); return Cs,gs,Z

def f4_botoes():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"Hiperparâmetros são os botões do algoritmo",ha="center",fontsize=11,fontweight="bold")
    panels=[("Árvore de decisão",IND,[("profundidade\nmáxima",0.35),("amostras\npor folha",0.2)]),("k-vizinhos",CYA,[("nº de\nvizinhos",0.5),("tipo de\npeso",0.8)]),("SVM (kernel RBF)",GRN,[("C\n(rigidez)",0.7),("gamma\n(alcance)",0.3)]),("Floresta aleatória",AMB,[("nº de\nárvores",0.8),("atributos\npor divisão",0.4)])]
    for i,(t,c,knobs) in enumerate(panels):
        x=2+i*24.5
        rbox(ax,x,16,23,71,fc="#F9FAFB",ec=c,lw=2,r=2); ax.text(x+11.5,81.5,t,ha="center",fontsize=8.8,fontweight="bold",color=c)
        for j,(lb,v) in enumerate(knobs):
            cx=x+11.5; cy=67-j*28
            th=np.linspace(np.pi*0.8,np.pi*0.2,60)*-1+np.pi  # arc
            ang=np.linspace(-np.pi*0.8,np.pi*0.8,60)
            ax.add_patch(Circle((cx,cy),7,fc="white",ec=INK,lw=1.6))
            for a in np.linspace(-0.8*np.pi,0.8*np.pi,9): ax.plot([cx+7.6*np.sin(a),cx+9*np.sin(a)],[cy+7.6*np.cos(a),cy+9*np.cos(a)],c=GRY,lw=1)
            a=-0.8*np.pi+v*1.6*np.pi; ax.plot([cx,cx+6*np.sin(a)],[cy,cy+6*np.cos(a)],c=c,lw=3,solid_capstyle="round"); ax.add_patch(Circle((cx,cy),1.1,fc=INK))
            ax.text(cx,cy-10.5,lb,ha="center",va="top",fontsize=7.6,color=INK,linespacing=1.05)
    ax.text(50,9,"Você escolhe a posição de cada botão ANTES do treino. O AutoML procura as melhores posições.",ha="center",fontsize=9,color=INK)
    ax.text(50,3,"(Ilustração: as posições dos botões são apenas exemplos.)",ha="center",fontsize=7.5,color=GRY,style="italic")
    save(fig,"f4_botoes")

def f4_tipos():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"Tipos de hiperparâmetro: cada um pede um jeito de sortear",ha="center",fontsize=11,fontweight="bold")
    T=[("Contínuo","taxa de aprendizado\n0,001 a 1",IND),("Inteiro","nº de árvores\n10 a 500",CYA),("Categórico","kernel: linear,\nrbf, poly",GRN),("Em escala log","C do SVM\n0,01 a 1000",AMB),("Condicional","gamma só existe\nse kernel = rbf",RED)]
    for i,(t,d,c) in enumerate(T):
        x=2+i*19.6; rbox(ax,x,60,18,28,"",fc="white",ec=c,lw=2,r=1.5); ax.text(x+9,82,t,ha="center",fontsize=9,fontweight="bold",color=c); ax.text(x+9,69,d,ha="center",fontsize=7.8,color=INK,va="center",linespacing=1.2)
        if i==0: ax.plot([x+2,x+16],[63,63],c=c,lw=3,solid_capstyle="round"); ax.scatter([x+10.5],[63],s=60,c=c,zorder=3)
        if i==1: ax.scatter(np.linspace(x+2,x+16,8),[63]*8,s=22,c=c)
        if i==2:
            for k in range(3): ax.scatter([x+4+k*5],[63],s=50,marker="s",c=[c,GRY,GRY][k])
        if i==3:
            for k in range(6): ax.plot([x+2+k*2.8]*2,[62,64.5],c=c,lw=1.5)
            ax.plot([x+2,x+16],[63,63],c=c,lw=1)
        if i==4: rbox(ax,x+3,61,5,4,"",fc=c,ec=c,r=0.3); ax.annotate("",xy=(x+14,63),xytext=(x+8.5,63),arrowprops=dict(arrowstyle="-|>",color=c))
    ax.text(50,48,"Sorteando em escala logarítmica, cada \"ordem de grandeza\" tem a mesma chance:",ha="center",fontsize=9,color=INK)
    for k,v in enumerate([0.01,0.1,1,10,100,1000]):
        x=18+k*13; rbox(ax,x,28,11,10,f"{v:g}".replace(".",","),fc=AMB,ec=AMB,fs=9,tc="white",bold=True,r=1); 
    ax.text(50,18,"0,01, 0,1, 1, 10, 100, 1000: um sorteio uniforme de 0,01 a 1000 quase nunca visitaria os valores pequenos.",ha="center",fontsize=8.5,color=GRY,style="italic")
    save(fig,"f4_tipos")

def f4_gridrandom():
    fig,axs=plt.subplots(1,2,figsize=(8.6,4.0))
    rng=np.random.default_rng(2)
    xs=np.linspace(0,1,200)
    imp=lambda x: np.exp(-((x-0.62)/0.13)**2)   # parâmetro importante (horizontal)
    for ax,tp in zip(axs,("grade","aleatória")):
        if tp=="grade":
            g=np.array([0.17,0.5,0.83]); P=np.array([(a,b) for a in g for b in g])
        else:
            P=rng.uniform(0.05,0.95,(9,2)); 
            P[0,0]=0.62
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Busca em "+tp+" (9 tentativas)",fontsize=9.5,fontweight="bold",color=IND if tp=="grade" else GRN)
        ax.fill_between(xs,1.0,1.0+0*xs)
        # importante: eixo x; fundo gradiente
        xx,yy=np.meshgrid(xs,np.linspace(0,1,50)); ax.imshow(imp(xx),extent=[0,1,0,1],origin="lower",cmap=LinearSegmentedColormap.from_list("a",["white","#FDE68A"]),alpha=0.7,aspect="auto")
        ax.scatter(P[:,0],P[:,1],s=70,c=IND if tp=="grade" else GRN,edgecolors="white",zorder=3)
        # marginais no topo
        for x0 in P[:,0]: ax.plot([x0,x0],[-0.0,-0.04],c=INK,lw=1.2,clip_on=False)
        ax.text(0.5,-0.08,f"{len(np.unique(np.round(P[:,0],3)))} valores distintos do parâmetro importante",ha="center",fontsize=8.5,color=INK,transform=ax.transAxes,fontweight="bold")
        ax.set_xlabel("")
        ax.text(0.62,1.02,"faixa boa",ha="center",fontsize=8,color="#B45309",transform=ax.transAxes) if False else None
    axs[0].set_ylabel("parâmetro sem importância",fontsize=8.5); axs[1].set_ylabel("")
    fig.text(0.5,0.0,"Horizontal = parâmetro que importa (faixa amarela é a região boa). Vertical = parâmetro quase irrelevante.",ha="center",fontsize=8.3,color=GRY,style="italic")
    fig.tight_layout(rect=(0,0.04,1,1))
    save(fig,"f4_gridrandom")

def f4_explosao():
    fig,ax=plt.subplots(figsize=(8.0,3.6))
    d=np.arange(1,11); ax.semilogy(d,5.0**d,"o-",c=RED,lw=2.5,label="5 valores por hiperparâmetro")
    ax.semilogy(d,3.0**d,"s-",c=AMB,lw=2.5,label="3 valores por hiperparâmetro")
    ax.axhline(1000,c=GRY,ls="--",lw=1); ax.text(1.1,1500,"1.000 avaliações",fontsize=8,color=GRY)
    ax.axhline(1e6,c=GRY,ls="--",lw=1); ax.text(1.1,1.5e6,"1 milhão de avaliações",fontsize=8,color=GRY)
    ax.annotate("grade com 10 hiperparâmetros\ne 5 valores cada: ~9,8 milhões",(10,5.0**10),(3.2,2e7),arrowprops=dict(arrowstyle="->",color=RED),fontsize=8.5,color=RED,fontweight="bold")
    ax.set_xlabel("número de hiperparâmetros"); ax.set_ylabel("avaliações necessárias (escala log)"); ax.legend(frameon=False,loc="lower right"); ax.set_xticks(d); ax.set_ylim(1,3e8)
    ax.set_title("A explosão da busca em grade",fontweight="bold")
    save(fig,"f4_explosao")

def f4_paisagem():
    Cs,gs,Z=landscape(); rng=np.random.default_rng(5)
    fig,axs=plt.subplots(1,2,figsize=(8.6,3.9),sharey=True)
    cm=LinearSegmentedColormap.from_list("p",["#1E1B4B",IND,CYA,"#A7F3D0","#FEF9C3"])
    X,y=load_digits(return_X_y=True); X=X/16.0; cv=StratifiedKFold(5,shuffle=True,random_state=0)
    ev=lambda c,g: cross_val_score(SVC(C=c,gamma=g),X,y,cv=cv).mean()
    gC=np.logspace(-2,3,4); gG=np.logspace(-5,0,4)
    PG=[(c,g) for c in gC for g in gG]
    PR=[(10**rng.uniform(-2,3),10**rng.uniform(-5,0)) for _ in range(16)]
    for ax,P,t,col in zip(axs,(PG,PR),("Grade 4 × 4 (16 tentativas)","Aleatória (16 tentativas)"),(IND,GRN)):
        im=ax.pcolormesh(np.log10(gs),np.log10(Cs),Z,cmap=cm,vmin=0.1,vmax=1.0,shading="nearest")
        sc=[ev(c,g) for c,g in P]; b=int(np.argmax(sc))
        ax.scatter([np.log10(g) for c,g in P],[np.log10(c) for c,g in P],s=34,c="white",edgecolors=col,linewidths=1.8,zorder=3)
        ax.scatter([np.log10(P[b][1])],[np.log10(P[b][0])],s=210,marker="*",c=AMB,edgecolors="white",zorder=4)
        ax.set_title(f"{t}\nmelhor acurácia = {max(sc):.3f}",fontsize=9,fontweight="bold",color=col); ax.set_xlabel("log10(gamma)",fontsize=8.5)
    axs[0].set_ylabel("log10(C)",fontsize=8.5)
    cb=fig.colorbar(im,ax=axs,fraction=0.03,pad=0.02); cb.set_label("acurácia (validação cruzada)",fontsize=8)
    fig.suptitle("A paisagem de hiperparâmetros de um SVM (conjunto Digits)",fontsize=10.5,fontweight="bold",y=1.03)
    save(fig,"f4_paisagem",tight=True)

def f4_aninhada():
    fig,ax=canvas(8.6,4.1)
    ax.text(50,97,"Validação aninhada: um laço dentro do outro",ha="center",fontsize=11,fontweight="bold")
    ax.text(3,86,"Laço EXTERNO\n(mede o desempenho)",fontsize=8.5,fontweight="bold",color=RED,va="center")
    for k in range(5):
        rbox(ax,28+k*13.5,80,12.5,12,"TESTE" if k==2 else "treino",fc=RED if k==2 else "#C7D2FE",ec="white",fs=8.5,tc="white" if k==2 else INK,bold=(k==2),r=1)
    ax.text(50,72,"na rodada 3 do laço externo, as 4 partes de treino são usadas para BUSCAR:",ha="center",fontsize=8.5,color=INK)
    ax.text(3,52,"Laço INTERNO\n(escolhe os\nhiperparâmetros)",fontsize=8.5,fontweight="bold",color=IND,va="center")
    labs=["treino","treino","treino","VALID."]
    for r in range(3):
        for k in range(4):
            isv=(k==r)
            rbox(ax,28+k*13.5,60-r*11,12.5,8.5,"VALID." if isv else "treino",fc=AMB if isv else "#E0E7FF",ec="white",fs=7.5,tc="white" if isv else INK,bold=isv,r=0.8)
    ax.text(87,50,"escolhe\no melhor\nconjunto de\nbotões",fontsize=8.3,color=IND,va="center",ha="center",fontweight="bold")
    ax.text(50,15,"O teste do laço externo nunca participa da escolha dos hiperparâmetros.",ha="center",fontsize=9,color=INK)
    ax.text(50,7,"A nota é a média dos testes externos: uma estimativa honesta do desempenho de TODO o processo de busca.",ha="center",fontsize=8.5,color=GRY,style="italic")
    save(fig,"f4_aninhada")

def f4_orcamento():
    Cs,gs,Z=landscape(); v=Z.ravel(); rng=np.random.default_rng(1)
    K=40; runs=np.array([np.maximum.accumulate(rng.choice(v,K,replace=True)) for _ in range(400)])
    fig,ax=plt.subplots(figsize=(8.0,3.5)); k=np.arange(1,K+1)
    ax.fill_between(k,np.percentile(runs,10,0),np.percentile(runs,90,0),color=CYA,alpha=0.25,label="faixa de 10% a 90%")
    ax.plot(k,runs.mean(0),c=IND,lw=2.5,label="média da busca aleatória"); ax.axhline(v.max(),c=GRN,ls="--",lw=1.3); ax.text(K,v.max()+0.004,"melhor da paisagem",ha="right",fontsize=8,color=GRN)
    ax.set_xlabel("número de tentativas"); ax.set_ylabel("melhor acurácia até agora"); ax.set_ylim(0.8,1.005); ax.legend(frameon=False,loc="center right")
    ax.text(K*0.5,0.81,"(no início, a faixa de 10% chega a valores bem menores e sai do gráfico)",ha="center",fontsize=7.5,color=GRY,style="italic")
    ax.set_title("Retornos decrescentes: as primeiras tentativas valem mais",fontweight="bold")
    save(fig,"f4_orcamento")

if __name__=="__main__":
    for f in (f4_botoes,f4_tipos,f4_gridrandom,f4_explosao,f4_paisagem,f4_aninhada,f4_orcamento): f()
