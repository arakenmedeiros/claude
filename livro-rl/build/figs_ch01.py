from fc import *
from figs_cover import star
import gymnasium as gym
from sklearn.tree import DecisionTreeClassifier

def f1_paradigmas():
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    rng=np.random.default_rng(3)
    # supervisionado
    ax=axs[0]; A=rng.normal([-1,0],0.6,(30,2)); B=rng.normal([1,0.3],0.6,(30,2))
    ax.scatter(*A.T,c=IND,s=28,label="gato"); ax.scatter(*B.T,c=AMB,s=28,label="cão"); ax.plot([0,0.1],[-1.8,2],c=INK,ls="--",lw=1.5)
    ax.set_title("Supervisionado\n\"aprenda com o gabarito\"",color=IND); ax.legend(frameon=False,fontsize=8,loc="lower right"); clean(ax,True)
    # nao supervisionado
    ax=axs[1]; P=np.vstack([rng.normal(c,0.35,(25,2)) for c in [(-1,-.6),(1,-.5),(0,1.1)]])
    ax.scatter(*P.T,c=GRY,s=28)
    for c,col in zip([(-1,-.6),(1,-.5),(0,1.1)],[CYA,GRN,PNK]): ax.add_patch(Ellipse(c,1.6,1.3,fc=col,alpha=0.18,ec=col,lw=1.5))
    ax.set_title("Não supervisionado\n\"descubra a estrutura\"",color=CYA); clean(ax,True)
    # reforco
    ax=axs[2]; ax.set_xlim(0,10); ax.set_ylim(0,10); clean(ax,True)
    rbox(ax,0.8,6.2,3.6,2.4,"Agente",fc=IND,tc="white",bold=True,fs=10); rbox(ax,5.6,1.4,3.6,2.4,"Ambiente",fc=GRN,tc="white",bold=True,fs=10)
    arrow(ax,(4.4,7.4),(8.6,3.9),c=INK,rad=-0.3); arrow(ax,(5.6,2.6),(1.6,6.1),c=AMB,rad=-0.3)
    ax.text(7.7,6.6,"ação",fontsize=9,fontweight="bold"); ax.text(1.0,3.2,"recompensa\n+ estado",fontsize=9,fontweight="bold",color="#B45309")
    ax.set_title("Por reforço\n\"aprenda tentando\"",color=AMB)
    for a in axs: a.title.set_fontsize(10.5)
    save(fig,"f1_paradigmas")

def f1_ciclo():
    fig,ax=canvas(9.5,4.6)
    rbox(ax,3,58,28,28,"",fc="#EEF2FF",ec=IND,lw=2,r=3); ax.text(17,79,"AGENTE",ha="center",fontsize=13,fontweight="bold",color=IND)
    ax.text(17,69,"quem decide\n(robô aspirador)",ha="center",va="center",fontsize=9.5)
    rbox(ax,69,58,28,28,"",fc="#ECFDF5",ec=GRN,lw=2,r=3); ax.text(83,79,"AMBIENTE",ha="center",fontsize=13,fontweight="bold",color=GRN)
    ax.text(83,69,"tudo que reage\n(sala, móveis,\nsujeira)",ha="center",va="center",fontsize=9.5)
    arrow(ax,(31,80),(69,80),c=INK,lw=2.2); ax.text(50,83,"ação  A$_t$",ha="center",fontsize=10,fontweight="bold")

    arrow(ax,(69,70),(31,70),c=CYA,lw=2.2); ax.text(50,72,"estado  S$_{t+1}$",ha="center",fontsize=10,fontweight="bold",color="#0E7490")
    arrow(ax,(69,62),(31,62),c=AMB,lw=2.2); ax.text(50,64,"recompensa  R$_{t+1}$",ha="center",fontsize=10,fontweight="bold",color="#B45309")
    # exemplos
    items=[("Estado","o que o agente percebe","“há sujeira à frente,\nbateria em 40%”",CYA),("Ação","o que o agente pode fazer","avançar, girar, voltar à base",INK),("Recompensa","um número que diz se foi bom","+1 por sujeira coletada,\n−10 ao bater",AMB)]
    for k,(t,d,e,c) in enumerate(items):
        x=4+k*32; rbox(ax,x,6,30,38,"",fc="white",ec=c,lw=1.8,r=2.5)
        ax.text(x+15,38,t,ha="center",fontsize=11.5,fontweight="bold",color=c if c!=INK else INK)
        ax.text(x+15,29,d,ha="center",va="center",fontsize=8.8,color=INK)
        ax.text(x+15,15,e,ha="center",va="center",fontsize=8.5,style="italic",color=GRY,linespacing=1.4)
    ax.text(50,95,"O ciclo do aprendizado por reforço",ha="center",fontsize=12,fontweight="bold",color=IND)
    save(fig,"f1_ciclo")

def f1_componentes():
    fig,ax=canvas(10,3.6)
    cards=[("POLÍTICA  π","“O que eu faço aqui?”","Regra que liga estados a ações.\nEx.: “se há sujeira,\navance”.",IND),
           ("FUNÇÃO VALOR  v","“Quão bom é estar aqui?”","Recompensa total esperada\na partir de um estado.\nEx.: “perto da sujeira\nvale muito”.",GRN),
           ("MODELO  p","“O que vai acontecer?”","Previsão do próximo\nestado e da recompensa.\nÉ opcional!",AMB)]
    for k,(t,q,d,c) in enumerate(cards):
        x=2+k*33; rbox(ax,x,6,31,86,"",fc="white",ec=c,lw=2.2,r=3); rbox(ax,x,72,31,20,t,fc=c,tc="white",bold=True,fs=11,r=3)
        ax.text(x+15.5,58,q,ha="center",fontsize=9.6,fontweight="bold",color=c)
        ax.text(x+15.5,35,d,ha="center",va="center",fontsize=8.8,linespacing=1.5)
    save(fig,"f1_componentes")

def f1_desconto():
    fig,axs=plt.subplots(1,2,figsize=(10,3.6)); t=np.arange(0,31)
    for g,c in [(0.5,RED),(0.9,AMB),(0.99,GRN)]: axs[0].plot(t,g**t,c=c,lw=2.2,label=f"γ = {g}")
    axs[0].set_xlabel("passos no futuro (k)"); axs[0].set_ylabel("peso γᵏ da recompensa"); axs[0].set_title("Quanto vale uma recompensa daqui a k passos?"); axs[0].legend(frameon=False)
    axs[0].fill_between(t,0.9**t,alpha=0.12,color=AMB)
    # retorno de 10 recompensas iguais a 1
    gs=np.linspace(0,0.99,100); G=[(1-g**10)/(1-g) if g<1 else 10 for g in gs]
    axs[1].plot(gs,G,c=IND,lw=2.5); axs[1].set_xlabel("fator de desconto γ"); axs[1].set_ylabel("retorno de 10 recompensas = 1"); axs[1].set_title("γ pequeno = míope; γ grande = visionário")
    axs[1].axvspan(0,0.3,color=RED,alpha=0.07); axs[1].axvspan(0.8,1,color=GRN,alpha=0.08)
    axs[1].text(0.15,2,"míope",ha="center",color=RED,fontweight="bold"); axs[1].text(0.9,2,"visionário",ha="center",color="#047857",fontweight="bold")
    save(fig,"f1_desconto")

def f1_credito():
    fig,ax=canvas(10,3.4)
    n=9; xs=np.linspace(7,93,n)
    for i,x in enumerate(xs):
        c=IND if i<n-1 else AMB
        dot(ax,x,50,s=620,c=c); ax.text(x,50,f"a{i+1}" if i<n-1 else "★",ha="center",va="center",color="white",fontweight="bold",fontsize=10)
        if i<n-1: arrow(ax,(x+3.6,50),(xs[i+1]-3.8,50),c=GRY,lw=1.4,ms=9)
    ax.text(xs[:-1].mean(),72,"jogadas com recompensa 0 (nenhum feedback)",ha="center",fontsize=10,color=GRY)
    ax.text(xs[-1],72,"+1\nvitória!",ha="center",fontsize=10,fontweight="bold",color="#B45309")
    arrow(ax,(xs[-1],41),(xs[3],41),c=RED,rad=-0.25,lw=2); ax.text(52,2,"Quem merece o crédito? Uma jogada decisiva lá atrás?\nUm erro do adversário? Esse é o problema da atribuição de crédito.",ha="center",fontsize=9.8,color=RED,fontweight="bold",linespacing=1.4)
    save(fig,"f1_credito")

def f1_timeline():
    fig,ax=canvas(11,4.2,(0,100),(0,100))
    ev=[(1911,"Thorndike","Lei do efeito"),(1957,"Bellman","Programação\ndinâmica"),(1959,"Samuel","Damas que\naprendem"),(1988,"Sutton","Diferença\ntemporal"),(1992,"Watkins\n& Dayan","Q-learning"),(1995,"Tesauro","TD-Gammon"),(2015,"Mnih et al.","DQN\n(Atari)"),(2016,"Silver et al.","AlphaGo"),(2017,"Schulman et al.","PPO"),(2022,"Ouyang et al.","RLHF\n(InstructGPT)")]
    ax.plot([3,97],[50,50],c=INK,lw=3,zorder=1)
    x_of=lambda i:5+i*(90/(len(ev)-1))
    cols=[IND,CYA,GRN,AMB,RED,PNK,VIO,IND,CYA,GRN]
    for i,(y,w,d) in enumerate(ev):
        x=x_of(i); up=i%2==0; c=cols[i]
        dot(ax,x,50,s=180,c=c,lw=2); ax.plot([x,x],[50,50+(12 if up else -12)],c=c,lw=1.6)
        yy=50+(14 if up else -14)
        ax.text(x,yy+(1 if up else -1),str(y),ha="center",va="bottom" if up else "top",fontsize=12,fontweight="bold",color=c)
        ax.text(x,yy+(8 if up else -8),f"{w}\n{d}",ha="center",va="bottom" if up else "top",fontsize=8.2,linespacing=1.3)
    save(fig,"f1_timeline")

def cem_run(seed=0,iters=15,N=100):
    env=gym.make("CartPole-v1"); rng=np.random.default_rng(seed); pol=None; hist=[]; last=None
    for it in range(iters):
        R=[];D=[]
        for _ in range(N):
            o,_=env.reset(seed=int(rng.integers(1e9))); tot=0;ob=[];ac=[];done=False
            while not done:
                if pol is None: a=int(rng.integers(2))
                else:
                    p=pol.predict_proba(o[None])[0]; a=int(rng.choice(pol.classes_,p=p))
                ob.append(o);ac.append(a);o,r,t,u,_=env.step(a);tot+=r;done=t or u
            R.append(tot);D.append((np.array(ob),np.array(ac)))
        R=np.array(R); hist.append(R); thr=np.percentile(R,70)
        X=np.vstack([d[0] for d,r in zip(D,R) if r>=thr]); y=np.concatenate([d[1] for d,r in zip(D,R) if r>=thr])
        pol=DecisionTreeClassifier(max_depth=4,random_state=0).fit(X,y)
    return hist
def f1_cartpole():
    H=cem_run(); m=[h.mean() for h in H]; lo=[np.percentile(h,25) for h in H]; hi=[np.percentile(h,75) for h in H]
    fig,axs=plt.subplots(1,2,figsize=(10,3.6)); x=np.arange(1,len(H)+1)
    axs[0].fill_between(x,lo,hi,color=IND,alpha=0.18,label="intervalo 25%–75%"); axs[0].plot(x,m,c=IND,lw=2.5,marker="o",ms=4,label="retorno médio")
    axs[0].axhline(500,c=GRN,ls="--",lw=1.2); axs[0].text(1.2,505,"máximo possível (500)",color="#047857",fontsize=8.5)
    axs[0].set_xlabel("iteração do método da entropia cruzada"); axs[0].set_ylabel("retorno por episódio"); axs[0].set_title("O agente melhora a cada iteração"); axs[0].legend(frameon=False,loc="upper left",bbox_to_anchor=(0,0.88),fontsize=8.5)
    axs[1].hist(H[0],bins=np.arange(0,520,20),color=RED,alpha=0.75,label="iteração 1 (aleatório)"); axs[1].hist(H[-1],bins=np.arange(0,520,20),color=GRN,alpha=0.75,label=f"iteração {len(H)}")
    axs[1].set_xlabel("retorno do episódio"); axs[1].set_ylabel("nº de episódios"); axs[1].set_title("Antes × depois de aprender"); axs[1].legend(frameon=False,fontsize=8.5)
    save(fig,"f1_cartpole")
if __name__=="__main__":
    for f in [f1_paradigmas,f1_ciclo,f1_componentes,f1_desconto,f1_credito,f1_timeline,f1_cartpole]: f()
