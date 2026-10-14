from fc import *
from figs_cover import star
import gymnasium as gym, time
from matplotlib.colors import LinearSegmentedColormap

def f6_dirigindo():
    fig,ax=plt.subplots(figsize=(8.6,3.6))
    rot=["sair do\nescritório","pegar a\navenida","trânsito\nparado!","saída da\ncidade","chegar\nem casa"]; prev=[30,31,41,36,33]; x=np.arange(5)
    ax.plot(x,prev,c=IND,lw=2.4,marker="o",ms=8,label="previsão do TD (atualizada a cada trecho)"); ax.axhline(33,c=RED,ls="--",lw=1.5,label="tempo real da viagem (33 min)")
    ax.scatter([4],[33],s=260,marker="*",c=AMB,zorder=4,label="único momento em que o MC aprende")
    ax.set_xticks(x); ax.set_xticklabels(rot,fontsize=9); ax.set_ylim(24,48); ax.set_ylabel("previsão do tempo total (min)")
    ax.set_title("“Quanto tempo vou levar?”: o TD corrige a previsão a cada trecho",fontsize=10.5)
    ax.annotate("o trânsito já faz a previsão subir",xy=(2,41),xytext=(2.4,45),arrowprops=dict(arrowstyle="->",color=IND),fontsize=9,color=IND,fontweight="bold")
    ax.legend(frameon=False,fontsize=8.5,loc="lower left"); save(fig,"f6_dirigindo")

def f6_td_erro():
    fig,ax=canvas(10,3.4)
    rbox(ax,3,30,20,40,"V(S)\nestimativa\natual",fc="#EEF2FF",ec=IND,lw=2,r=3,fs=10,bold=True)
    rbox(ax,36,52,28,26,"alvo TD\nR + γ·V(S′)",fc="#FFFBEB",ec=AMB,lw=2,r=3,fs=10,bold=True)
    rbox(ax,36,14,28,26,"erro TD\nδ = alvo − V(S)",fc="#FEF2F2",ec=RED,lw=2,r=3,fs=10,bold=True)
    rbox(ax,77,30,20,40,"V(S)\nnova\nestimativa",fc="#ECFDF5",ec=GRN,lw=2,r=3,fs=10,bold=True)
    arrow(ax,(23,60),(36,65),c=GRY); arrow(ax,(23,40),(36,28),c=GRY); arrow(ax,(50,52),(50,40.5),c=AMB,lw=2)
    arrow(ax,(64,27),(77,42),c=RED,lw=2); ax.text(73,27,"× α",fontsize=11,fontweight="bold",color=RED)
    arrow(ax,(64,65),(77,58),c=GRY); ax.text(50,92,"V(S) ← V(S) + α · [ R + γ·V(S′) − V(S) ]     “novo = velho + passo × erro”",ha="center",fontsize=11,fontweight="bold",color=IND)
    save(fig,"f6_td_erro")

def passeio(alg,alpha,n_ep=100,semente=0):
    rng=np.random.default_rng(semente); V=np.full(7,0.5); V[0]=V[6]=0; verdade=np.arange(1,6)/6; erros=[]; 
    for ep in range(n_ep):
        s=3; traj=[s]
        while s not in (0,6):
            s+=int(rng.choice([-1,1])); traj.append(s)
            if alg=="td":
                r=1.0 if s==6 else 0.0; V[traj[-2]]+=alpha*(r+V[s]-V[traj[-2]])
        if alg=="mc":
            G=1.0 if traj[-1]==6 else 0.0
            for st in set(traj[:-1]): V[st]+=alpha*(G-V[st])
        erros.append(np.sqrt(np.mean((V[1:6]-verdade)**2)))
    return np.array(erros)
def f6_td_vs_mc():
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.6))
    ax=axs[0]; ax.set_xlim(0,10); ax.set_ylim(0,4); ax.axis("off"); ax.set_title("Passeio aleatório de 5 estados")
    for i,l in enumerate("ABCDE"):
        ax.add_patch(Rectangle((1+i*1.5-0.0,1.5),1.2,1.2,fc="#EEF2FF",ec=IND,lw=2)); ax.text(1.6+i*1.5,2.1,l,ha="center",va="center",fontweight="bold",fontsize=11)
    ax.add_patch(Rectangle((0,1.5),0.8,1.2,fc="#FEE2E2",ec=RED,lw=2)); ax.text(0.4,2.1,"0",ha="center",va="center",fontweight="bold",color=RED)
    ax.add_patch(Rectangle((8.7,1.5),0.8,1.2,fc="#D1FAE5",ec=GRN,lw=2)); ax.text(9.1,2.1,"+1",ha="center",va="center",fontweight="bold",color="#047857")
    ax.text(5,0.7,"anda para esquerda ou direita com 50%;\nrecompensa +1 só ao sair pela direita",ha="center",fontsize=9)
    ax=axs[1]
    for alg,al,c,ls in [("mc",0.02,RED,"-"),("mc",0.04,RED,"--"),("td",0.05,IND,"-"),("td",0.1,IND,"--"),("td",0.15,IND,":")]:
        E=np.mean([passeio(alg,al,100,s) for s in range(100)],axis=0); ax.plot(E,c=c,ls=ls,lw=2,label=("MC" if alg=="mc" else "TD(0)")+f", α = {al}")
    ax.set_xlabel("episódios"); ax.set_ylabel("erro quadrático médio (RMS)"); ax.set_title("TD aprende mais rápido que MC"); ax.legend(frameon=False,fontsize=8)
    save(fig,"f6_td_vs_mc")

def eps_greedy(Q,s,eps,rng,nA=4): return int(rng.integers(nA)) if rng.random()<eps else int(np.argmax(Q[s]+1e-9*rng.random(nA)))
def cliff_treinar(env,alg,n_ep=500,alpha=0.5,eps=0.1,semente=0):
    rng=np.random.default_rng(semente); Q=np.zeros((48,4)); rets=[]
    for ep in range(n_ep):
        s,_=env.reset(seed=int(rng.integers(1e9))); a=eps_greedy(Q,s,eps,rng); fim=False; tot=0
        while not fim:
            s2,r,te,tr,_=env.step(a); fim=te or tr; tot+=r; a2=eps_greedy(Q,s2,eps,rng)
            alvo=r+(Q[s2,a2] if alg=="sarsa" else Q[s2].max())*(not te); Q[s,a]+=alpha*(alvo-Q[s,a]); s,a=s2,a2
        rets.append(tot)
    return Q,np.array(rets)
def caminho(env,Q):
    s,_=env.reset(seed=0); p=[s]; fim=False
    while not fim and len(p)<60:
        s,r,te,tr,_=env.step(int(Q[s].argmax())); p.append(s); fim=te or tr
    return p
def f6_cliff():
    env=gym.make("CliffWalking-v1"); Qs,_=cliff_treinar(env,"sarsa",semente=1); Qq,_=cliff_treinar(env,"q",semente=1)
    Rs=np.mean([cliff_treinar(env,"sarsa",semente=k)[1] for k in range(30)],axis=0); Rq=np.mean([cliff_treinar(env,"q",semente=k)[1] for k in range(30)],axis=0)
    Rm=lambda x,k=10:np.convolve(x,np.ones(k)/k,mode="valid")
    fig=plt.figure(figsize=(9.6,3.9)); ax=fig.add_axes([0.0,0.08,0.52,0.82]); ax.set_xlim(0,12); ax.set_ylim(0,4); ax.axis("off"); ax.set_title("Caminhos gulosos aprendidos",fontsize=10)
    for s in range(48):
        i,j=divmod(s,12); y=3-i; col="#FEE2E2" if 37<=s<=46 else "#EEF2FF"
        ax.add_patch(Rectangle((j,y),0.95,0.95,fc=col,ec="white"))
    ax.text(6,0.45,"PRECIPÍCIO (−100)",ha="center",va="center",color=RED,fontsize=10,fontweight="bold"); ax.text(0.47,0.45,"início",ha="center",va="center",fontsize=8,fontweight="bold"); ax.text(11.47,0.45,"meta",ha="center",va="center",fontsize=8,fontweight="bold")
    for Q,c,lab,dy in [(Qs,GRN,"SARSA (caminho seguro)",0.12),(Qq,IND,"Q-learning (beira do abismo)",-0.12)]:
        p=caminho(env,Q); xy=np.array([(s%12+0.47,3-s//12+0.47+dy) for s in p]); ax.plot(xy[:,0],xy[:,1],c=c,lw=3,label=lab,marker="o",ms=3)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,0.02),frameon=False,fontsize=8.5,ncol=2)
    ax2=fig.add_axes([0.6,0.2,0.38,0.68]); ax2.plot(Rm(Rs),c=GRN,lw=2,label="SARSA"); ax2.plot(Rm(Rq),c=IND,lw=2,label="Q-learning"); ax2.set_ylim(-120,0)
    ax2.set_xlabel("episódios"); ax2.set_ylabel("retorno por episódio (durante o treino)"); ax2.set_title("Desempenho online (ε = 0,1)",fontsize=10); ax2.legend(frameon=False)
    fig.savefig(OUT+"f6_cliff.png",facecolor="white",bbox_inches="tight",pad_inches=0.12,dpi=200); plt.close(fig); print("fig f6_cliff")

def mdp_vies(duplo, n_ep=300, semente=0, alpha=0.1, eps=0.1, nB=8):
    rng=np.random.default_rng(semente); QA=np.zeros((2,nB)); QB=np.zeros((2,nB)); esq=[]
    def escolhe(s,n):
        q=(QA[s,:n]+QB[s,:n]) if duplo else QA[s,:n]
        return int(rng.integers(n)) if rng.random()<eps else int(np.argmax(q+1e-9*rng.random(n)))
    for ep in range(n_ep):
        a=escolhe(0,2); esq.append(a==0)
        if a==1:                                          # direita: fim, r = 0
            T=QA if (not duplo or rng.random()<0.5) else QB
            T[0,1]+=alpha*(0-T[0,1]); continue
        b=escolhe(1,nB); r=rng.normal(-0.1,1.0)           # esquerda -> B, depois uma ação em B
        if not duplo:
            QA[0,0]+=alpha*(QA[1].max()-QA[0,0]); QA[1,b]+=alpha*(r-QA[1,b])
        elif rng.random()<0.5:
            QA[0,0]+=alpha*(QB[1,QA[1].argmax()]-QA[0,0]); QA[1,b]+=alpha*(r-QA[1,b])
        else:
            QB[0,0]+=alpha*(QA[1,QB[1].argmax()]-QB[0,0]); QB[1,b]+=alpha*(r-QB[1,b])
    return np.array(esq)

def f6_vies():
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.6)); ax=axs[0]; ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off"); ax.set_title("Um MDP que engana o máximo")
    for x,y,t,c in [(1.5,3,"A",IND),(5,3,"B",AMB)]: dot(ax,x,y,s=1400,c=c); ax.text(x,y,t,ha="center",va="center",color="white",fontweight="bold",fontsize=14)
    arrow(ax,(2.2,3),(4.3,3),c=INK,lw=2); ax.text(3.2,3.4,"esquerda\nr = 0",ha="center",fontsize=8.5)
    arrow(ax,(1.5,2.3),(1.5,0.8),c=INK,lw=2); ax.text(2.0,1.4,"direita: fim, r = 0",fontsize=8.5)
    ax.text(6.5,3.3,"muitas ações,\ncada uma com\nr ~ N(−0,1; 1)\ne depois fim",fontsize=8.5,va="center")
    ax.text(5,0.3,"Em média, ir à esquerda é RUIM (−0,1), mas há máximos otimistas…",ha="center",fontsize=8.5,color=RED,fontweight="bold")
    ax=axs[1]
    for alg,c,lab in [("q",RED,"Q-learning"),("double",GRN,"Double Q-learning")]:
        L=np.mean([mdp_vies(alg=="double",300,s) for s in range(500)],axis=0); ax.plot(np.convolve(L,np.ones(5)/5,mode="valid")*100,c=c,lw=2,label=lab)
    ax.axhline(5,c=GRY,ls="--"); ax.text(300,1.5,"mínimo possível (ε/2 = 5%)",ha="right",fontsize=8.5,color=GRY)
    ax.set_xlabel("episódios"); ax.set_ylabel("% de escolhas \"esquerda\" em A"); ax.set_title("Viés de maximização"); ax.legend(frameon=False)
    save(fig,"f6_vies")

def f6_taxi():
    try: env=gym.make("Taxi-v4")
    except Exception: env=gym.make("Taxi-v3")
    nS,nA=env.observation_space.n,env.action_space.n; curvas=[]
    for sem in range(5):
        rng=np.random.default_rng(sem); Q=np.zeros((nS,nA)); rets=[]
        for ep in range(4000):
            eps=max(0.05,1-ep/2000); s,_=env.reset(seed=int(rng.integers(1e9))); fim=False; tot=0
            while not fim:
                a=int(rng.integers(nA)) if rng.random()<eps else int(Q[s].argmax()); s2,r,te,tr,_=env.step(a); fim=te or tr; tot+=r
                Q[s,a]+=0.1*(r+0.99*Q[s2].max()*(not te)-Q[s,a]); s=s2
            rets.append(tot)
        curvas.append(np.convolve(rets,np.ones(50)/50,mode="valid"))
    C=np.array(curvas); fig,ax=plt.subplots(figsize=(8,3.5)); x=np.arange(C.shape[1])
    ax.fill_between(x,C.min(0),C.max(0),color=IND,alpha=0.18,label="mín–máx de 5 sementes"); ax.plot(x,C.mean(0),c=IND,lw=2.2,label="média"); ax.set_ylim(-400,30)
    ax.set_xlabel("episódios"); ax.set_ylabel("retorno por episódio"); ax.set_title("Q-learning aprendendo a dirigir o táxi"); ax.legend(frameon=False,loc="lower right")
    save(fig,"f6_taxi")

def f6_backups():
    fig,axs=plt.subplots(1,3,figsize=(10,3.4))
    tit=["SARSA\nQ(S,A) ← R + γ Q(S′,A′)","Q-learning\nQ(S,A) ← R + γ max Q(S′,·)","Expected SARSA\nQ(S,A) ← R + γ Σ π·Q(S′,·)"]
    for k,ax in enumerate(axs):
        ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off"); ax.set_title(tit[k],fontsize=9.5,color=[GRN,IND,AMB][k])
        ax.scatter([5],[9],s=120,c=INK,zorder=3); ax.plot([5,5],[8.7,7.3],c=INK,lw=1.4); ax.scatter([5],[6.8],s=700,facecolors="white",edgecolors=GRN,linewidths=2,zorder=3); ax.text(5,6.8,"S′",ha="center",va="center",fontsize=9)
        for x in [2.5,5,7.5]: ax.plot([5,x],[6.3,3.3],c=GRY,lw=1.2); ax.scatter([x],[3.0],s=100,c=INK,zorder=3)
        if k==0: ax.scatter([5],[3.0],s=350,facecolors="none",edgecolors=GRN,linewidths=2.5,zorder=4); ax.text(5,0.8,"a única ação A′\nque o agente escolheu",ha="center",fontsize=8.5)
        if k==1: ax.scatter([2.5],[3.0],s=350,facecolors="none",edgecolors=IND,linewidths=2.5,zorder=4); ax.text(5,0.8,"a melhor ação (máximo),\nmesmo que não seja escolhida",ha="center",fontsize=8.5)
        if k==2:
            for x in [2.5,5,7.5]: ax.scatter([x],[3.0],s=350,facecolors="none",edgecolors=AMB,linewidths=2.5,zorder=4)
            ax.text(5,0.8,"média sobre todas as ações,\nponderada pela política",ha="center",fontsize=8.5)
    save(fig,"f6_backups")

def f6_nstep():
    fig,ax=canvas(10,3.4); labels=["TD(0)\n1 passo","2 passos","3 passos","n passos","Monte Carlo\n(até o fim)"]
    for k,(x,l) in enumerate(zip(np.linspace(10,90,5),labels)):
        ax.scatter([x],[78],s=260,c=IND,zorder=3,edgecolors="white")
        n=k+1 if k<4 else 5
        for j in range(n):
            ax.plot([x,x],[75-j*9.5,66-j*9.5],c=INK,lw=1.4); ax.scatter([x],[63-j*9.5],s=70,c=INK if j<n-1 or k<4 else GRN,zorder=3)
        if k==4: ax.scatter([x],[63-4*9.5],marker="s",s=160,c=GRN,zorder=3)
        ax.text(x,5,l,ha="center",fontsize=9,fontweight="bold",color=[IND,CYA,GRN,AMB,RED][k])
    arrow(ax,(14,95),(86,95),c=GRY,lw=1.5,style="<|-|>"); ax.text(50,98.5,"menos variância, mais viés   ⟷   mais variância, menos viés",ha="center",fontsize=9.5,color=GRY,fontweight="bold")
    save(fig,"f6_nstep")

def f6_cartpole():
    from sklearn.preprocessing import KBinsDiscretizer
    env=gym.make("CartPole-v1"); rng=np.random.default_rng(0); obs=[]; o,_=env.reset(seed=0)
    for _ in range(20000):
        o,r,te,tr,_=env.step(int(rng.integers(2))); obs.append(o)
        if te or tr: o,_=env.reset()
    disc=KBinsDiscretizer(n_bins=[3,3,8,10],encode="ordinal",strategy="quantile").fit(np.array(obs)); bordas=[e[1:-1] for e in disc.bin_edges_]; dims=[len(b)+1 for b in bordas]
    est=lambda o:int(np.ravel_multi_index(tuple(int(np.digitize(o[i],bordas[i])) for i in range(4)),dims)); C=[]
    for sem in range(3):
        rng=np.random.default_rng(sem); Q=np.zeros((int(np.prod(dims)),2)); rets=[]
        for ep in range(4000):
            eps=max(0.01,0.3*(1-ep/3000)); al=max(0.05,0.5*(1-ep/3000)); o,_=env.reset(seed=int(rng.integers(1e9))); s=est(o); fim=False; tot=0
            while not fim:
                a=int(rng.integers(2)) if rng.random()<eps else int(Q[s].argmax()); o2,r,te,tr,_=env.step(a); s2=est(o2); fim=te or tr; tot+=r
                Q[s,a]+=al*(r+0.99*Q[s2].max()*(not te)-Q[s,a]); s=s2
            rets.append(tot)
        C.append(np.convolve(rets,np.ones(100)/100,mode="valid"))
    C=np.array(C); fig,ax=plt.subplots(figsize=(8,3.4)); x=np.arange(C.shape[1]); ax.fill_between(x,C.min(0),C.max(0),color=GRN,alpha=0.2,label="mín–máx (3 sementes)"); ax.plot(x,C.mean(0),c="#047857",lw=2.2,label="média")
    ax.set_xlabel("episódios"); ax.set_ylabel("retorno por episódio"); ax.set_title("Q-learning tabular no CartPole (estado discretizado)"); ax.legend(frameon=False,loc="upper left"); save(fig,"f6_cartpole")
if __name__=="__main__":
    import sys
    fs={f.__name__[3:]:f for f in [f6_dirigindo,f6_td_erro,f6_td_vs_mc,f6_cliff,f6_vies,f6_taxi,f6_backups,f6_nstep,f6_cartpole]}
    for k in (sys.argv[1:] or fs): fs[k]()
