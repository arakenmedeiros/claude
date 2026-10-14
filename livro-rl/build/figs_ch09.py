from fc import *
from figs_cover import star
from matplotlib.colors import LinearSegmentedColormap
def f9_politicas():
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.5))
    ax=axs[0]; z=np.array([2.0,0.5,-1.0]); p=np.exp(z)/np.exp(z).sum(); b=ax.bar(["esquerda","nada","direita"],p,color=[IND,GRY,AMB]); 
    for r,v in zip(b,p): ax.text(r.get_x()+r.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontweight="bold",fontsize=9)
    ax.set_ylim(0,1); ax.set_ylabel("π(a | s)"); ax.set_title("Ações discretas: softmax\nπ(a|s) = e^h(s,a) / Σ e^h(s,a′)",fontsize=9.5)
    ax=axs[1]; x=np.linspace(-3,3,300); ax.plot(x,np.exp(-0.5*((x-0.6)/0.5)**2)/(0.5*np.sqrt(2*np.pi)),c=IND,lw=2.5,label="μ = 0,6; σ = 0,5"); ax.plot(x,np.exp(-0.5*((x-0.6)/1.2)**2)/(1.2*np.sqrt(2*np.pi)),c=AMB,lw=2.5,label="μ = 0,6; σ = 1,2 (explora mais)")
    ax.fill_between(x,np.exp(-0.5*((x-0.6)/0.5)**2)/(0.5*np.sqrt(2*np.pi)),color=IND,alpha=0.12); ax.set_xlabel("ação contínua (ex.: força aplicada)"); ax.set_ylabel("densidade"); ax.set_title("Ações contínuas: gaussiana\na rede aprende μ(s) e σ(s)",fontsize=9.5); ax.legend(frameon=False,fontsize=8.5)
    save(fig,"f9_politicas")
def f9_ideia():
    fig,axs=plt.subplots(1,3,figsize=(10,3.2),sharey=True); nomes=["esq.","nada","dir."]
    antes=np.array([0.34,0.33,0.33]); depois=np.array([0.20,0.28,0.52]); 
    axs[0].bar(nomes,antes,color=GRY); axs[0].set_title("1. A política joga\num episódio"); axs[0].set_ylim(0,0.8)
    axs[1].bar(nomes,[0.1,0.2,0.7],color=[RED,GRY,GRN]); axs[1].set_title("2. “direita” rendeu\nbom retorno (G alto)",color="#047857")
    axs[2].bar(nomes,depois,color=[IND,IND,GRN]); axs[2].set_title("3. A política passa a\nescolher “direita” mais")
    axs[1].set_ylabel(""); axs[0].set_ylabel("probabilidade / crédito")
    for ax in axs: ax.tick_params(labelsize=8.5)
    fig.suptitle("A ideia do gradiente de política: reforçar as ações que deram bons resultados",fontsize=10.5,fontweight="bold",y=1.04)
    save(fig,"f9_ideia")
def f9_baseline():
    rng=np.random.default_rng(0); p=np.array([0.3,0.4,0.3]); mu=np.array([9.0,10.0,11.0]); N=3000
    def grad(base):
        g=[]
        for _ in range(N):
            a=rng.choice(3,p=p); G=rng.normal(mu[a],1.0); e=-p.copy(); e[a]+=1   # ∇ ln π (softmax, logits)
            g.append(e*(G-base))
        return np.array(g)
    g0=grad(0.0); g1=grad(10.0)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.5),sharex=True,sharey=True)
    for ax,g,t,c in [(axs[0],g0,"Sem linha de base (b = 0)",RED),(axs[1],g1,"Com linha de base (b = 10)",GRN)]:
        ax.scatter(g[:,0],g[:,2],s=5,c=c,alpha=0.25); ax.scatter(*g[:,[0,2]].mean(0),s=120,c=INK,marker="X",zorder=5); ax.set_title(f"{t}\ndesvio-padrão ≈ {g[:,2].std():.1f}",fontsize=9.5); ax.set_xlabel("gradiente estimado, componente 1"); 
    axs[0].set_ylabel("gradiente estimado, componente 3"); fig.suptitle("Mesma média (X), variância muito menor com a linha de base",fontsize=10.5,fontweight="bold",y=1.04)
    save(fig,"f9_baseline")
def f9_ator_critico():
    fig,ax=canvas(10,3.8)
    rbox(ax,2,38,16,26,"Ambiente",fc="#ECFDF5",ec=GRN,lw=2,r=2.5,fs=10.5,bold=True)
    rbox(ax,34,60,24,30,"ATOR\nπ(a | s; θ)\n“o que fazer”",fc="#EEF2FF",ec=IND,lw=2,r=2.5,fs=10,bold=True)
    rbox(ax,34,8,24,30,"CRÍTICO\nV(s; w)\n“quão bom é”",fc="#FFFBEB",ec=AMB,lw=2,r=2.5,fs=10,bold=True)
    arrow(ax,(34,78),(18,58),c=IND,lw=2); ax.text(24,75,"ação a",fontsize=9,color=IND,fontweight="bold")
    arrow(ax,(18,46),(34,24),c=INK,lw=1.6); ax.text(16,30,"s, r, s′",fontsize=9)
    rbox(ax,70,32,26,36,"erro TD\nδ = r + γV(s′) − V(s)\n“melhor ou pior\nque o esperado?”",fc="#FEF2F2",ec=RED,lw=2,r=2.5,fs=9,bold=True)
    arrow(ax,(58,22),(70,40),c=AMB,lw=2); arrow(ax,(70,60),(58,76),c=RED,lw=2); ax.text(60,83,"reforça ou\nenfraquece a ação",fontsize=8.5,color=RED,ha="left")
    arrow(ax,(83,32),(58,18),c=RED,lw=1.5,rad=-0.1); ax.text(76,16,"corrige o crítico",fontsize=8.5,color=RED)
    ax.text(50,97,"Ator-crítico: o crítico avalia, o ator melhora",ha="center",fontsize=11.5,fontweight="bold")
    save(fig,"f9_ator_critico")
def f9_regiao():
    fig,ax=plt.subplots(figsize=(6.5,4.2)); a,b=np.meshgrid(np.linspace(-3,3,200),np.linspace(-3,3,200)); J=np.exp(-((a-1.8)**2+(b-1.4)**2)/3)+0.6*np.exp(-((a+1.5)**2+(b+1.0)**2)/1.5)
    ax.contourf(a,b,J,levels=14,cmap=LinearSegmentedColormap.from_list("c",["#F5F3FF","#C4B5FD","#7C3AED","#FDE68A"])); 
    th=np.array([-0.5,-0.2]); ax.scatter(*th,s=120,c=INK,zorder=5); ax.text(th[0]-0.2,th[1]-0.55,"política atual",fontsize=9,fontweight="bold")
    ax.add_patch(plt.Circle(th,1.0,fc="none",ec=GRN,lw=2.5,ls="--")); ax.text(th[0]+0.1,th[1]+1.15,"região de confiança\n(KL pequena)",color="#047857",fontsize=9,fontweight="bold",ha="center")
    ax.annotate("",xy=(0.35,0.5),xytext=th,arrowprops=dict(arrowstyle="-|>",color=GRN,lw=3)); ax.annotate("",xy=(2.4,2.4),xytext=th,arrowprops=dict(arrowstyle="-|>",color=RED,lw=2,ls=":"))
    ax.text(0.2,2.75,"passo grande demais:\npode piorar a política",color=RED,fontsize=8.5,ha="center",fontweight="bold")
    ax.set_xlabel("parâmetros da política (θ₁)"); ax.set_ylabel("parâmetros da política (θ₂)"); ax.set_title("TRPO/PPO: passos seguros na direção que melhora o retorno",fontsize=10)
    save(fig,"f9_regiao")
def f9_curvas():
    import glob
    fig,ax=plt.subplots(figsize=(8.6,3.8))
    for nome,lab,c in [("reinforce","REINFORCE",RED),("baseline","REINFORCE + linha de base",IND),("a2c","Ator-crítico (20 passos)",GRN)]:
        R=[np.load(p) for p in sorted(glob.glob(f"/tmp/w/pg_{nome}_*.npy"))]
        if not R: continue
        n=min(len(r) for r in R); M=np.array([np.convolve(r[:n],np.ones(30)/30,mode="valid") for r in R]); x=np.arange(M.shape[1])+30
        ax.fill_between(x,M.min(0),M.max(0),color=c,alpha=0.15); ax.plot(x,M.mean(0),c=c,lw=2.3,label=f"{lab} ({len(R)} sementes)")
    ax.set_xlabel("episódio"); ax.set_ylabel("retorno (média móvel de 30)"); ax.set_title("Gradientes de política no CartPole"); ax.legend(frameon=False,fontsize=8.5,loc="center right"); save(fig,"f9_curvas")
if __name__=="__main__":
    import sys
    fs={f.__name__[3:]:f for f in [f9_politicas,f9_ideia,f9_baseline,f9_ator_critico,f9_regiao,f9_curvas]}
    for k in (sys.argv[1:] or fs): fs[k]()
