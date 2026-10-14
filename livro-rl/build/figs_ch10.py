from fc import *
from figs_cover import star
def f10_ppo_clip():
    eps=0.2; r=np.linspace(0.4,1.6,300); fig,axs=plt.subplots(1,2,figsize=(9.6,3.5),sharey=False)
    for ax,A,t in [(axs[0],1.0,"Vantagem positiva (A > 0): a ação foi boa"),(axs[1],-1.0,"Vantagem negativa (A < 0): a ação foi ruim")]:
        sub=r*A; clip=np.clip(r,1-eps,1+eps)*A; L=np.minimum(sub,clip)
        ax.plot(r,sub,c=GRY,lw=1.4,ls="--",label="r · A (sem corte)"); ax.plot(r,L,c=IND,lw=3,label="objetivo do PPO: min(r·A, clip(r)·A)")
        ax.axvline(1,c=MID,lw=1); ax.axvspan(1-eps,1+eps,color=GRN,alpha=0.08); ax.set_xlabel("razão r = π_nova(a|s) / π_antiga(a|s)"); ax.set_title(t,fontsize=9.5); ax.legend(frameon=False,fontsize=7.5,loc="upper left" if A>0 else "lower left")
    axs[0].set_ylabel("objetivo (por amostra)"); axs[0].text(1.0,0.62,"zona de confiança",ha="center",fontsize=8,color="#047857"); fig=axs[0].figure
    save(fig,"f10_ppo_clip")
def f10_arquiteturas():
    fig,ax=canvas(10,4.4)
    cols=[("DDPG",IND,2),("TD3",GRN,35),("SAC",AMB,68)]
    for nome,c,x in cols:
        rbox(ax,x,5,30,90,"",fc="white",ec=c,lw=2.2,r=3); rbox(ax,x,84,30,11,nome,fc=c,tc="white",bold=True,fs=11.5,r=3)
    # DDPG
    for y,t in [(70,"ator determinístico μ(s)"),(52,"1 crítico Q(s, a)"),(36,"ruído na ação (exploração)"),(20,"alvos suavizados\n(τ pequeno)")]: rbox(ax,4,y-5,26,10,t,fc="#EEF2FF",ec=IND,lw=1,r=1.5,fs=8)
    for y,t in [(74,"ator determinístico μ(s)"),(60,"2 críticos: usa o MENOR Q"),(46,"atualiza o ator\nmenos vezes (atraso)"),(30,"suaviza a ação-alvo\ncom ruído"),(14,"ruído na ação")]: rbox(ax,37,y-5.5,26,10,t,fc="#ECFDF5",ec=GRN,lw=1,r=1.5,fs=7.8)
    for y,t in [(74,"ator ESTOCÁSTICO\n(gaussiana)"),(60,"2 críticos: usa o MENOR Q"),(46,"bônus de ENTROPIA\nno objetivo"),(30,"temperatura α\najustada sozinha"),(14,"exploração embutida")]: rbox(ax,70,y-5.5,26,10,t,fc="#FFFBEB",ec=AMB,lw=1,r=1.5,fs=7.8)
    ax.set_ylim(0,106); ax.text(50,105,"Três algoritmos off-policy para ações contínuas",ha="center",fontsize=11,fontweight="bold",va="top")
    save(fig,"f10_arquiteturas")
def f10_mapa():
    fig,ax=canvas(10,4.2)
    rbox(ax,36,80,28,14,"Qual é o seu problema?",fc="#EEF2FF",ec=IND,lw=2,r=2.5,fs=10,bold=True)
    rbox(ax,3,50,28,14,"Ações DISCRETAS",fc="#ECFDF5",ec=GRN,lw=2,r=2.5,fs=9.5,bold=True); rbox(ax,69,50,28,14,"Ações CONTÍNUAS",fc="#FFFBEB",ec=AMB,lw=2,r=2.5,fs=9.5,bold=True)
    arrow(ax,(44,80),(20,64.5),c=INK); arrow(ax,(56,80),(80,64.5),c=INK)
    for x,y,t,c in [(1,3,"PPO\nrobusto e\nsimples",IND),(18,3,"DQN /\nRainbow\nreusa dados",GRN),(50,3,"SAC\neficiente em\ndados",AMB),(66,3,"TD3\ndeterminístico\ne estável",GRN),(82,3,"PPO\nsimulação\nbarata",IND)]:
        rbox(ax,x,3,15.5,26,t,fc="white",ec=c,lw=2,r=2,fs=8)
    arrow(ax,(14,50),(8.5,29.5),c=GRY); arrow(ax,(20,50),(25.5,29.5),c=GRY); arrow(ax,(76,50),(58,29.5),c=GRY); arrow(ax,(83,50),(74,29.5),c=GRY); arrow(ax,(90,50),(90,29.5),c=GRY)
    ax.text(50,98,"Um guia rápido para escolher o algoritmo",ha="center",fontsize=11.5,fontweight="bold")
    save(fig,"f10_mapa")
def f10_curvas():
    import glob
    fig,ax=plt.subplots(figsize=(8.6,3.8)); cfg=[("SAC",AMB),("TD3",GRN),("PPO",IND)]
    for nome,c in cfg:
        R=[np.load(p) for p in sorted(glob.glob(f"/tmp/w/pend_{nome}_*.npy"))]
        if not R: continue
        n=min(len(r) for r in R); M=np.array([np.convolve(r[:n],np.ones(5)/5,mode="valid") for r in R]); x=(np.arange(M.shape[1])+5)*200
        ax.fill_between(x,M.min(0),M.max(0),color=c,alpha=0.15); ax.plot(x,M.mean(0),c=c,lw=2.4,label=f"{nome} ({len(R)} sementes)")
    ax.set_xscale("log"); ax.set_xlabel("passos de interação (escala log)"); ax.set_ylabel("retorno por episódio (média móvel de 5)"); ax.set_title("Pendulum: eficiência de dados de PPO, TD3 e SAC"); ax.legend(frameon=False,loc="lower right"); save(fig,"f10_curvas")
if __name__=="__main__":
    import sys
    fs={f.__name__[4:]:f for f in [f10_ppo_clip,f10_arquiteturas,f10_mapa,f10_curvas]}
    for k in (sys.argv[1:] or fs): fs[k]()
