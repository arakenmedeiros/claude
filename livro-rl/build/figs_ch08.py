from fc import *
from figs_cover import star
from matplotlib.colors import LinearSegmentedColormap
def f8_neuronio():
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={"width_ratios":[1.15,1]})
    ax=axs[0]; ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off"); ax.set_title("Um neurônio artificial",fontsize=10.5)
    for i,(y,l) in enumerate([(5,"x₁"),(3,"x₂"),(1,"x₃")]):
        dot(ax,1,y,s=500,c=CYA); ax.text(1,y,l,ha="center",va="center",color="white",fontweight="bold"); ax.annotate("",xy=(4.6,3),xytext=(1.4,y),arrowprops=dict(arrowstyle="-|>",color=GRY,lw=1.4)); ax.text(2.7,(y+3)/2+0.15+0.0*i,["w₁","w₂","w₃"][i],fontsize=9.5,fontweight="bold",color=IND)
    dot(ax,5.2,3,s=2400,c=IND); ax.text(5.2,3,"Σ + b",ha="center",va="center",color="white",fontweight="bold",fontsize=11)
    ax.annotate("",xy=(7.4,3),xytext=(6.1,3),arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.8)); rbox(ax,7.4,2.35,1.4,1.3,"f(·)",fc="#FEF3C7",ec=AMB,lw=2,r=0.3,fs=11,bold=True); ax.annotate("",xy=(9.9,3),xytext=(8.8,3),arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.8)); ax.text(9.4,3.45,"saída",fontsize=9,ha="center")
    ax.text(5,0.2,"saída = f( w₁x₁ + w₂x₂ + w₃x₃ + b )",ha="center",fontsize=10,fontweight="bold",color=IND)
    ax=axs[1]; x=np.linspace(-3,3,200); ax.plot(x,np.maximum(0,x),c=GRN,lw=2.5,label="ReLU: max(0, x)"); ax.plot(x,np.tanh(x),c=IND,lw=2.2,label="tanh"); ax.plot(x,1/(1+np.exp(-x)),c=AMB,lw=2.2,label="sigmoide")
    ax.axhline(0,c=MID,lw=0.8); ax.axvline(0,c=MID,lw=0.8); ax.set_ylim(-1.3,3); ax.set_title("Funções de ativação f(·)",fontsize=10.5); ax.legend(frameon=False,fontsize=8.5,loc="upper left"); ax.set_xlabel("entrada x")
    save(fig,"f8_neuronio")
def f8_rede():
    fig,ax=canvas(9.5,4.2); camadas=[(10,[85,62,38,15],"Entrada\n(estado)",CYA),(36,[88,70,52,34,16],"Camada\noculta 1\n(ReLU)",IND),(62,[88,70,52,34,16],"Camada\noculta 2\n(ReLU)",IND),(88,[62,38],"Saída\n(valores Q)",AMB)]
    for (x,ys,t,c),(x2,ys2,_,_) in zip(camadas[:-1],camadas[1:]):
        for y in ys:
            for y2 in ys2: ax.plot([x,x2],[y,y2],c=MID,lw=0.7,zorder=1)
    for x,ys,t,c in camadas:
        for y in ys: dot(ax,x,y,s=420,c=c,lw=1.2)
        ax.text(x,3,t,ha="center",va="top",fontsize=8.5,fontweight="bold",color=c if c!=AMB else "#B45309")
    for y,l in zip([85,62,38,15],["posição\ncarrinho","velocidade\ncarrinho","ângulo\nposte","vel. angular\nposte"]): ax.text(5,y,l,ha="right",va="center",fontsize=7.5)
    for y,l in zip([62,38],["Q(s, ←)","Q(s, →)"]): ax.text(93,y,l,ha="left",va="center",fontsize=9.5,fontweight="bold")
    ax.set_xlim(-2,102); ax.set_ylim(-14,100); ax.text(50,98,"Rede Q para o CartPole: o estado entra, um valor Q para cada ação sai",ha="center",fontsize=11,fontweight="bold")
    save(fig,"f8_rede")
def f8_dqn():
    fig,ax=canvas(10,5.0)
    rbox(ax,2,70,16,22,"Ambiente",fc="#ECFDF5",ec=GRN,lw=2,r=2.5,fs=10.5,bold=True)
    rbox(ax,28,70,18,22,"Agente\nε-guloso",fc="#EEF2FF",ec=IND,lw=2,r=2.5,fs=10.5,bold=True)
    rbox(ax,56,64,22,32,"",fc="#FFFBEB",ec=AMB,lw=2,r=2.5); ax.text(67,97.5,"Memória de experiências",ha="center",fontsize=9.5,fontweight="bold",color="#B45309"); 
    for k in range(5): rbox(ax,58,66+k*4.2,18,3.2,"(s, a, r, s′, fim)",fc="white",ec=AMB,lw=0.8,r=0.8,fs=6.5)
    arrow(ax,(18,81),(28,81),c=INK,lw=1.8); ax.text(23,86,"s, r",fontsize=8.5,ha="center"); arrow(ax,(28,75),(18,75),c=INK,lw=1.8); ax.text(23,69.5,"ação a",fontsize=8.5,ha="center")
    arrow(ax,(46,82),(56,82),c=INK,lw=1.8); ax.text(51,86,"guarda",fontsize=8.5,ha="center")
    rbox(ax,56,24,22,26,"Lote aleatório\n(minibatch)",fc="#F5F3FF",ec=VIO,lw=2,r=2.5,fs=10,bold=True); arrow(ax,(67,64),(67,50.5),c=VIO,lw=2); ax.text(75,57,"sorteia",fontsize=8.5,color=VIO)
    rbox(ax,28,34,18,16,"Rede Q\n(θ) treinada",fc="#EEF2FF",ec=IND,lw=2,r=2.5,fs=9.5,bold=True); rbox(ax,28,6,18,16,"Rede-alvo\n(θ⁻) congelada",fc="#FEF2F2",ec=RED,lw=2,r=2.5,fs=9.5,bold=True)
    arrow(ax,(56,37),(46,43),c=INK,lw=1.6); arrow(ax,(56,30),(46,17),c=INK,lw=1.6); ax.text(56,12,"calcula o alvo\ny = r + γ max Q(s′; θ⁻)",fontsize=8.5,ha="left",color=RED,fontweight="bold")
    arrow(ax,(37,34),(37,22.5),c=GRY,lw=1.4,ls="--"); ax.text(39.5,27.5,"cópia a\ncada C passos",fontsize=8,color=GRY,va="center")
    arrow(ax,(37,50),(37,70),c=IND,lw=1.8); ax.text(39,60,"decide",fontsize=8.5,color=IND)
    ax.text(10,40,"perda:\n(y − Q(s,a;θ))²\n→ gradiente → atualiza θ",fontsize=9,color=IND,fontweight="bold",va="center",ha="center")
    save(fig,"f8_dqn")
def f8_replay():
    fig,ax=canvas(10,3.2); rng=np.random.default_rng(1)
    ax.text(5,92,"Fluxo no tempo\n(correlacionado)",fontsize=9.5,fontweight="bold",va="top"); 
    for i in range(10): dot(ax,14+i*5.2,66,s=350,c=IND,alpha=0.35+0.06*i)
    arrow(ax,(66,66),(76,66),c=INK,lw=2); ax.text(71,76,"guarda",ha="center",fontsize=8.5)
    rbox(ax,77,30,21,56,"",fc="#FFFBEB",ec=AMB,lw=2,r=2.5); ax.text(87.5,92,"Memória",ha="center",fontsize=10,fontweight="bold",color="#B45309")
    for k in range(8): rbox(ax,79,33+k*6.2,17,4.6,"",fc=["#C7D2FE","#A5F3FC","#FDE68A","#BBF7D0","#FBCFE8"][k%5],ec="white",lw=0.5,r=0.6)
    for k in rng.choice(8,3,replace=False): arrow(ax,(77,35+k*6.2),(60,15),c=VIO,lw=1.3,rad=0.1)
    rbox(ax,22,5,38,20,"Lote embaralhado\n(quase independente)",fc="#F5F3FF",ec=VIO,lw=2,r=2.5,fs=10,bold=True)
    ax.text(14,40,"Sem memória: o agente aprende só\ncom o que acabou de ver\ne esquece o resto.",fontsize=8.8,color=RED,fontweight="bold",va="center")
    save(fig,"f8_replay")
def f8_cnn():
    fig,ax=canvas(10,3.8); cam=[(4,14,34,"4 imagens\n84×84",CYA),(24,12,28,"Conv 32\n8×8, passo 4",IND),(40,10,22,"Conv 64\n4×4, passo 2",IND),(56,8,18,"Conv 64\n3×3, passo 1",IND),(72,5,34,"Totalmente\nconectada\n512",VIO),(90,4,16,"Q por ação",AMB)]
    for x,w,h,t,c in cam:
        ax.add_patch(Rectangle((x,50-h/2),w*0.55,h,fc=c,ec="white",alpha=0.85)); ax.text(x+w*0.28,50-h/2-5,t,ha="center",va="top",fontsize=8,fontweight="bold",color=c if c!=AMB else "#B45309")
    for (x,w,h,_,_),(x2,_,_,_,_) in zip(cam[:-1],cam[1:]): arrow(ax,(x+w*0.55+0.5,50),(x2-0.5,50),c=GRY,lw=1.4,ms=9)
    ax.text(50,92,"A rede convolucional do DQN para Atari (MNIH et al., 2015)",ha="center",fontsize=11,fontweight="bold")
    save(fig,"f8_cnn")
def f8_extensoes():
    fig,ax=canvas(10,3.8); itens=[("DQN","base",GRY),("Double DQN","reduz o viés de maximização",RED),("Repetição\npriorizada","mais atenção a transições surpreendentes",AMB),("Dueling","separa valor e vantagem",GRN),("Multi-step","alvos de n passos",CYA),("Distribucional","aprende a distribuição do retorno",IND),("Noisy Nets","exploração por ruído na rede",VIO)]
    ax.text(50,95,"Rainbow = DQN + seis melhorias combinadas (HESSEL et al., 2018)",ha="center",fontsize=11,fontweight="bold")
    for k,(t,d,c) in enumerate(itens):
        y=82-k*11.2; rbox(ax,3,y-4.5,26,9.5,t.replace("\n"," "),fc=c,tc="white",bold=True,fs=9.5,r=1.6); ax.text(32,y,d,va="center",fontsize=9.5)
    star(ax,88,45,7,c=AMB); ax.text(88,30,"Rainbow",ha="center",fontsize=12,fontweight="bold",color="#B45309")
    for k in range(7): arrow(ax,(70,82-k*11.2),(81,50),c=itens[k][2],lw=1.2,rad=0.0,ms=8)
    save(fig,"f8_extensoes")
def f8_dueling():
    fig,ax=canvas(9,3.6); rbox(ax,3,36,14,28,"Estado\n(camadas\ncomuns)",fc="#EEF2FF",ec=IND,lw=2,r=2.5,fs=9,bold=True)
    rbox(ax,31,60,26,22,"V(s)\n“quão bom é\nestar aqui”",fc="#ECFDF5",ec=GRN,lw=2,r=2.5,fs=9,bold=True); rbox(ax,31,18,26,22,"A(s, a)\n“quanto cada ação\nsupera a média”",fc="#FFFBEB",ec=AMB,lw=2,r=2.5,fs=9,bold=True)
    arrow(ax,(17,54),(31,68),c=INK,lw=1.6); arrow(ax,(17,46),(31,32),c=INK,lw=1.6); rbox(ax,70,38,24,24,"Q(s, a) =\nV(s) + A(s, a)\n− média de A",fc="#F5F3FF",ec=VIO,lw=2,r=2.5,fs=9.5,bold=True)
    arrow(ax,(57,70),(70,56),c=INK,lw=1.6); arrow(ax,(57,30),(70,44),c=INK,lw=1.6); ax.text(50,94,"Arquitetura dueling (WANG et al., 2016)",ha="center",fontsize=11,fontweight="bold")
    save(fig,"f8_dueling")
def f8_ablacao():
    import glob
    fig,ax=plt.subplots(figsize=(8.6,3.8)); cfg=[("completo","DQN completo",GRN),("duplo","Double DQN",IND),("sem_alvo","sem rede-alvo",AMB),("sem_memoria","sem memória",RED),("nenhum","sem memória e sem rede-alvo",GRY)]
    for nome,lab,c in cfg:
        R=[np.load(f) for f in sorted(glob.glob(f"/tmp/w/abl_{nome}_*.npy"))]
        if not R: continue
        n=min(len(r) for r in R); M=np.array([np.convolve(r[:n],np.ones(20)/20,mode="valid") for r in R]); x=np.arange(M.shape[1])
        ax.fill_between(x,M.min(0),M.max(0),color=c,alpha=0.15); ax.plot(x,M.mean(0),c=c,lw=2.2,label=f"{lab} ({len(R)} sementes)")
    ax.set_xlabel("episódio"); ax.set_ylabel("retorno (média móvel de 20)"); ax.set_title("Ablação do DQN no CartPole"); ax.legend(frameon=False,fontsize=8.5,loc="upper left"); save(fig,"f8_ablacao")
if __name__=="__main__":
    import sys
    fs={f.__name__[3:]:f for f in [f8_neuronio,f8_rede,f8_dqn,f8_replay,f8_cnn,f8_extensoes,f8_dueling,f8_ablacao]}
    for k in (sys.argv[1:] or fs): fs[k]()
