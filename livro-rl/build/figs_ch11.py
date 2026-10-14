from fc import *
from figs_cover import star
from mb_lib import *
def f11_ciclo():
    fig,ax=canvas(9.5,4.2)
    rbox(ax,3,60,22,30,"Mundo real",fc="#ECFDF5",ec=GRN,lw=2,r=2.5,fs=11,bold=True); rbox(ax,39,60,22,30,"Modelo\naprendido",fc="#FFFBEB",ec=AMB,lw=2,r=2.5,fs=11,bold=True); rbox(ax,75,60,22,30,"Política /\nvalor",fc="#EEF2FF",ec=IND,lw=2,r=2.5,fs=11,bold=True)
    arrow(ax,(25,80),(39,80),c=INK,lw=2); ax.text(32,85,"aprende o\nmodelo",fontsize=8.5,ha="center"); arrow(ax,(61,80),(75,80),c=AMB,lw=2); ax.text(68,85,"planeja\n(simula)",fontsize=8.5,ha="center",color="#B45309")
    arrow(ax,(75,68),(25,68),c=IND,lw=2,rad=-0.28); ax.text(50,36,"a política age no mundo real  →  nova experiência real",fontsize=9.5,ha="center",color=IND,fontweight="bold")
    ax.text(50,97,"A arquitetura Dyna: aprender, planejar e agir juntos",ha="center",fontsize=11.5,fontweight="bold"); ax.text(50,8,"cada passo real alimenta o modelo; o modelo gera muitos passos imaginados",ha="center",fontsize=9.5,color=GRY,style="italic")
    save(fig,"f11_ciclo")
def f11_dyna():
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.6)); ax=axs[0]; ax.set_xlim(0,9); ax.set_ylim(0,6); ax.axis("off"); ax.set_title("O labirinto do Dyna",fontsize=10)
    env=Labirinto()
    for i in range(6):
        for j in range(9):
            c="#374151" if (i,j) in env.paredes else "#EEF2FF"; ax.add_patch(Rectangle((j,5-i),0.95,0.95,fc=c,ec="white"))
    ax.text(0.47,5-2+0.47,"S",ha="center",va="center",fontweight="bold",color=IND); ax.text(8.47,5-0+0.47,"★",ha="center",va="center",fontsize=14,color="#B45309")
    ax=axs[1]
    for n,c in [(0,RED),(5,AMB),(50,GRN)]:
        P=np.mean([dyna_q(n,semente=k) for k in range(10)],axis=0); ax.plot(P,c=c,lw=2.2,label=f"{n} passos de planejamento")
    ax.set_yscale("log"); ax.set_xlabel("episódio"); ax.set_ylabel("passos até a meta (escala log)"); ax.set_title("Planejar acelera o aprendizado",fontsize=10); ax.legend(frameon=False,fontsize=8.5)
    save(fig,"f11_dyna")
def f11_horizonte():
    O,A,O2=coletar(8000,0); m=aprender_modelo(O,A,O2,n_arv=60)
    env=gym.make("CartPole-v1"); rng=np.random.default_rng(5); H=25; erros=[]; o,_=env.reset(seed=11); traj=[o]; acs=[]
    trajs=[]
    for _ in range(3000):
        a=int(rng.integers(2)); o2,r,te,tr,_=env.step(a); acs.append(a); traj.append(o2)
        if te or tr: trajs.append((np.array(traj),np.array(acs))); o,_=env.reset(); traj=[o]; acs=[]
    E=[[] for _ in range(H)]
    for tj,ac in trajs:
        for t0 in range(0,len(ac)-H):
            s=tj[t0].copy()
            for h in range(H):
                s=s+m.predict(np.append(s,ac[t0+h]).reshape(1,-1))[0]; E[h].append(np.abs(s-tj[t0+h+1]))
    E=np.array([np.mean(e,axis=0) for e in E]); fig,ax=plt.subplots(figsize=(7.6,3.6)); x=np.arange(1,H+1)
    for k,(n,c) in enumerate([("posição do carrinho",IND),("velocidade do carrinho",CYA),("ângulo do poste",AMB),("velocidade angular",RED)]): ax.plot(x,E[:,k],c=c,lw=2.2,label=n)
    ax.set_xlabel("passos previstos à frente (horizonte)"); ax.set_ylabel("erro absoluto médio"); ax.set_title("Erros do modelo se acumulam com o horizonte"); ax.legend(frameon=False,fontsize=8.5); save(fig,"f11_horizonte")
def f11_mcts_fases():
    fig,axs=plt.subplots(1,4,figsize=(10.4,3.3)); tit=["1. Seleção","2. Expansão","3. Simulação","4. Retropropagação"]; cols=[IND,GRN,AMB,RED]
    pos={0:(0.5,0.9),1:(0.25,0.62),2:(0.75,0.62),3:(0.12,0.34),4:(0.38,0.34),5:(0.62,0.34)}; arestas=[(0,1),(0,2),(1,3),(1,4),(2,5)]
    for k,ax in enumerate(axs):
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off"); ax.set_title(tit[k],fontsize=10,color=cols[k],fontweight="bold")
        for a,b in arestas: ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],c=MID,lw=1.5)
        for n,(x,y) in pos.items(): ax.scatter([x],[y],s=260,c="white",edgecolors=GRY,linewidths=1.5,zorder=3)
        if k==0:
            for a,b in [(0,1),(1,4)]: ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],c=IND,lw=3)
            ax.scatter([0.5,0.25,0.38],[0.9,0.62,0.34],s=260,c=IND,zorder=4); ax.text(0.5,0.08,"desce pela árvore\nescolhendo com UCT",ha="center",fontsize=8.5)
        if k==1:
            ax.scatter([0.12,0.38,0.62,0.25,0.75,0.5],[0.34,0.34,0.34,0.62,0.62,0.9],s=260,c="white",edgecolors=GRY,zorder=4); ax.plot([0.38,0.45],[0.34,0.1],c=GRN,lw=3); ax.scatter([0.45],[0.1],s=300,c=GRN,zorder=5); ax.text(0.5,0.0,"adiciona um nó novo",ha="center",fontsize=8.5,va="top")
        if k==2:
            ax.plot([0.38,0.45],[0.34,0.1],c=GRY,lw=1.5); ax.scatter([0.45],[0.1],s=300,c=GRN,zorder=5); ax.plot([0.45,0.32,0.6,0.5],[0.1,0.02,-0.04,-0.12],c=AMB,lw=2,ls="--",clip_on=False); ax.text(0.5,-0.2,"jogo aleatório até o fim",ha="center",fontsize=8.5,color="#B45309",clip_on=False)
        if k==3:
            for a,b in [(4,1),(1,0)]: ax.annotate("",xy=pos[b],xytext=pos[a],arrowprops=dict(arrowstyle="-|>",color=RED,lw=2.5))
            ax.scatter([0.5,0.25,0.38],[0.9,0.62,0.34],s=260,c=RED,zorder=4); ax.text(0.5,0.08,"atualiza vitórias e visitas\nno caminho de volta",ha="center",fontsize=8.5)
    save(fig,"f11_mcts_fases")
def f11_mcts_res():
    random.seed(0); sims=[2,5,20,50,200,500]; V=[];E=[];D=[]
    for n in sims: v,e,d=placar(n,60); V.append(v);E.append(e);D.append(d)
    fig,ax=plt.subplots(figsize=(7.6,3.5)); x=np.arange(len(sims)); V,E,D=map(np.array,(V,E,D)); tot=V+E+D
    ax.bar(x,V/tot*100,color=GRN,label="vitórias"); ax.bar(x,E/tot*100,bottom=V/tot*100,color=GRY,label="empates"); ax.bar(x,D/tot*100,bottom=(V+E)/tot*100,color=RED,label="derrotas")
    ax.set_xticks(x); ax.set_xticklabels(sims); ax.set_xlabel("simulações por jogada"); ax.set_ylabel("% das 60 partidas"); ax.set_title("MCTS × jogador aleatório na velha"); ax.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(0.5,-0.42)); save(fig,"f11_mcts_res")
def f11_mpc():
    fig,ax=canvas(10,3.6)
    for k,(x,t,c) in enumerate([(1,"1. Estado\natual",IND),(26,"2. Sorteia\nmuitas sequências\nde ações",CYA),(51,"3. Simula cada\numa no\nMODELO",AMB),(76,"4. Escolhe a\nmelhor e executa\nsó a 1ª ação",GRN)]):
        rbox(ax,x,32,22,36,t,fc="white",ec=c,lw=2.2,r=2.5,fs=8.8,bold=True)
        if k<3: arrow(ax,(x+22.2,50),(x+24.8,50),c=INK,lw=2,ms=9)
    arrow(ax,(85,30),(13,30),c=GRY,rad=-0.3,lw=1.6,ls="--"); ax.text(50,10,"repete a cada passo (replaneja): corrige os erros do modelo",ha="center",fontsize=9.5,color=GRY,style="italic")
    ax.text(50,92,"Controle preditivo por modelo (MPC) com amostragem aleatória",ha="center",fontsize=11.5,fontweight="bold")
    save(fig,"f11_mpc")
def f11_alphazero():
    fig,ax=canvas(10,3.9)
    rbox(ax,2,52,24,36,"Autojogo\nMCTS guiada pela rede",fc="#ECFDF5",ec=GRN,lw=2,r=2.5,fs=9.5,bold=True); rbox(ax,38,52,24,36,"Dados:\n(posição, π do MCTS,\nresultado z)",fc="#FFFBEB",ec=AMB,lw=2,r=2.5,fs=9.5,bold=True); rbox(ax,74,52,24,36,"Treino da rede\n(π̂, v̂) = f(posição)",fc="#EEF2FF",ec=IND,lw=2,r=2.5,fs=9.5,bold=True)
    arrow(ax,(26,70),(38,70),c=INK,lw=2); arrow(ax,(62,70),(74,70),c=INK,lw=2); arrow(ax,(86,52),(14,52),c=IND,lw=2,rad=-0.35); ax.text(50,24,"a rede melhorada guia a próxima rodada de autojogo",ha="center",fontsize=9.5,color=IND,fontweight="bold")
    ax.text(50,94,"AlphaZero: um ciclo de melhoria sem conhecimento humano",ha="center",fontsize=11.5,fontweight="bold"); ax.text(50,8,"a rede prevê (a) quais jogadas são promissoras e (b) quem está ganhando; a MCTS usa isso para buscar melhor",ha="center",fontsize=8.8,color=GRY,style="italic")
    save(fig,"f11_alphazero")
def f11_amostras():
    res=[]
    for n in [500,1500,5000]:
        O,A,O2=coletar(n,1); m=aprender_modelo(O,A,O2,n_arv=40); R=avaliar_mpc(m,n_ep=3); res.append((n,R.mean(),R.std()))
    fig,ax=plt.subplots(figsize=(7.4,3.5)); n,mu,sd=zip(*res); ax.bar([str(x) for x in n],mu,yerr=sd,color=[AMB,CYA,GRN],capsize=4); ax.axhline(22,c=RED,ls="--"); ax.text(2.4,25,"aleatório (≈ 22)",color=RED,fontsize=8.5,ha="right")
    ax.set_xlabel("transições aleatórias usadas para aprender o modelo"); ax.set_ylabel("retorno com MPC (3 episódios)"); ax.set_title("Eficiência de dados do aprendizado baseado em modelo"); save(fig,"f11_amostras")
if __name__=="__main__":
    import sys
    fs={f.__name__[4:]:f for f in [f11_ciclo,f11_dyna,f11_horizonte,f11_mcts_fases,f11_mcts_res,f11_mpc,f11_alphazero,f11_amostras]}
    for k in (sys.argv[1:] or fs): fs[k]()
