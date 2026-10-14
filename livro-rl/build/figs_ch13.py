from fc import *
from figs_cover import star
PAG=np.array([[(3,3),(0,5)],[(5,0),(1,1)]])
def ipd(n=20000,alpha=0.1,gamma=0.9,eps_ini=0.3,semente=0):
    rng=np.random.default_rng(semente); Q=[np.zeros((5,2)),np.zeros((5,2))]; est=[0,0]; coop=[]
    for t in range(n):
        eps=max(0.02,eps_ini*(1-t/(0.7*n))); ac=[int(rng.integers(2)) if rng.random()<eps else int(np.argmax(Q[i][est[i]]+1e-9*rng.random(2))) for i in range(2)]
        r=PAG[ac[0],ac[1]]; nv=[min(1+2*ac[i]+ac[1-i],4) for i in range(2)]
        for i in range(2): Q[i][est[i],ac[i]]+=alpha*(r[i]+gamma*Q[i][nv[i]].max()-Q[i][est[i],ac[i]])
        est=nv; coop.append(1-np.mean(ac))
    return np.array(coop)
def pennies(n=30000,alpha=0.05,T=0.3,semente=0):
    rng=np.random.default_rng(semente); Q=[np.zeros(2),np.zeros(2)]; P=[]; sm=lambda q:np.exp((q-q.max())/T)/np.exp((q-q.max())/T).sum()
    for t in range(n):
        p=[sm(Q[0]),sm(Q[1])]; a=[int(rng.random()<p[0][1]),int(rng.random()<p[1][1])]; r1=1.0 if a[0]==a[1] else -1.0
        Q[0][a[0]]+=alpha*(r1-Q[0][a[0]]); Q[1][a[1]]+=alpha*(-r1-Q[1][a[1]]); P.append((p[0][1],p[1][1]))
    return np.array(P)
def f13_jogos():
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.6))
    def matriz(ax,titulo,linhas,cols,cel,cores):
        ax.set_xlim(0,3); ax.set_ylim(-0.9,3); ax.axis("off"); ax.set_title(titulo,fontsize=10.5)
        for i in range(2):
            for j in range(2):
                ax.add_patch(Rectangle((1+j,1-i),0.96,0.96,fc=cores[i][j],ec="white",lw=2)); ax.text(1.48+j,1.48-i,cel[i][j],ha="center",va="center",fontsize=11,fontweight="bold")
        for j,c in enumerate(cols): ax.text(1.48+j,2.15,c,ha="center",fontsize=9.5,fontweight="bold")
        for i,l in enumerate(linhas): ax.text(0.9,1.48-i,l,ha="right",va="center",fontsize=9.5,fontweight="bold")
    matriz(axs[0],"Dilema do prisioneiro",["coopera","trai"],["coopera","trai"],[["3 , 3","0 , 5"],["5 , 0","1 , 1"]],[["#D1FAE5","#FEE2E2"],["#FEE2E2","#FEF3C7"]])
    matriz(axs[1],"Cara ou coroa (soma zero)",["cara","coroa"],["cara","coroa"],[["+1 , −1","−1 , +1"],["−1 , +1","+1 , −1"]],[["#DBEAFE","#FEE2E2"],["#FEE2E2","#DBEAFE"]])
    axs[0].text(1.5,-0.4,"equilíbrio de Nash: (trai, trai)\nmas (coopera, coopera) pagaria mais a ambos",ha="center",fontsize=8.5,color=RED,fontweight="bold"); axs[1].text(1.5,-0.4,"equilíbrio só em estratégias mistas:\ncada um escolhe 50%–50%",ha="center",fontsize=8.5,color=IND,fontweight="bold")
    save(fig,"f13_jogos")
def f13_tipos():
    fig,axs=plt.subplots(1,3,figsize=(10,3.2)); tit=[("Cooperativo","mesmo objetivo\n(equipe de robôs)",GRN),("Competitivo","objetivos opostos\n(xadrez, pôquer)",RED),("Misto","interesses parcialmente\ncomuns (trânsito, mercado)",AMB)]
    for ax,(t,d,c) in zip(axs,tit):
        ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off"); ax.set_title(t,color=c,fontsize=11,fontweight="bold")
        for x,y in [(2.5,6),(7.5,6),(5,2.8)]: ax.scatter([x],[y],s=900,c=c,alpha=0.85,edgecolors="white",linewidths=2)
        if t=="Cooperativo":
            for a,b in [((2.5,6),(7.5,6)),((2.5,6),(5,2.8)),((7.5,6),(5,2.8))]: ax.plot([a[0],b[0]],[a[1],b[1]],c=GRN,lw=2); 
            ax.scatter([5],[9],marker="*",s=500,c=AMB,zorder=5)
        elif t=="Competitivo":
            ax.annotate("",xy=(7,6),xytext=(3,6),arrowprops=dict(arrowstyle="<|-|>",color=RED,lw=2.5)); ax.text(5,6.8,"×",ha="center",fontsize=20,color=RED,fontweight="bold")
        else:
            ax.plot([2.5,7.5],[6,6],c=AMB,lw=2); ax.annotate("",xy=(5.2,3.2),xytext=(2.8,5.6),arrowprops=dict(arrowstyle="-|>",color=RED,lw=2,ls="--")); ax.plot([7.5,5],[6,2.8],c=GRN,lw=2)
        ax.text(5,0.3,d,ha="center",fontsize=9)
    save(fig,"f13_tipos")
def f13_ctde():
    fig,ax=canvas(10,3.9)
    for x in [6,39,72]: rbox(ax,x,4,22,26,"Agente\nvê só o que\nobserva",fc="#EEF2FF",ec=IND,lw=2,r=2.5,fs=9,bold=True)
    rbox(ax,10,58,80,26,"CRÍTICO CENTRALIZADO (usado só no treino)\nvê observações e ações de TODOS os agentes",fc="#FFFBEB",ec=AMB,lw=2.2,r=2.5,fs=10,bold=True)
    for x in [17,50,83]: arrow(ax,(x,30.5),(x,57.5),c=AMB,lw=1.8,ms=10,style="<|-|>")
    ax.text(50,94,"Treino centralizado, execução descentralizada (CTDE)",ha="center",fontsize=11.5,fontweight="bold"); ax.text(50,44,"no treino: o crítico ajuda cada agente",ha="center",fontsize=9,color="#B45309",fontweight="bold"); ax.text(50,-3,"na execução, cada agente usa só a própria política e as próprias observações",ha="center",fontsize=9,color=GRY,style="italic")
    save(fig,"f13_ctde")
def f13_rlhf():
    fig,ax=canvas(10,4.4)
    et=[("1. Ajuste supervisionado","humanos escrevem boas\nrespostas; o modelo\naprende a imitá-las",IND),("2. Modelo de recompensa","humanos comparam pares\nde respostas; um modelo\naprende r(x, y)",AMB),("3. Otimização por RL","o modelo é ajustado (PPO)\npara maximizar r, com\npenalidade KL ao original",GRN)]
    for k,(t,d,c) in enumerate(et):
        x=3+k*33; rbox(ax,x,22,30,60,"",fc="white",ec=c,lw=2.2,r=3); rbox(ax,x,68,30,14,t,fc=c,tc="white",bold=True,fs=9.5,r=3); ax.text(x+15,44,d,ha="center",va="center",fontsize=9,linespacing=1.5)
        if k<2: arrow(ax,(x+30.3,52),(x+32.7,52),c=INK,lw=2,ms=9)
    ax.text(50,94,"RLHF: aprendizado por reforço a partir de feedback humano",ha="center",fontsize=11.5,fontweight="bold"); ax.text(50,9,"a recompensa deixa de ser escrita à mão e passa a ser aprendida das preferências humanas",ha="center",fontsize=9.5,color=GRY,style="italic")
    save(fig,"f13_rlhf")
def f13_bradley():
    fig,ax=plt.subplots(figsize=(6.6,3.4)); d=np.linspace(-6,6,200); ax.plot(d,1/(1+np.exp(-d)),c=IND,lw=3); ax.axhline(0.5,c=MID,lw=1); ax.axvline(0,c=MID,lw=1)
    for x,t in [(-3,"r(B) bem maior que r(A)"),(3,"r(A) bem maior que r(B)")]: ax.text(x,0.85 if x>0 else 0.15,t,ha="center",fontsize=8.5,color=GRY)
    ax.set_xlabel("r(A) − r(B)"); ax.set_ylabel("P(humano prefere A a B)"); ax.set_title("Modelo de Bradley–Terry: preferência = sigmoide da diferença de recompensas",fontsize=9.5); save(fig,"f13_bradley")
def f13_goodhart():
    rng=np.random.default_rng(0); n=300000; verd=rng.normal(0,1,n); erro=rng.standard_t(2.2,n)*0.5; proxy=verd+erro
    Ns=np.unique(np.logspace(0,4.7,26).astype(int)); R=[];V=[]
    for N in Ns:
        k=max(30,int(40000//N)); idx=rng.integers(0,n,(k,N)); b=idx[np.arange(k),np.argmax(proxy[idx],axis=1)]; R.append(proxy[b].mean()); V.append(verd[b].mean())
    V=np.convolve(np.pad(V,(1,1),mode="edge"),np.ones(3)/3,mode="valid"); fig,ax=plt.subplots(figsize=(8,3.7)); ax2=ax.twinx(); ax2.spines["right"].set_visible(True)
    ax.semilogx(Ns,V,c=GRN,lw=2.8,label="recompensa verdadeira (eixo esquerdo)"); ax2.semilogx(Ns,R,c=RED,lw=2.4,label="recompensa proxy (eixo direito)"); i=int(np.argmax(V)); ax.scatter([Ns[i]],[V[i]],s=120,c=GRN,zorder=5)
    ax.annotate("a partir daqui, otimizar\nmais piora o que importa",xy=(Ns[i],V[i]),xytext=(Ns[i]*3,V[i]-0.35),arrowprops=dict(arrowstyle="->",color=GRN),fontsize=9,color="#047857",fontweight="bold")
    ax.set_xlabel("quanto se otimiza o proxy (melhor de N candidatos)"); ax.set_ylabel("verdadeira",color="#047857"); ax2.set_ylabel("proxy",color=RED); ax.set_title("Lei de Goodhart: otimizar demais um proxy imperfeito (simulação)")
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,frameon=False,fontsize=8.5,loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=2); save(fig,"f13_goodhart")
def f13_ipd():
    fim=[ipd(semente=s)[-2000:].mean() for s in range(40)]; ex=[ipd(semente=s) for s in (3,0)]
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.5)); ax=axs[0]; ax.hist(fim,bins=np.linspace(0,1,11),color=IND,alpha=0.85,edgecolor="white"); ax.set_xlabel("fração de cooperação no final do treino"); ax.set_ylabel("nº de execuções (40 sementes)"); ax.set_title("Cooperar ou trair? Depende da semente",fontsize=10)
    ax=axs[1]; k=500
    for c,(lab,col) in zip(ex,[("semente 3: cooperação",GRN),("semente 0: traição mútua",RED)]): ax.plot(np.convolve(c,np.ones(k)/k,mode="valid"),c=col,lw=2.2,label=lab)
    ax.set_xlabel("rodada"); ax.set_ylabel("fração de cooperação (média móvel)"); ax.set_title("Duas execuções, dois resultados",fontsize=10); ax.legend(frameon=False,fontsize=8.5); save(fig,"f13_ipd")
def f13_pennies():
    P=pennies(); fig,ax=plt.subplots(figsize=(8,3.4)); k=400; ax.plot(np.convolve(P[:,0],np.ones(k)/k,mode="valid"),c=IND,lw=2.2,label="jogador 1: P(coroa) (média móvel)"); ax.plot(np.convolve(P[:,1],np.ones(k)/k,mode="valid"),c=AMB,lw=2.2,label="jogador 2: P(coroa) (média móvel)"); ax.axhline(0.5,c=INK,ls="--",lw=1.2); ax.text(P.shape[0]*0.99,0.52,"equilíbrio de Nash (0,5)",ha="right",fontsize=9)
    ax.set_xlabel("rodada"); ax.set_ylabel("probabilidade"); ax.set_ylim(0.2,0.8); ax.set_title("Cara ou coroa com aprendizes independentes",fontsize=11); ax.legend(frameon=False,fontsize=8.5,loc="upper right"); save(fig,"f13_pennies")
def f13_roteiro():
    fig,ax=canvas(10,4.4); camadas=[("Fundamentos","bandits,\nMDPs, PD",IND),("Tabulares","Monte Carlo,\nTD",CYA),("Profundo","DQN, PG,\nPPO, SAC",GRN),("Avançado","modelo, offline,\nmultiagente, RLHF",AMB)]
    for k,(t,d,c) in enumerate(camadas):
        x=2+k*25; rbox(ax,x,48,21,34,"",fc="white",ec=c,lw=2.2,r=2.5); ax.text(x+10.5,74,t,ha="center",fontsize=9.5,fontweight="bold",color=c if c!=AMB else "#B45309"); ax.text(x+10.5,59,d,ha="center",fontsize=8.5,va="center",linespacing=1.5)
        if k<3: arrow(ax,(x+21.3,65),(x+24.7,65),c=INK,lw=2,ms=9)
    for x,t,c in [(2,"projetos de\ncapítulo",IND),(37,"projeto\nintegrador",RED),(72,"seu próprio\nproblema",GRN)]: rbox(ax,x,8,26,22,t,fc=c,tc="white",bold=True,fs=9.5,r=2)
    for x in [15,50,85]: arrow(ax,(x,47),(x,31),c=GRY,ms=9)
    ax.text(50,95,"O caminho percorrido e o que vem depois",ha="center",fontsize=11.5,fontweight="bold"); save(fig,"f13_roteiro")
if __name__=="__main__":
    import sys
    fs={f.__name__[4:]:f for f in [f13_jogos,f13_tipos,f13_ctde,f13_rlhf,f13_bradley,f13_goodhart,f13_ipd,f13_pennies,f13_roteiro]}
    for k in (sys.argv[1:] or fs): fs[k]()
