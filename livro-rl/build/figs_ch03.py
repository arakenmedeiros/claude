from fc import *
from figs_cover import star
import gymnasium as gym

def f3_markov():
    fig,ax=canvas(10,3.0,(0,100),(0,40))
    xs=[8,24,40,56,72,90]; labs=["s₀","s₁","s₂","s₃","s₄","s₅"]
    for i,(x,l) in enumerate(zip(xs,labs)):
        a=0.25+0.15*i if i<5 else 1; c=IND if i<5 else AMB
        dot(ax,x,22,s=900 if i==5 else 600,c=c,alpha=0.35 if i<4 else (0.7 if i==4 else 1)); ax.text(x,22,l,ha="center",va="center",color="white",fontweight="bold",fontsize=11)
        if i<5: arrow(ax,(x+4,22),(xs[i+1]-4.5,22),c=GRY,lw=1.5)
    ax.text(24,36,"passado (pode ser esquecido)",ha="center",fontsize=10,color=GRY,style="italic")
    ax.text(90,33,"presente",ha="center",fontsize=11,fontweight="bold",color="#B45309")
    ax.text(50,6,"Propriedade de Markov: o estado atual já contém tudo o que importa do passado para prever o futuro.",ha="center",fontsize=10,fontweight="bold",color=IND)
    save(fig,"f3_markov")

def f3_robo():
    fig,ax=canvas(10,5.2)
    def estado(x,y,t,c): dot(ax,x,y,s=3600,c=c,lw=2.5); ax.text(x,y,t,ha="center",va="center",color="white",fontweight="bold",fontsize=11,linespacing=1.3)
    def acao(x,y): ax.scatter([x],[y],s=60,c=INK,zorder=5)
    estado(20,52,"bateria\nALTA",GRN); estado(80,52,"bateria\nBAIXA",RED)
    # alta-buscar
    acao(34,80); arrow(ax,(25,59),(33,79),c=INK,lw=1.4); ax.text(26,75,"buscar",fontsize=9,fontweight="bold",rotation=45)
    arrow(ax,(35,80),(21,58),c=GRN,lw=1.6,rad=-0.25); ax.text(18,80,"0,7 | +4",fontsize=8.5,color="#047857",fontweight="bold")
    arrow(ax,(35,80),(74,59),c=AMB,lw=1.6,rad=-0.15); ax.text(55,84,"0,3 | +4",fontsize=8.5,color="#B45309",fontweight="bold")
    # alta-esperar
    acao(9,28); arrow(ax,(16,45),(9.6,29),c=INK,lw=1.4); ax.text(2,36,"esperar",fontsize=9,fontweight="bold")
    arrow(ax,(9,27),(17,46),c=GRN,lw=1.6,rad=0.5); ax.text(10,18,"1,0 | +1",fontsize=8.5,color="#047857",fontweight="bold")
    # baixa-buscar
    acao(66,26); arrow(ax,(75,45),(67,27),c=INK,lw=1.4); ax.text(61,38,"buscar",fontsize=9,fontweight="bold",rotation=-50)
    arrow(ax,(66,26),(78,45),c=RED,lw=1.6,rad=0.3); ax.text(71,21,"0,6 | +4",fontsize=8.5,color="#B91C1C",fontweight="bold")
    arrow(ax,(65,26),(25,45),c=AMB,lw=1.6,rad=-0.2); ax.text(38,26,"0,4 | −3  (resgate)",fontsize=8.5,color="#B45309",fontweight="bold")
    # baixa-esperar
    acao(93,28); arrow(ax,(86,45),(92.5,29),c=INK,lw=1.4); ax.text(95,37,"esperar",fontsize=9,fontweight="bold",ha="center")
    arrow(ax,(93,27),(84,46),c=RED,lw=1.6,rad=-0.5); ax.text(86,18,"1,0 | +1",fontsize=8.5,color="#B91C1C",fontweight="bold")
    # baixa-recarregar
    acao(50,62); arrow(ax,(73,56),(51,62),c=INK,lw=1.4); ax.text(56,67,"recarregar",fontsize=9,fontweight="bold")
    arrow(ax,(50,62),(27,56),c=CYA,lw=1.8); ax.text(34,58,"1,0 | 0",fontsize=8.5,color="#0E7490",fontweight="bold")
    ax.text(50,97,"Um MDP pequeno: o robô de entrega e sua bateria",ha="center",fontsize=11.5,fontweight="bold",color=INK)
    ax.text(50,3,"rótulos “p | r”: probabilidade da transição | recompensa recebida",ha="center",fontsize=8.5,color=GRY,style="italic")
    save(fig,"f3_robo")

def f3_backup():
    fig,axs=plt.subplots(1,2,figsize=(9,3.8))
    for ax in axs: ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax=axs[0]; ax.set_title("Valor do estado  v$_π$(s)",color=IND)
    ax.scatter([5],[9],s=900,facecolors="white",edgecolors=IND,linewidths=2.5,zorder=3); ax.text(5,9,"s",ha="center",va="center",fontweight="bold")
    for x in [3,7]:
        ax.plot([5,x],[8.4,5.7],c=INK,lw=1.4); ax.scatter([x],[5.4],s=120,c=INK,zorder=3)
    ax.text(3.2,6.6,"π(a₁|s)",fontsize=8.5,color=INK); ax.text(6.0,6.6,"π(a₂|s)",fontsize=8.5,color=INK)
    for x0,xs_ in [(3,[2,4]),(7,[6,8])]:
        for x in xs_:
            ax.plot([x0,x],[5.1,2.4],c=GRY,lw=1.2); ax.scatter([x],[2],s=700,facecolors="white",edgecolors=GRN,linewidths=2,zorder=3); ax.text(x,2,"s′",ha="center",va="center",fontsize=9)
    ax.text(5,0.4,"média sobre ações (política) e sobre\nresultados possíveis (ambiente)",ha="center",fontsize=8.5,color=GRY)
    ax=axs[1]; ax.set_title("Valor da ação  q$_π$(s,a)",color=AMB)
    ax.scatter([5],[9],s=120,c=INK,zorder=3); ax.text(5.5,9.0,"(s, a)",fontsize=9,fontweight="bold")
    for x in [3,7]:
        ax.plot([5,x],[8.7,5.0],c=GRY,lw=1.2); ax.scatter([x],[4.6],s=700,facecolors="white",edgecolors=GRN,linewidths=2,zorder=3); ax.text(x,4.6,"s′",ha="center",va="center",fontsize=9)
    for x0,xs_ in [(3,[2,4]),(7,[6,8])]:
        for x in xs_:
            ax.plot([x0,x],[4.1,1.6],c=INK,lw=1.2); ax.scatter([x],[1.3],s=100,c=INK,zorder=3)
    ax.text(5,0.1,"depois do resultado, escolhe-se a próxima ação pela política",ha="center",fontsize=8.5,color=GRY)
    save(fig,"f3_backup")

def f3_frozenlake():
    env=gym.make("FrozenLake-v1",is_slippery=True); Pf=env.unwrapped.P; nS,nA=16,4
    Pm=np.zeros((nS,nA,nS)); Rm=np.zeros((nS,nA))
    for s in range(nS):
        for a in range(nA):
            for pr,s2,r,d in Pf[s][a]: Pm[s,a,s2]+=pr; Rm[s,a]+=pr*r
    pol=np.full((nS,nA),0.25); Ppi=np.einsum("sa,sat->st",pol,Pm); rpi=(pol*Rm).sum(1); v=np.linalg.solve(np.eye(nS)-0.99*Ppi,rpi)
    mapa=["SFFF","FHFH","FFFH","HFFG"]
    fig,axs=plt.subplots(1,2,figsize=(9,4.1))
    ax=axs[0]; ax.set_xlim(0,4); ax.set_ylim(0,4); ax.axis("off"); ax.set_title("O lago congelado (FrozenLake)")
    for i in range(4):
        for j in range(4):
            c=mapa[i][j]; col={"S":"#BFDBFE","F":"#E0F2FE","H":"#1E3A8A","G":"#FDE68A"}[c]
            ax.add_patch(Rectangle((j,3-i),0.96,0.96,fc=col,ec="white")); 
            ax.text(j+0.48,3-i+0.48,{"S":"início","F":"gelo","H":"buraco","G":"meta ★"}[c],ha="center",va="center",fontsize=9,color="white" if c=="H" else INK,fontweight="bold")
    ax=axs[1]; ax.set_xlim(0,4); ax.set_ylim(0,4); ax.axis("off"); ax.set_title("Valor v$_π$(s) da política aleatória (γ = 0,99)")
    from matplotlib.colors import LinearSegmentedColormap
    cm=LinearSegmentedColormap.from_list("v",["#EEF2FF",CYA,GRN,"#FDE68A"]); vm=v.max()
    for s in range(16):
        i,j=divmod(s,4); c=mapa[i][j]
        col="#1E3A8A" if c=="H" else cm(v[s]/vm)
        ax.add_patch(Rectangle((j,3-i),0.96,0.96,fc=col,ec="white")); ax.text(j+0.48,3-i+0.48,f"{v[s]:.3f}" if c!="H" else "0",ha="center",va="center",fontsize=10,fontweight="bold",color="white" if c=="H" else INK)
    save(fig,"f3_frozenlake")

def f3_clientes():
    W,H=9.5,4.2; fig,ax=canvas(W,H)
    nomes=["Novo /\nocasional","Regular","Fiel","Em risco","Inativo"]; pos=[(12,52),(36,80),(66,80),(48,24),(88,52)]; cols=[CYA,IND,GRN,AMB,RED]
    for (x,y),n,c in zip(pos,nomes,cols): dot(ax,x,y,s=3400,c=c,lw=2.5); ax.text(x,y,n,ha="center",va="center",color="white",fontweight="bold",fontsize=9.5,linespacing=1.2)
    E=[(0,1,0.4,0.0),(1,2,0.5,0.18),(2,1,0.2,0.18),(1,3,0.2,0.12),(3,4,0.5,0.1),(3,1,0.2,0.12),(0,3,0.3,0.1),(2,4,0.2,0.1),(4,3,0.1,0.1)]
    E=[e for e in E if not (e[0]==2 and e[1]==1) or True]
    kx,ky=W/100,H/100
    for a,b,p,rad in E:
        (x1,y1),(x2,y2)=pos[a],pos[b]; dx,dy=(x2-x1)*kx,(y2-y1)*ky; d=np.hypot(dx,dy); ux,uy=dx/d,dy/d; r=0.42
        s=(x1+ux*r/kx,y1+uy*r/ky); t=(x2-ux*(r+0.05)/kx,y2-uy*(r+0.05)/ky)
        sgn=1 if (a,b) not in [(2,1),(3,1),(4,3)] else 1
        arrow(ax,s,t,c=GRY,lw=1.4+3.5*p,rad=rad*(1 if a<b else 1),ms=10)
        mx,my=(s[0]+t[0])/2,(s[1]+t[1])/2; nx,ny=-uy,ux; off=rad*d*0.5
        ax.text(mx+nx*off/kx*(-1),my+ny*off/ky*(-1),f"{p:.1f}".replace(".",","),ha="center",va="center",fontsize=8.5,fontweight="bold",color=INK,bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=0.9))
    ax.text(50,97,"Cadeia de Markov de clientes de uma loja (valores ilustrativos)",ha="center",fontsize=11.5,fontweight="bold",color=INK)
    ax.text(50,3,"cada seta: probabilidade de passar de um estado a outro no mês seguinte",ha="center",fontsize=8.5,color=GRY,style="italic")
    save(fig,"f3_clientes")
if __name__=="__main__":
    for f in [f3_markov,f3_robo,f3_backup,f3_frozenlake,f3_clientes]: f()
