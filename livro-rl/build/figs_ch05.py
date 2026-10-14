from fc import *
from figs_cover import star
import gymnasium as gym, time
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
env=gym.make("Blackjack-v1",sab=True)
def gerar(pol):
    obs,_=env.reset(); ep=[]; fim=False
    while not fim:
        a=pol(obs); o2,r,te,tr,_=env.step(a); ep.append((obs,a,r)); obs=o2; fim=te or tr
    return ep
def retornos(ep):
    G=0.0; ret=[0]*len(ep)
    for t in range(len(ep)-1,-1,-1): G+=ep[t][2]; ret[t]=G
    return ret
def pol20(o): return 0 if o[0]>=20 else 1
def mc_pred(n):
    soma=defaultdict(float); cont=defaultdict(int)
    for _ in range(n):
        ep=gerar(pol20); ret=retornos(ep); vis=set()
        for t,(s,a,r) in enumerate(ep):
            if s not in vis: vis.add(s); soma[s]+=ret[t]; cont[s]+=1
    return {s:soma[s]/cont[s] for s in soma}
def grade_v(V,ace):
    M=np.full((10,10),np.nan)
    for (j,d,u),v in V.items():
        if u==ace and 12<=j<=21 and 1<=d<=10: M[j-12,d-1]=v
    return M
def f5_blackjack_v():
    V1=mc_pred(10000); V2=mc_pred(300000); cm=LinearSegmentedColormap.from_list("v",[RED,"#FEF3C7",GRN])
    fig,axs=plt.subplots(1,3,figsize=(8.6,3.2))
    for ax,(V,ace,t) in zip(axs,[(V1,False,"10 mil episódios\n(sem ás usável)"),(V2,False,"300 mil episódios\n(sem ás usável)"),(V2,True,"300 mil episódios\n(com ás usável)")]):
        M=grade_v(V,ace); im=ax.imshow(M,origin="lower",cmap=cm,vmin=-1,vmax=1,aspect="auto")
        ax.set_xticks(range(10)); ax.set_xticklabels(["A"]+[str(i) for i in range(2,11)]); ax.set_yticks(range(10)); ax.set_yticklabels(range(12,22)); ax.set_xlabel("carta visível do crupiê"); ax.set_title(t,fontsize=9.5)
    axs[0].set_ylabel("soma do jogador"); fig.subplots_adjust(left=0.08,right=0.88,bottom=0.2,top=0.82,wspace=0.25); cax=fig.add_axes([0.9,0.2,0.015,0.62]); fig.colorbar(im,cax=cax,label="valor estimado")
    fig.savefig(OUT+"f5_blackjack_v.png",facecolor="white",bbox_inches="tight",pad_inches=0.12); plt.close(fig); print("fig f5_blackjack_v")
def f5_controle():
    rng=np.random.default_rng(0); Q=defaultdict(lambda:np.zeros(2)); N=defaultdict(lambda:np.zeros(2))
    def pol_eps(eps):
        def p(o):
            if rng.random()<eps: return int(rng.integers(2))
            q=Q[o]; return int(q.argmax()) if q[0]!=q[1] else int(rng.integers(2))
        return p
    n=300000; marc=[]
    for k in range(n):
        eps=max(0.05,0.5*(1-k/n)); ep=gerar(pol_eps(eps)); ret=retornos(ep); vis=set()
        for t,(s,a,r) in enumerate(ep):
            if (s,a) not in vis: vis.add((s,a)); N[s][a]+=1; Q[s][a]+=(ret[t]-Q[s][a])/N[s][a]
        if (k+1) in (500,1000,2000,5000,10000,25000,50000,100000,200000,300000):
            g=pol_eps(0.0); R=[gerar(g)[-1][2] for _ in range(20000)]; marc.append((k+1,np.mean(R)))
    fig,axs=plt.subplots(1,3,figsize=(8.6,3.3),gridspec_kw={"width_ratios":[1.4,1,1]})
    x,y=zip(*marc); axs[0].semilogx(x,y,c=IND,lw=2.2,marker="o",ms=4); axs[0].axhline(-0.354,c=RED,ls="--",lw=1.2); axs[0].text(x[-1],-0.33,"regra \"parar só em 20\"",color=RED,fontsize=8,ha="right")
    axs[0].set_xlabel("episódios de treino"); axs[0].set_ylabel("retorno médio"); axs[0].set_title("O agente aprende a jogar")
    cmp_=ListedColormap(["#FDE68A",IND])
    for ax,ace,t in [(axs[1],False,"sem ás usável"),(axs[2],True,"com ás usável")]:
        M=np.full((10,10),np.nan)
        for (j,d,u),q in Q.items():
            if u==ace and 12<=j<=21 and 1<=d<=10 and N[(j,d,u)].sum()>0: M[j-12,d-1]=int(q.argmax())
        ax.imshow(M,origin="lower",cmap=cmp_,vmin=0,vmax=1,aspect="auto"); ax.set_xticks(range(10)); ax.set_xticklabels(["A"]+[str(i) for i in range(2,11)],fontsize=8); ax.set_yticks(range(10)); ax.set_yticklabels(range(12,22),fontsize=8)
        ax.set_title(f"política: {t}",fontsize=9.5); ax.set_xlabel("carta do crupiê")
    fig.text(0.62,0.01,"amarelo = parar     azul = pedir carta",fontsize=8.5,ha="center"); fig.tight_layout(rect=(0,0.05,1,1))
    fig.savefig(OUT+"f5_controle.png",facecolor="white",bbox_inches="tight",pad_inches=0.12); plt.close(fig); print("fig f5_controle")
def f5_is():
    rng=np.random.default_rng(1); rho=[];G=[]
    for _ in range(100000):
        obs,_=env.reset(); fim=False; r_=1.0; ret=0.0
        while not fim:
            a=int(rng.integers(2)); r_*=(2.0 if a==pol20(obs) else 0.0); obs,r,te,tr,_=env.step(a); ret+=r; fim=te or tr
        rho.append(r_);G.append(ret)
    rho=np.array(rho);G=np.array(G); n=np.arange(1,len(G)+1); o=np.cumsum(rho*G)/n; w=np.cumsum(rho*G)/np.maximum(np.cumsum(rho),1e-12)
    fig,ax=plt.subplots(figsize=(8,3.6)); ax.semilogx(n[9:],o[9:],c=RED,lw=1.6,label="amostragem por importância ordinária"); ax.semilogx(n[9:],w[9:],c=IND,lw=2,label="amostragem por importância ponderada")
    ax.axhline(-0.354,c=GRN,ls="--",lw=1.3); ax.text(2500,-0.42,"valor verdadeiro ≈ −0,354",color="#047857",fontsize=9)
    ax.set_xlabel("episódios gerados pela política de comportamento (aleatória)"); ax.set_ylabel("estimativa do valor da política-alvo"); ax.set_ylim(-0.9,0.1); ax.set_title("Aprendendo sobre uma política observando outra"); ax.legend(frameon=False,fontsize=8.5,loc="lower right")
    save(fig,"f5_is")
def f5_idea():
    fig,ax=canvas(10,3.8)
    dot(ax,8,50,s=1500,c=IND); ax.text(8,50,"s",ha="center",va="center",color="white",fontsize=14,fontweight="bold")
    trajs=[(78,[(25,70),(45,78),(65,72)],"G = +1",GRN),(50,[(25,52),(48,45),(66,52)],"G = −1",RED),(22,[(25,30),(45,20),(66,28)],"G = +1",GRN)]
    for yend,pts_,lab,c in trajs:
        px=[8]+[p[0] for p in pts_]+[88]; py=[50]+[p[1] for p in pts_]+[yend]
        ax.plot(px,py,c=c,lw=2,alpha=0.85); ax.scatter(px[1:-1],py[1:-1],s=40,c=c,zorder=3); ax.scatter([88],[yend],marker="s",s=140,c=c,zorder=3)
        ax.text(93,yend,lab,fontsize=10.5,fontweight="bold",color=c if c!=GRN else "#047857",va="center")
    ax.text(50,95,"Monte Carlo: jogue vários episódios a partir de s e tire a média dos retornos",ha="center",fontsize=11,fontweight="bold")
    ax.text(50,5,"v(s) ≈ ( +1 − 1 + 1 ) / 3 = 0,33        (com milhares de episódios, a média se estabiliza)",ha="center",fontsize=10,color=IND,fontweight="bold")
    save(fig,"f5_idea")
def f5_visita():
    fig,ax=canvas(10,3.2)
    seq=["A","B","C","B","D","fim"]; xs=np.linspace(8,82,6)
    for i,(x,s) in enumerate(zip(xs,seq)):
        c=AMB if s=="B" else (GRN if s=="fim" else IND); dot(ax,x,45,s=1100,c=c); ax.text(x,45,s,ha="center",va="center",color="white",fontsize=12,fontweight="bold")
        if i<5: arrow(ax,(x+4.3,45),(xs[i+1]-4.6,45),c=GRY,lw=1.6)
    ax.text(xs[1],68,"1ª visita a B",ha="center",fontsize=10,fontweight="bold",color="#B45309"); ax.text(xs[3],68,"2ª visita a B",ha="center",fontsize=10,fontweight="bold",color="#B45309")
    ax.text(24,14,"Primeira visita:\nretorno só da 1ª visita",ha="center",fontsize=9.5,color=IND,fontweight="bold"); ax.text(70,14,"Toda visita:\nretorno de cada visita",ha="center",fontsize=9.5,color=IND,fontweight="bold")
    save(fig,"f5_visita")
def f5_backup():
    fig,axs=plt.subplots(1,2,figsize=(8,3.8))
    for ax in axs: ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    ax=axs[0]; ax.set_title("Programação dinâmica\n(uma etapa, com o modelo)",fontsize=10,color=IND)
    ax.scatter([5],[9],s=700,facecolors="white",edgecolors=IND,linewidths=2.5); ax.scatter([3.5,6.5],[6,6],s=100,c=INK)
    for x in [3.5,6.5]: ax.plot([5,x],[8.5,6.2],c=INK,lw=1.2)
    for x0,xs_ in [(3.5,[2.6,4.4]),(6.5,[5.6,7.4])]:
        for x in xs_: ax.plot([x0,x],[5.8,3],c=GRY,lw=1.2); ax.scatter([x],[2.6],s=500,facecolors="white",edgecolors=GRN,linewidths=2)
    ax.text(5,0.6,"olha só um passo à frente\n(usa estimativas dos vizinhos)",ha="center",fontsize=8.5,color=GRY)
    ax=axs[1]; ax.set_title("Monte Carlo\n(episódio completo, sem modelo)",fontsize=10,color=AMB)
    ax.scatter([5],[9],s=700,facecolors="white",edgecolors=IND,linewidths=2.5); 
    pts_=[(5,9),(4,7.2),(5.5,5.4),(4.5,3.6),(5.2,1.9)]
    for p,q in zip(pts_[:-1],pts_[1:]): ax.plot([p[0],q[0]],[p[1]-0.4,q[1]+0.1],c=INK,lw=1.6)
    for p in pts_[1:-1]: ax.scatter([p[0]],[p[1]],s=60,c=INK,zorder=3)
    ax.scatter([5.2],[1.6],marker="s",s=260,c=GRN,zorder=3); ax.text(6.3,1.6,"fim",fontsize=9,color="#047857",fontweight="bold")
    ax.text(5,0.0,"percorre até o fim e usa o retorno real",ha="center",fontsize=8.5,color=GRY)
    save(fig,"f5_backup")
if __name__=="__main__":
    import sys
    fs={"idea":f5_idea,"visita":f5_visita,"backup":f5_backup,"bj":f5_blackjack_v,"ctl":f5_controle,"is":f5_is}
    for k in (sys.argv[1:] or fs): fs[k]()
