from fc import *
from dp_lib import *
from figs_cover import star
import gymnasium as gym
from matplotlib.colors import LinearSegmentedColormap
SETAS={0:(0,0.28),1:(0,-0.28),2:(-0.28,0),3:(0.28,0)}
def desenha_grade(ax,V,n=4,pol=None,term=(0,15),fmt="{:.0f}",cmap=None,vmin=None,vmax=None):
    ax.set_xlim(0,n); ax.set_ylim(0,n); ax.axis("off")
    cm=cmap or LinearSegmentedColormap.from_list("v",["#4F46E5","#818CF8","#E0E7FF","white"])
    vmin=V.min() if vmin is None else vmin; vmax=V.max() if vmax is None else vmax
    for s in range(n*n):
        i,j=divmod(s,n); y=n-1-i; t=(V[s]-vmin)/(vmax-vmin+1e-9)
        col="#FDE68A" if s in term else cm(t)
        ax.add_patch(Rectangle((j,y),0.96,0.96,fc=col,ec="white"))
        if s in term: ax.text(j+0.48,y+0.48,"★",ha="center",va="center",fontsize=16,color="#B45309")
        elif pol is None: ax.text(j+0.48,y+0.48,fmt.format(V[s]),ha="center",va="center",fontsize=9,fontweight="bold")
        else:
            ax.text(j+0.48,y+0.2,fmt.format(V[s]),ha="center",va="center",fontsize=7.5,color=INK)
            for a in np.where(pol[s]>0.5)[0]:
                dx,dy=SETAS[a]; ax.annotate("",xy=(j+0.48+dx,y+0.6+dy),xytext=(j+0.48,y+0.6),arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.5))
def f4_avaliacao():
    P,R,term=grade(4); pol=np.full((16,4),0.25); hist=[]; Vinf,k=avaliacao(P,R,pol,1.0,1e-10,hist=hist)
    ks=[0,1,2,3,10]; fig,axs=plt.subplots(2,3,figsize=(9.5,6.2)); axs=axs.ravel()
    for ax,kk in zip(axs,ks):
        V=np.zeros(16) if kk==0 else hist[kk-1]; desenha_grade(ax,V,fmt="{:.1f}",vmin=-22,vmax=0); ax.set_title(f"k = {kk}",fontsize=10)
    Q=gulosa(P,R,Vinf); pg=np.zeros((16,4)); pg[np.arange(16),np.isclose(Q,Q.max(1,keepdims=True)).astype(float).argmax(1)]=0
    pg=(np.isclose(Q,Q.max(1,keepdims=True))).astype(float)
    desenha_grade(axs[5],Vinf,pol=pg,fmt="{:.0f}",vmin=-22,vmax=0); axs[5].set_title("k = ∞  e política gulosa",fontsize=10)
    fig.suptitle("Avaliação iterativa da política aleatória (recompensa −1 por passo; ★ = destino)",fontweight="bold",fontsize=11)
    save(fig,"f4_avaliacao")
def f4_gpi():
    fig,ax=canvas(8,3.6,(0,100),(0,50))
    b=lambda x:40-(24/84)*(x-8); g=lambda x:8+(28/84)*(x-8)
    ax.plot([8,92],[b(8),b(92)],c=IND,lw=3); ax.plot([8,92],[g(8),g(92)],c=GRN,lw=3)
    ax.text(8,b(8)+3,"v = v$_π$   (avaliação)",color=IND,fontsize=10,fontweight="bold"); ax.text(8,g(8)-8,"π = gulosa(v)   (melhoria)",color="#047857",fontsize=10,fontweight="bold")
    xs=[12,24,34,43,50,55,58.5]; path=[(xs[0],g(xs[0]))]
    for k in range(len(xs)-1):
        path.append((xs[k],b(xs[k]))) if k%2==0 else path.append((xs[k],g(xs[k])))
        path.append((xs[k+1],b(xs[k+1]) if k%2==0 else g(xs[k+1])))
    path=[(12,g(12)),(14,b(14)),(26,g(26)),(28,b(28)),(38,g(38)),(40,b(40)),(48,g(48)),(50,b(50)),(55,g(55)),(56,b(56)),(59,25.4)]
    ax.plot([p[0] for p in path],[p[1] for p in path],c=AMB,lw=2,marker="o",ms=3.5)
    star(ax,59.6,25.3,3.2,c=RED); ax.text(56,31.5,"v*, π*",color=RED,fontsize=11,fontweight="bold")
    ax.text(50,47,"Iteração de política generalizada: duas forças que se empurram até concordarem",ha="center",fontsize=10.5,fontweight="bold")
    save(fig,"f4_gpi")
def f4_frozenlake():
    env=gym.make("FrozenLake-v1",is_slippery=True); Pf=env.unwrapped.P; nS,nA=16,4
    Pm=np.zeros((nS,nA,nS)); Rm=np.zeros((nS,nA))
    for s in range(nS):
        for a in range(nA):
            for pr,s2,r,d in Pf[s][a]: Pm[s,a,s2]+=pr; Rm[s,a]+=pr*r
    V,pol,k=iteracao_valor(Pm,Rm,0.99,1e-10)
    mapa="SFFFFHFHFFFHHFFG"; fig,ax=plt.subplots(figsize=(4.6,4.6)); ax.set_xlim(0,4); ax.set_ylim(0,4); ax.axis("off")
    cm=LinearSegmentedColormap.from_list("v",["#EEF2FF",CYA,GRN,"#FDE68A"]); setas={0:(-0.3,0),1:(0,-0.3),2:(0.3,0),3:(0,0.3)}
    for s in range(16):
        i,j=divmod(s,4); y=3-i; c=mapa[s]
        col="#1E3A8A" if c=="H" else cm(V[s]/V.max())
        ax.add_patch(Rectangle((j,y),0.96,0.96,fc=col,ec="white"))
        if c=="H": ax.text(j+0.48,y+0.48,"buraco",ha="center",va="center",color="white",fontsize=8,fontweight="bold")
        elif c=="G": ax.text(j+0.48,y+0.48,"★",ha="center",va="center",fontsize=18,color="#B45309")
        else:
            ax.text(j+0.48,y+0.17,f"{V[s]:.2f}",ha="center",va="center",fontsize=8)
            dx,dy=setas[int(pol[s].argmax())]; ax.annotate("",xy=(j+0.48+dx,y+0.6+dy),xytext=(j+0.48,y+0.6),arrowprops=dict(arrowstyle="-|>",color=INK,lw=2))
    ax.set_title(f"Valores ótimos e política ótima (iteração de valor, {k} varreduras)",fontsize=9.5)
    save(fig,"f4_frozenlake")
def f4_convergencia():
    P,R,term=grade(4); hv=[]; avaliacao(P,R,np.full((16,4),0.25),1.0,1e-10,hist=hv)
    h=[np.abs(hv[k]-hv[k-1]).max() for k in range(1,len(hv))]
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.5))
    axs[0].semilogy(range(1,len(h)+1),h,c=IND,lw=2.2); axs[0].set_xlabel("varredura"); axs[0].set_ylabel("maior mudança em v (escala log)"); axs[0].set_title("Gridworld: avaliação da política aleatória")
    env=gym.make("FrozenLake-v1",is_slippery=True); Pf=env.unwrapped.P; Pm=np.zeros((16,4,16)); Rm=np.zeros((16,4))
    for s in range(16):
        for a in range(4):
            for pr,s2,r,d in Pf[s][a]: Pm[s,a,s2]+=pr; Rm[s,a]+=pr*r
    for g,c in [(0.9,AMB),(0.99,IND),(0.999,GRN)]:
        hh=[]; iteracao_valor(Pm,Rm,g,1e-10,hist=hh); axs[1].semilogy(range(1,len(hh)+1),hh,c=c,lw=2,label=f"γ = {g}")
    axs[1].set_xlabel("varredura"); axs[1].set_title("FrozenLake: γ maior, convergência mais lenta"); axs[1].legend(frameon=False)
    save(fig,"f4_convergencia")
def f4_maldicao():
    fig,ax=plt.subplots(figsize=(7.5,3.6)); d=np.arange(1,13)
    for b,c in [(4,CYA),(10,IND),(20,RED)]: ax.semilogy(d,float(b)**d,c=c,lw=2.2,marker="o",ms=4,label=f"{b} faixas por variável")
    ax.axhline(1e9,c=GRY,ls="--"); ax.text(1.2,1.6e9,"1 bilhão de estados",color=GRY,fontsize=9)
    ax.set_xlabel("número de variáveis do estado (dimensões)"); ax.set_ylabel("nº de estados (escala log)"); ax.set_title("A maldição da dimensionalidade: estados crescem exponencialmente"); ax.legend(frameon=False)
    save(fig,"f4_maldicao")
if __name__=="__main__":
    from figs_cover import star
    for f in [f4_avaliacao,f4_gpi,f4_frozenlake,f4_convergencia,f4_maldicao]: f()
