from fc import *
from matplotlib.colors import LinearSegmentedColormap
from figs_cover import star
import gymnasium as gym, time
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor

def f7_generalizacao():
    rng=np.random.default_rng(2); xs=np.linspace(0,1,400); v=lambda x:np.sin(2.5*np.pi*x)*0.8*np.exp(-x)+0.4*x
    xt=np.sort(rng.uniform(0,1,40)); yt=v(xt)+rng.normal(0,0.12,40)
    fig,axs=plt.subplots(1,4,figsize=(11,2.9),sharey=True)
    Xg=xs.reshape(-1,1); Xt=xt.reshape(-1,1)
    ag=KBinsDiscretizer(n_bins=6,encode="ordinal",strategy="uniform").fit(Xt); idx=ag.transform(Xt).ravel().astype(int); med=np.array([yt[idx==k].mean() for k in range(6)]); pa=med[ag.transform(Xg).ravel().astype(int)]
    modelos=[("Tabela (agregação\nem 6 faixas)",pa,RED),("Reta\n(linear)",LinearRegression().fit(Xt,yt).predict(Xg),AMB),
             ("Polinômio\n(grau 5)",make_pipeline(PolynomialFeatures(5),Ridge(alpha=1e-3)).fit(Xt,yt).predict(Xg),GRN),
             ("Funções RBF\n(random features)",make_pipeline(RBFSampler(gamma=20,n_components=60,random_state=0),Ridge(alpha=1e-2)).fit(Xt,yt).predict(Xg),IND)]
    for ax,(t,p,c) in zip(axs,modelos):
        ax.plot(xs,v(xs),c=GRY,lw=1.2,ls="--"); ax.scatter(xt,yt,s=10,c=INK,alpha=0.6); ax.plot(xs,p,c=c,lw=2.4); ax.set_title(t,fontsize=9,color=c if c!=AMB else "#B45309"); ax.set_xlabel("estado s")
    axs[0].set_ylabel("valor v(s)"); fig.suptitle("Aproximar a função valor com poucos pontos: quanto mais flexível e suave o modelo, melhor generaliza",fontsize=10,fontweight="bold",y=1.04)
    save(fig,"f7_generalizacao")
def f7_gradiente():
    fig,ax=plt.subplots(figsize=(6.2,4.2)); a,b=np.meshgrid(np.linspace(-2,3,200),np.linspace(-2,3,200)); J=lambda w1,w2:0.5*(w1-1)**2+2*(w2-0.5)**2+0.3*(w1-1)*(w2-0.5)
    cs=ax.contourf(a,b,J(a,b),levels=14,cmap=LinearSegmentedColormap.from_list("c",["#EEF2FF","#818CF8","#312E81"])); ax.contour(a,b,J(a,b),levels=14,colors="white",linewidths=0.4)
    w=np.array([2.7,2.6]); path=[w.copy()]
    for _ in range(40):
        g=np.array([(w[0]-1)+0.3*(w[1]-0.5)*0.5*2/2*1,4*(w[1]-0.5)+0.3*(w[0]-1)*0.5*2/2*1]); g=np.array([ (w[0]-1)+0.3*(w[1]-0.5), 4*(w[1]-0.5)+0.3*(w[0]-1)]); w=w-0.17*g; path.append(w.copy())
    P=np.array(path); ax.plot(P[:,0],P[:,1],c=AMB,lw=2,marker="o",ms=3.5); ax.scatter([1],[0.5],marker="*",s=260,c="#FDE68A",edgecolors="white",zorder=5)
    ax.text(-1.8,-0.2,"estrela: mínimo do erro",color="white",fontsize=9,fontweight="bold"); ax.text(2.2,2.8,"início",color="white",fontsize=9,fontweight="bold")
    ax.set_xlabel("peso w₁"); ax.set_ylabel("peso w₂"); ax.set_title("Descida do gradiente: um passo de cada vez, ladeira abaixo"); save(fig,"f7_gradiente")
def f7_tiles():
    fig,ax=plt.subplots(figsize=(6.2,4.6)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    cores=[IND,GRN,AMB]; ponto=(5.3,4.6); w=2.5
    for t,(c,off) in enumerate(zip(cores,[(0,0),(0.8,0.5),(1.6,1.0)])):
        for i in range(-1,5):
            for j in range(-1,5):
                x0,y0=i*w-off[0]*0+off[0],j*w-off[1]*0+off[1]
                ax.add_patch(Rectangle((x0-w+ (0),y0-w),w,w,fc="none",ec=c,lw=0.7,alpha=0.45))
        # tile ativo
        xi=np.floor((ponto[0]-off[0]+w)/w)*w-w+off[0]; yi=np.floor((ponto[1]-off[1]+w)/w)*w-w+off[1]
        ax.add_patch(Rectangle((xi,yi),w,w,fc=c,ec=c,lw=2,alpha=0.3))
    ax.scatter(*ponto,s=140,c=INK,zorder=5,marker="X"); ax.text(ponto[0]+0.35,ponto[1]-1.2,"estado\natual",fontsize=9,fontweight="bold")
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_title("Tile coding: 3 grades deslocadas; cada uma acende 1 quadrado",fontsize=10)
    ax.set_xlabel("variável 1 (ex.: posição)"); ax.set_ylabel("variável 2 (ex.: velocidade)"); save(fig,"f7_tiles")
def mc_treinar(semente=0,n_ep=500,N_T=8,N_B=8):
    env=gym.make("MountainCar-v0"); rng=np.random.default_rng(semente); low=np.array([-1.2,-0.07]); high=np.array([0.6,0.07])
    def feats(o):
        z=(o-low)/(high-low)*N_B; idx=[]
        for t in range(N_T):
            off=t/N_T; c=np.floor(z+off*np.array([1,3])).astype(int).clip(0,N_B); idx.append(t*(N_B+1)**2+c[0]*(N_B+1)+c[1])
        return idx
    W=np.zeros((3,N_T*(N_B+1)**2)); alpha=0.5/N_T; passos=[]; q=lambda idx,a:W[a,idx].sum()
    for ep in range(n_ep):
        o,_=env.reset(seed=int(rng.integers(1e9))); idx=feats(o); a=int(np.argmax([q(idx,b) for b in range(3)])); n=0; fim=False
        while not fim:
            o2,r,te,tr,_=env.step(a); fim=te or tr; n+=1; idx2=feats(o2)
            if te: W[a,idx]+=alpha*(r-q(idx,a)); break
            a2=int(np.argmax([q(idx2,b) for b in range(3)])); W[a,idx]+=alpha*(r+q(idx2,a2)-q(idx,a)); idx,a=idx2,a2
        passos.append(n)
    return W,feats,np.array(passos)
def f7_mountaincar():
    res=[mc_treinar(s) for s in range(3)]; P=np.array([r[2] for r in res]); W,feats,_=res[0]
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.7)); x=np.arange(P.shape[1])
    axs[0].plot(x,P.mean(0),c=IND,lw=1.5,alpha=0.5); axs[0].plot(np.convolve(P.mean(0),np.ones(10)/10,mode="valid"),c=IND,lw=2.5); axs[0].set_yscale("log"); axs[0].set_xlabel("episódio"); axs[0].set_ylabel("passos até a meta (escala log)"); axs[0].set_title("Semi-gradiente SARSA com tile coding")
    pos=np.linspace(-1.2,0.6,60); vel=np.linspace(-0.07,0.07,60); Z=np.zeros((60,60))
    for i,p in enumerate(pos):
        for j,v in enumerate(vel): idx=feats(np.array([p,v])); Z[j,i]=-max(W[a,idx].sum() for a in range(3))
    im=axs[1].imshow(Z,origin="lower",extent=[-1.2,0.6,-0.07,0.07],aspect="auto",cmap=LinearSegmentedColormap.from_list("c",["#FDE68A",CYA,IND,"#1E1B4B"])); axs[1].set_xlabel("posição"); axs[1].set_ylabel("velocidade"); axs[1].set_title("Custo estimado até a meta (−max q)"); fig.colorbar(im,ax=axs[1],label="passos")
    save(fig,"f7_mountaincar")
def f7_triangulo():
    fig,ax=canvas(8,4.6,(0,100),(0,70)); cs=[(32,40,IND,"Aproximação\nde funções"),(68,40,AMB,"Bootstrapping"),(50,18,GRN,"Off-policy")]
    pos=[(34,42),(66,42),(50,22)]
    for (x,y),(t,c) in zip(pos,[("Aproximação\nde funções",IND),("Bootstrapping",AMB),("Aprendizado\noff-policy",GRN)]):
        ax.scatter([x],[y],s=24000,c=c,alpha=0.28,edgecolors=c,linewidths=2)
    ax.text(24,50,"Aproximação\nde funções",ha="center",fontsize=10.5,fontweight="bold",color=IND); ax.text(76,50,"Bootstrapping",ha="center",fontsize=10.5,fontweight="bold",color="#B45309"); ax.text(50,9,"Aprendizado off-policy",ha="center",fontsize=10.5,fontweight="bold",color="#047857")
    star(ax,50,34,4.5,c=RED); ax.text(50,43,"INSTABILIDADE\ne divergência",ha="center",fontsize=9.5,fontweight="bold",color=RED)
    ax.text(50,66,"O triângulo mortal (deadly triad)",ha="center",fontsize=12,fontweight="bold")
    ax.text(50,1.5,"Com dois dos três, o aprendizado costuma ser estável; com os três juntos, pode divergir.",ha="center",fontsize=8.8,color=GRY,style="italic")
    save(fig,"f7_triangulo")
def baird(passos=3000,alpha=0.01,gamma=0.99,semente=0):
    rng=np.random.default_rng(semente); X=np.zeros((7,8))
    for i in range(6): X[i,i]=2; X[i,7]=1
    X[6,6]=1; X[6,7]=2
    w=np.array([1,1,1,1,1,1,10,1],float); s=int(rng.integers(7)); hist=[w.copy()]
    for t in range(passos):
        if rng.random()<6/7: s2=int(rng.integers(6)); rho=0.0
        else: s2=6; rho=7.0
        delta=gamma*X[s2]@w-X[s]@w; w+=alpha*rho*delta*X[s]; s=s2; hist.append(w.copy())
    return np.array(hist)
def f7_baird():
    h=baird(); fig,axs=plt.subplots(1,2,figsize=(9.5,3.6)); ax=axs[0]; ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off"); ax.set_title("Contraexemplo de Baird: 7 estados")
    xs=np.linspace(0.8,9.2,6)
    for i,x in enumerate(xs): dot(ax,x,4.4,s=450,c=IND); ax.text(x,4.4,str(i+1),ha="center",va="center",color="white",fontweight="bold",fontsize=9)
    dot(ax,5,1.9,s=700,c=AMB); ax.text(5,1.9,"7",ha="center",va="center",color="white",fontweight="bold")
    for x in xs: ax.annotate("",xy=(x,3.9),xytext=(5,2.35),arrowprops=dict(arrowstyle="->",color=GRY,lw=0.8,ls="--"))
    ax.text(8.6,2.2,"tracejada: vai a um dos\nestados 1–6 (6/7)\nsólida: vai a 7 (1/7)",fontsize=8,color=GRY,va="center"); ax.text(0.1,0.3,"política-alvo: sempre a sólida\nr = 0 em todo lugar ⇒ v = 0",fontsize=8.5,color=INK,fontweight="bold")
    ax=axs[1]; cm=plt.cm.viridis(np.linspace(0,1,8))
    for k in range(8): ax.plot(h[:,k],c=cm[k],lw=1.4,label=f"w{k+1}")
    ax.set_yscale("symlog"); ax.set_xlabel("passos de atualização"); ax.set_ylabel("pesos (escala log simétrica)"); ax.set_title("Os pesos divergem para o infinito"); ax.legend(frameon=False,fontsize=7,ncol=4)
    save(fig,"f7_baird")
def f7_fqi():
    env=gym.make("CartPole-v1"); rng=np.random.default_rng(0)
    O=[];A=[];R=[];O2=[];T=[]; o,_=env.reset(seed=0)
    for _ in range(20000):
        a=int(rng.integers(2)); o2,r,te,tr,_=env.step(a); O.append(o);A.append(a);R.append(r);O2.append(o2);T.append(te); o=o2
        if te or tr: o,_=env.reset()
    O,A,R,O2,T=map(np.array,(O,A,R,O2,T)); X=np.column_stack([O,A])
    def q_max(m):
        n=len(O2); Xb=np.vstack([np.column_stack([O2,np.zeros(n)]),np.column_stack([O2,np.ones(n)])]); q=m.predict(Xb); return np.maximum(q[:n],q[n:])
    def aval(m,n=10):
        out=[]
        for ep in range(n):
            o,_=env.reset(seed=3000+ep); tot=0; fim=False
            while not fim:
                q=m.predict(np.array([[*o,0],[*o,1]])); o,r,te,tr,_=env.step(int(q.argmax())); tot+=r; fim=te or tr
            out.append(tot)
        return np.mean(out)
    curvas={}
    for nome,fab in [("ExtraTrees",lambda:ExtraTreesRegressor(n_estimators=30,min_samples_leaf=5,random_state=0)),("RBF + Ridge",lambda:make_pipeline(StandardScaler(),RBFSampler(gamma=0.3,n_components=400,random_state=0),Ridge(alpha=1.0)))]:
        m=None; pts=[]
        for it in range(1,21):
            y=R.astype(float) if m is None else R+0.99*q_max(m)*(~T); m=fab().fit(X,y)
            if it in (1,2,3,5,8,12,16,20): pts.append((it,aval(m)))
        curvas[nome]=pts
    fig,ax=plt.subplots(figsize=(7.5,3.5))
    for (nome,pts),c in zip(curvas.items(),[GRN,IND]): x,y=zip(*pts); ax.plot(x,y,c=c,lw=2.4,marker="o",label=nome)
    ax.set_xlabel("iteração do Fitted Q-Iteration"); ax.set_ylabel("retorno médio (10 episódios)"); ax.set_title("FQI no CartPole com 20.000 transições aleatórias"); ax.legend(frameon=False); save(fig,"f7_fqi")
def f7_instavel():
    env=gym.make("CartPole-v1"); rng=np.random.default_rng(0); amostra=[]; o,_=env.reset(seed=0)
    for _ in range(5000):
        o,r,te,tr,_=env.step(int(rng.integers(2))); amostra.append(o)
        if te or tr: o,_=env.reset()
    amostra=np.array(amostra); esc=StandardScaler().fit(amostra); rbf=RBFSampler(gamma=0.5,n_components=300,random_state=0).fit(esc.transform(amostra)); f=lambda o:rbf.transform(esc.transform(np.atleast_2d(o)))
    modelos=[SGDRegressor(learning_rate="constant",eta0=0.005,alpha=1e-6,random_state=0) for _ in range(2)]
    for m in modelos: m.partial_fit(f(amostra[0]),[10.0])
    ret=[]
    for ep in range(400):
        eps=max(0.02,0.5*(1-ep/300)); o,_=env.reset(seed=int(rng.integers(1e9))); x=f(o); fim=False; tot=0
        while not fim:
            a=int(rng.integers(2)) if rng.random()<eps else int(np.argmax([m.predict(x)[0] for m in modelos]))
            o2,r,te,tr,_=env.step(a); fim=te or tr; tot+=r; x2=f(o2); alvo=r if te else r+0.99*max(m.predict(x2)[0] for m in modelos); modelos[a].partial_fit(x,[alvo]); x=x2
        ret.append(tot)
    fig,ax=plt.subplots(figsize=(7.5,3.2)); ax.plot(ret,c=RED,lw=0.8,alpha=0.5); ax.plot(np.convolve(ret,np.ones(20)/20,mode="valid"),c=RED,lw=2.4); ax.set_xlabel("episódio"); ax.set_ylabel("retorno"); ax.set_title("Q-learning online com RBF + SGD no CartPole: o aprendizado não decola"); ax.set_ylim(0,60)
    save(fig,"f7_instavel")
if __name__=="__main__":
    import sys
    fs={f.__name__[3:]:f for f in [f7_generalizacao,f7_gradiente,f7_tiles,f7_mountaincar,f7_triangulo,f7_baird,f7_fqi,f7_instavel]}
    for k in (sys.argv[1:] or fs): fs[k]()
