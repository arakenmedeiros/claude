from fc import *
import json, os
from scipy.stats import norm, gaussian_kde
from bo_lib import *
CACHE5="/home/user/claude/automl/build/f5_cache.json"

def f5_garimpo():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"Garimpo inteligente: cada furo custa caro",ha="center",fontsize=11,fontweight="bold")
    rng=np.random.default_rng(8)
    rbox(ax,2,18,60,70,"",fc="#FEF3C7",ec=AMB,lw=2,r=2)
    xs=np.linspace(2,62,60); ys=np.linspace(18,88,40); XX,YY=np.meshgrid(xs,ys); Z=np.exp(-(((XX-44)/9)**2+((YY-62)/10)**2))+0.5*np.exp(-(((XX-18)/12)**2+((YY-40)/9)**2))
    ax.contourf(XX,YY,Z,levels=8,cmap="YlOrBr",alpha=0.8)
    P=np.array([(12,30),(30,75),(52,32),(20,52),(36,45),(44,58),(46,66),(41,63)])
    for i,(x,y) in enumerate(P):
        ax.scatter([x],[y],s=60,c=INK if i<4 else GRN,zorder=3,edgecolors="white"); ax.text(x+1.5,y+2,str(i+1),fontsize=8,fontweight="bold",color=INK if i<4 else "#047857")
    ax.scatter([45],[63],s=260,marker="*",c=RED,zorder=4,edgecolors="white")
    ax.text(32,12.5,"Pretos (1 a 4): furos para conhecer o terreno",ha="center",fontsize=8,color=INK); ax.text(32,6.5,"Verdes (5 a 8): furos guiados pelo que já se descobriu",ha="center",fontsize=8,color="#047857")
    rbox(ax,66,50,32,38,"Explorar\nvisitar regiões\nainda pouco\nconhecidas",fc="#EEF2FF",ec=IND,fs=8.8,tc=INK,lw=1.6)
    rbox(ax,66,8,32,38,"Aproveitar\ncavar perto do\nque já parece\nrico",fc="#ECFDF5",ec=GRN,fs=8.8,tc=INK,lw=1.6)
    ax.annotate("",xy=(82,47.5),xytext=(82,49.5),arrowprops=dict(arrowstyle="<->",color=AMB,lw=2))
    save(fig,"f5_garimpo")

def f5_gp():
    rng=np.random.default_rng(3); X=list(rng.uniform(0,10,3)); Y=[f(x) for x in X]
    marks=(3,5,8,12); fig,axs=plt.subplots(2,4,figsize=(8.6,4.3),gridspec_kw={"height_ratios":[3,1.2]},sharex=True)
    xx=GRID[:,0]
    while len(X)<=max(marks):
        n=len(X)
        if n in marks:
            gp=ajustar(X,Y,0); mu,sd,ei,pi,ucb=aquisicoes(gp,Y); c=marks.index(n)
            a=axs[0,c]; a.plot(xx,f(xx),"--",c=GRY,lw=1.2); a.plot(xx,mu,c=IND,lw=2); a.fill_between(xx,mu-2*sd,mu+2*sd,color=IND,alpha=0.18)
            a.scatter(X,Y,c=INK,s=22,zorder=4); a.set_ylim(-0.6,1.6); a.set_yticks([]); a.set_title(f"{n} avaliações",fontsize=9,fontweight="bold")
            b=axs[1,c]; b.fill_between(xx,0,ei/ max(ei.max(),1e-12),color=GRN,alpha=0.5); nx=xx[np.argmax(ei)]; b.axvline(nx,c=RED,lw=1.6); b.set_yticks([]); b.set_ylim(0,1.15)
            a.axvline(nx,c=RED,lw=1,ls=":")
            if c==0: a.set_ylabel("função e modelo",fontsize=8); b.set_ylabel("aquisição",fontsize=8)
        gp=ajustar(X,Y,0); mu,sd,ei,pi,ucb=aquisicoes(gp,Y); x=float(xx[np.argmax(ei)]); X.append(x); Y.append(f(x))
    axs[0,0].plot([],[],"--",c=GRY,label="função real (desconhecida)"); axs[0,0].plot([],[],c=IND,label="modelo (média)"); axs[0,0].fill_between([],[],[],color=IND,alpha=0.18,label="incerteza (±2 desvios)")
    fig.legend(loc="lower center",ncol=3,fontsize=7.8,frameon=False,bbox_to_anchor=(0.5,-0.03))
    fig.suptitle("A otimização bayesiana em ação (cálculo real com processo gaussiano)",fontsize=10.5,fontweight="bold",y=1.0)
    fig.tight_layout(rect=(0,0.03,1,0.97)); save(fig,"f5_gp")

def f5_aquisicao():
    rng=np.random.default_rng(3); X=list(rng.uniform(0,10,3))+[3.2,5.0,6.2]; Y=[f(x) for x in X]
    gp=ajustar(X,Y,0); mu,sd,ei,pi,ucb=aquisicoes(gp,Y); xx=GRID[:,0]
    fig,axs=plt.subplots(1,4,figsize=(8.6,2.9))
    a=axs[0]; a.plot(xx,mu,c=IND,lw=2); a.fill_between(xx,mu-2*sd,mu+2*sd,color=IND,alpha=0.18); a.scatter(X,Y,c=INK,s=22,zorder=4); a.axhline(max(Y),c=GRY,ls="--",lw=1); a.set_title("o modelo",fontsize=9,fontweight="bold"); a.set_yticks([]); a.set_ylim(-0.6,1.5)
    for a,(v,t,c,d) in zip(axs[1:],[(pi,"PI: chance de melhorar",CYA,"só vê se melhora"),(ei,"EI: melhora esperada",GRN,"pesa o tamanho"),(ucb,"UCB: média + k·incerteza",AMB,"otimismo calibrado")]):
        a.plot(xx,v,c=c,lw=2); nx=xx[np.argmax(v)]; a.axvline(nx,c=RED,lw=1.5); a.set_title(t,fontsize=8.3,fontweight="bold",color=c); a.set_yticks([]); a.set_xlabel(f"próximo: x ≈ {nx:.1f}",fontsize=8)
    axs[0].set_xlabel("pontos já avaliados",fontsize=8)
    fig.suptitle("Funções de aquisição: três jeitos de decidir onde olhar a seguir",fontsize=10,fontweight="bold",y=1.03)
    fig.tight_layout(); save(fig,"f5_aquisicao")

def f5_loop():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"O ciclo da otimização bayesiana",ha="center",fontsize=11,fontweight="bold")
    B=[(8,62,"1. Treinar o modelo\nsubstituto com o\nhistórico de tentativas",IND),(38,62,"2. Calcular a função\nde aquisição\n(onde vale olhar?)",GRN),(68,62,"3. Escolher o ponto\nde maior\naquisição",AMB)]
    for x,y,t,c in B: rbox(ax,x,y,26,26,t,fc=c,ec=c,fs=8.6,tc="white",bold=True,r=2)
    arrow(ax,(34.5,75),(37.5,75),lw=2,ms=12); arrow(ax,(64.5,75),(67.5,75),lw=2,ms=12)
    rbox(ax,68,18,26,26,"4. Avaliar de verdade\n(treinar o modelo\ne validar)",fc=RED,ec=RED,fs=8.6,tc="white",bold=True,r=2)
    rbox(ax,8,18,26,26,"5. Acrescentar o\nresultado ao\nhistórico",fc=VIO,ec=VIO,fs=8.6,tc="white",bold=True,r=2)
    arrow(ax,(81,61.5),(81,44.5),lw=2,ms=12); arrow(ax,(67.5,31),(34.5,31),lw=2,ms=12); arrow(ax,(21,44.5),(21,61.5),lw=2,ms=12)
    ax.text(50,8,"O passo 4 é o caro. O modelo substituto é barato e ajuda a gastar poucas avaliações caras.",ha="center",fontsize=9,color=INK,style="italic")
    ax.text(52,40,"repete até acabar o orçamento",ha="center",fontsize=8.5,color=GRY)
    save(fig,"f5_loop")

def f5_tpe():
    rng=np.random.default_rng(1); x=rng.uniform(0,10,60); y=f(x)+rng.normal(0,0.03,60)
    thr=np.quantile(y,0.75); good=x[y>=thr]; bad=x[y<thr]; xx=np.linspace(0,10,400)
    l=gaussian_kde(good,bw_method=0.5)(xx); g=gaussian_kde(bad,bw_method=0.35)(xx)
    fig,axs=plt.subplots(1,2,figsize=(8.6,3.4))
    a=axs[0]; a.scatter(bad,[0.02]*len(bad),c=GRY,s=14,label="tentativas ruins (75%)"); a.scatter(good,[0.05]*len(good),c=GRN,s=22,label="tentativas boas (25%)")
    a.plot(xx,l,c=GRN,lw=2.2,label="l(x): onde as boas caem"); a.plot(xx,g,c=RED,lw=2.2,label="g(x): onde as ruins caem"); a.set_yticks([]); a.set_ylim(0,max(l.max(),g.max())*1.75); a.legend(fontsize=7.3,frameon=False,loc="upper center",ncol=2); a.set_title("Duas densidades",fontsize=9.5,fontweight="bold"); a.set_xlabel("valor do hiperparâmetro")
    b=axs[1]; r=l/np.maximum(g,1e-6); b.plot(xx,r,c=IND,lw=2.2); nx=xx[np.argmax(r)]; b.axvline(nx,c=RED,lw=1.5); b.set_yticks([]); b.set_title("Razão l(x)/g(x): maior = mais promissor",fontsize=9.5,fontweight="bold"); b.set_xlabel(f"próxima tentativa: x ≈ {nx:.1f}")
    fig.suptitle("TPE: separe o histórico em \"bons\" e \"ruins\" e sorteie onde os bons predominam",fontsize=10,fontweight="bold",y=1.03)
    fig.tight_layout(); save(fig,"f5_tpe")

def comparacao_dados():
    if os.path.exists(CACHE5): return json.load(open(CACHE5))
    import optuna
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X,y=load_digits(return_X_y=True); X=X/16.0; cv=StratifiedKFold(3,shuffle=True,random_state=0)
    fn=lambda C,g: cross_val_score(SVC(C=C,gamma=g),X,y,cv=cv).mean()
    N=30; rand=[]; tpe=[]
    for s in range(8):
        rng=np.random.default_rng(s); rand.append(np.maximum.accumulate([fn(10**rng.uniform(-3,4),10**rng.uniform(-8,1)) for _ in range(N)]).tolist())
        st=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=s,n_startup_trials=8))
        st.optimize(lambda tr: fn(tr.suggest_float("C",1e-3,1e4,log=True),tr.suggest_float("g",1e-8,1e1,log=True)),n_trials=N)
        tpe.append(np.maximum.accumulate([t.value for t in st.trials]).tolist())
    B=np.array([bo(s) for s in range(40)]); R=np.array([aleatoria(s) for s in range(40)])
    d={"svc_rand":rand,"svc_tpe":tpe,"toy_bo":B.tolist(),"toy_rand":R.tolist()}; json.dump(d,open(CACHE5,"w")); return d

def f5_comparacao():
    d=comparacao_dados(); fig,axs=plt.subplots(1,2,figsize=(8.6,3.5))
    for a,(ka,kb,la,lb,t,xl) in zip(axs,[("toy_rand","toy_bo","aleatória","bayesiana (GP + EI)","Função-agulha 1D (40 repetições)","avaliações"),("svc_rand","svc_tpe","aleatória","TPE (Optuna)","SVM em Digits (8 repetições)","avaliações")]):
        A=np.array(d[ka]); B=np.array(d[kb]); k=np.arange(1,A.shape[1]+1)
        for M,c,l in [(A,GRY,la),(B,IND,lb)]:
            a.plot(k,M.mean(0),c=c,lw=2.2,label=l); a.fill_between(k,np.percentile(M,25,0),np.percentile(M,75,0),color=c,alpha=0.15)
        a.set_title(t,fontsize=9.3,fontweight="bold"); a.set_xlabel(xl,fontsize=8.5); a.legend(frameon=False,fontsize=8,loc="lower right")
    axs[0].set_ylabel("melhor valor até agora"); axs[1].set_ylim(0.85,1.0)
    fig.suptitle("A otimização bayesiana nem sempre ganha: depende do problema e do orçamento",fontsize=10,fontweight="bold",y=1.02)
    fig.tight_layout(); save(fig,"f5_comparacao")

def f5_optuna():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"Anatomia do Optuna",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,60,28,28,"Estudo\n(Study)\no experimento\ninteiro",fc=IND,ec=IND,fs=9,tc="white",bold=True,r=2)
    rbox(ax,36,60,28,28,"Tentativa\n(Trial)\num teste com\num conjunto\nde valores",fc=CYA,ec=CYA,fs=9,tc="white",bold=True,r=2)
    rbox(ax,70,60,28,28,"Objetivo\n(objective)\nsua função:\ndevolve a nota",fc=GRN,ec=GRN,fs=9,tc="white",bold=True,r=2)
    arrow(ax,(30.5,74),(35.5,74),lw=2,ms=11); arrow(ax,(64.5,74),(69.5,74),lw=2,ms=11)
    rbox(ax,2,14,28,34,"Amostrador\n(Sampler)\nescolhe os valores:\nTPE, aleatório,\nCMA-ES...",fc="#EEF2FF",ec=IND,fs=8.3,tc=INK,lw=1.8,r=2)
    rbox(ax,36,14,28,34,"Podador\n(Pruner)\ninterrompe cedo\ntentativas\nruins",fc="#ECFEFF",ec=CYA,fs=8.3,tc=INK,lw=1.8,r=2)
    rbox(ax,70,14,28,34,"suggest_float\nsuggest_int\nsuggest_categorical\ndescrevem o\nespaço de busca",fc="#ECFDF5",ec=GRN,fs=8.3,tc=INK,lw=1.8,r=2)
    for x in (16,50,84): arrow(ax,(x,48.5),(x,59.5),lw=1.3,ms=9,c=GRY)
    ax.text(50,5,"Define-by-run: o espaço de busca é descrito dentro da própria função, com código comum de Python.",ha="center",fontsize=8.8,color=GRY,style="italic")
    save(fig,"f5_optuna")

if __name__=="__main__":
    for fn_ in (f5_garimpo,f5_gp,f5_aquisicao,f5_loop,f5_tpe,f5_comparacao,f5_optuna): fn_()
