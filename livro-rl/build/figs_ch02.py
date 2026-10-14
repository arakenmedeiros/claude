from fc import *
from figs_cover import star
from scipy.stats import beta as betad

def desempate(M,rng): return (M+1e-9*rng.random(M.shape)).argmax(1)
def executar(est,q,passos=1000,semente=1,**kw):
    rng=np.random.default_rng(semente); n,k=q.shape; idx=np.arange(n); Q=np.zeros((n,k)); N=np.zeros((n,k)); rec=np.zeros((n,passos)); ot=np.zeros((n,passos)); mel=q.argmax(1)
    for t in range(passos):
        a=est(Q,N,t,rng,**kw); r=rng.normal(q[idx,a],1.0); N[idx,a]+=1; Q[idx,a]+=(r-Q[idx,a])/N[idx,a]; rec[:,t]=r; ot[:,t]=(a==mel)
    return rec.mean(0),ot.mean(0)
def gulosa(Q,N,t,rng): return desempate(Q,rng)
def eps_g(Q,N,t,rng,eps=0.1):
    a=desempate(Q,rng); m=rng.random(len(a))<eps; a[m]=rng.integers(Q.shape[1],size=m.sum()); return a
def ucb(Q,N,t,rng,c=2.0):
    b=np.where(N==0,np.inf,c*np.sqrt(np.log(t+1)/np.maximum(N,1))); return desempate(Q+b,rng)

def f2_cassino():
    fig,ax=canvas(10,3.9)
    cols=[IND,GRN,AMB,PNK]
    for i in range(4):
        x=3+i*24; c=cols[i]
        rbox(ax,x,22,17,66,"",fc=c,r=2.5); rbox(ax,x+1.5,52,14,22,"",fc="#111827",r=1.5)
        for j in range(3): ax.text(x+4+j*4.5,63,"?",ha="center",va="center",color="#FDE68A",fontsize=15,fontweight="bold")
        rbox(ax,x+2.5,28,12,10,"",fc="#FDE68A",r=1); ax.text(x+8.5,33,"R$ ?",ha="center",va="center",fontsize=9,fontweight="bold",color=INK)
        ax.plot([x+17.3,x+20.5],[70,80],c=INK,lw=3); dot(ax,x+20.5,80,s=110,c=RED,lw=0.8)
        ax.text(x+8.5,14,f"braço {i+1}",ha="center",fontsize=10.5,fontweight="bold",color=c if c!=AMB else "#B45309")
        ax.text(x+8.5,7,"prêmio médio desconhecido",ha="center",fontsize=7,color=GRY)
    ax.text(50,96,"O problema dos bandits multi-armados: qual máquina vale mais a pena?",ha="center",fontsize=11.5,fontweight="bold",color=INK)
    save(fig,"f2_cassino")

def f2_distribuicoes():
    rng=np.random.default_rng(4); q=rng.normal(0,1,10)
    fig,ax=plt.subplots(figsize=(8.5,3.6))
    D=[rng.normal(m,1,2000) for m in q]
    v=ax.violinplot(D,positions=range(1,11),widths=0.8,showextrema=False)
    for i,b in enumerate(v["bodies"]): b.set_facecolor(AMB if i==q.argmax() else IND); b.set_alpha(0.55)
    ax.scatter(range(1,11),q,c=INK,zorder=3,s=26,label="valor verdadeiro q*(a)"); ax.axhline(0,c=GRY,lw=0.8,ls="--")
    ax.set_xticks(range(1,11)); ax.set_xlabel("braço (ação a)"); ax.set_ylabel("recompensa"); ax.set_title("Um \"banco de testes\" de 10 braços: cada braço paga com ruído em torno de uma média")
    ax.annotate("melhor braço\n(o agente não sabe!)",xy=(q.argmax()+1,q.max()),xytext=(q.argmax()+2.2,q.max()+1.7),arrowprops=dict(arrowstyle="->",color=AMB),color="#B45309",fontsize=9,fontweight="bold")
    ax.legend(frameon=False,loc="lower left"); save(fig,"f2_distribuicoes")

def f2_incremental():
    fig,ax=canvas(10,3.0,(0,100),(0,50))
    ax.plot([5,95],[22,22],c=INK,lw=2)
    Q,R=30,80; Qn=Q+0.3*(R-Q)
    for x,lab,c in [(Q,"estimativa\nvelha Q",IND),(R,"alvo\n(recompensa R)",AMB),(Qn,"estimativa\nnova",GRN)]:
        dot(ax,x,22,s=260,c=c); ax.text(x,10,lab,ha="center",va="center",fontsize=9.5,fontweight="bold",color=c if c!=AMB else "#B45309")
    arrow(ax,(Q+2,30),(R-2,30),c=GRY,lw=1.5); ax.text((Q+R)/2,34,"erro  =  R − Q",ha="center",fontsize=10,color=GRY,fontweight="bold")
    arrow(ax,(Q+2,25.5),(Qn-1,25.5),c=GRN,lw=3); ax.text(Qn-5,41,"passo pequeno: α × erro",ha="center",fontsize=10,color="#047857",fontweight="bold")
    ax.text(50,48,"Q  ←  Q + α · (R − Q)      “novo = velho + passo × (alvo − velho)”",ha="center",fontsize=11,fontweight="bold",color=IND)
    save(fig,"f2_incremental")

def f2_estrategias():
    q=np.random.default_rng(0).normal(0,1,(1000,10)); fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for nome,f,kw,c in [("gulosa (ε=0)",gulosa,{},RED),("ε=0,01",eps_g,{"eps":0.01},GRN),("ε=0,1",eps_g,{"eps":0.1},IND),("UCB (c=2)",ucb,{"c":2.0},AMB)]:
        r,o=executar(f,q,**kw); axs[0].plot(r,c=c,lw=1.8,label=nome); axs[1].plot(100*o,c=c,lw=1.8,label=nome)
    axs[0].set_xlabel("passo"); axs[0].set_ylabel("recompensa média"); axs[0].set_title("Recompensa média (1000 execuções)"); axs[0].legend(frameon=False,fontsize=8.5)
    axs[1].set_xlabel("passo"); axs[1].set_ylabel("% de escolhas do melhor braço"); axs[1].set_title("Frequência da ação ótima")
    save(fig,"f2_estrategias")

def f2_ucb():
    fig,ax=plt.subplots(figsize=(8,3.5)); Q=np.array([0.55,0.50,0.30,0.45]); N=np.array([60,50,5,12]); t=sum(N); b=1.0*np.sqrt(2*np.log(t)/N)
    x=np.arange(4); ax.bar(x,Q,color=IND,alpha=0.85,label="estimativa Q (o que já sei)"); ax.bar(x,b,bottom=Q,color=AMB,alpha=0.6,label="bônus de incerteza (o que ainda não sei)")
    for i in range(4): ax.text(i,0.06,f"n = {N[i]}",ha="center",fontsize=9,fontweight="bold",color="white")
    ax.scatter([2],[Q[2]+b[2]],marker="*",s=380,c=RED,zorder=5,edgecolors="white"); ax.text(2.0,Q[2]+b[2]+0.22,"UCB escolhe este!",ha="center",color=RED,fontweight="bold",fontsize=9.5)
    ax.set_ylim(0,2.2); ax.set_xticks(x); ax.set_xticklabels([f"braço {i+1}" for i in x]); ax.set_ylabel("limite superior de confiança"); ax.set_title("UCB: otimismo diante da incerteza"); ax.legend(frameon=False,loc="upper left",fontsize=8.5)
    save(fig,"f2_ucb")

def f2_thompson():
    rng=np.random.default_rng(2); p=[0.45,0.55,0.60]; S=np.zeros(3); N=np.zeros(3); fig,axs=plt.subplots(1,3,figsize=(10.5,3.1),sharey=False); xs=np.linspace(0,1,300); marcos=[0,30,300]; t=0
    for ax,m in zip(axs,marcos):
        while t<m:
            a=int(np.argmax(rng.beta(S+1,N-S+1))); r=float(rng.random()<p[a]); N[a]+=1; S[a]+=r; t+=1
        for i,c in enumerate([RED,IND,GRN]): ax.plot(xs,betad.pdf(xs,S[i]+1,N[i]-S[i]+1),c=c,lw=2,label=f"braço {i+1} (p={p[i]})"); ax.fill_between(xs,betad.pdf(xs,S[i]+1,N[i]-S[i]+1),color=c,alpha=0.08)
        ax.set_title(f"após {m} jogadas"); ax.set_xlabel("probabilidade de sucesso"); ax.set_yticks([])
    axs[2].legend(frameon=False,fontsize=7.5,loc="upper left"); axs[0].set_ylabel("crença (densidade)")
    save(fig,"f2_thompson")

def bern(est,p,passos=2000,n=300,semente=3):
    rng=np.random.default_rng(semente); p=np.asarray(p); k=len(p); idx=np.arange(n); S=np.zeros((n,k)); N=np.zeros((n,k)); ar=np.zeros((n,passos))
    for t in range(passos):
        a=est(S,N,t,rng); r=(rng.random(n)<p[a]).astype(float); N[idx,a]+=1; S[idx,a]+=r; ar[:,t]=p.max()-p[a]
    return np.cumsum(ar.mean(0))
def b_eps(S,N,t,rng,eps=0.1):
    Q=np.where(N>0,S/np.maximum(N,1),0.0); a=desempate(Q,rng); m=rng.random(len(a))<eps; a[m]=rng.integers(Q.shape[1],size=m.sum()); return a
def b_ucb(S,N,t,rng):
    Q=np.where(N>0,S/np.maximum(N,1),0.0); b=np.where(N==0,np.inf,np.sqrt(2*np.log(t+1)/np.maximum(N,1))); return desempate(Q+b,rng)
def b_th(S,N,t,rng): return rng.beta(S+1,N-S+1).argmax(1)
def f2_arrependimento():
    p=[0.45,0.55,0.60]; fig,ax=plt.subplots(figsize=(8,3.7))
    for nome,f,c in [("ε-gulosa (ε=0,1)",b_eps,IND),("UCB1",b_ucb,AMB),("Thompson",b_th,GRN),("gulosa pura",lambda S,N,t,r:b_eps(S,N,t,r,eps=0.0),RED)]:
        ax.plot(bern(f,p),c=c,lw=2,label=nome)
    ax.set_xlabel("passo"); ax.set_ylabel("arrependimento acumulado"); ax.set_title("Quanto cada estratégia \"perde\" por não saber o melhor braço"); ax.legend(frameon=False)
    save(fig,"f2_arrependimento")

def f2_contextual():
    fig,ax=canvas(10,3.4)
    steps=[("1. Contexto","chega um cliente:\nidade, profissão,\nhistórico",CYA),("2. Decisão","o agente escolhe\numa oferta (braço)",IND),("3. Recompensa","o cliente aceita?\n(1 ou 0)",AMB),("4. Aprendizado","atualiza o modelo\nque prevê a\nrecompensa",GRN)]
    for k,(t,d,c) in enumerate(steps):
        x=2+k*25; rbox(ax,x,28,22,52,"",fc="white",ec=c,lw=2.2,r=3); rbox(ax,x,64,22,16,t,fc=c,tc="white",bold=True,fs=10,r=3)
        ax.text(x+11,44,d,ha="center",va="center",fontsize=9,linespacing=1.5)
        if k<3: arrow(ax,(x+22.3,54),(x+24.7,54),c=INK,lw=2)
    arrow(ax,(86,24),(14,24),c=GRY,rad=-0.12,lw=1.6,ls="--"); ax.text(50,6,"o ciclo recomeça com outro cliente: a diferença para o bandit simples é o contexto",ha="center",fontsize=9.5,color=GRY,style="italic")
    save(fig,"f2_contextual")
if __name__=="__main__":
    for f in [f2_cassino,f2_distribuicoes,f2_incremental,f2_estrategias,f2_ucb,f2_thompson,f2_arrependimento,f2_contextual]: f()
