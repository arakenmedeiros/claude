from fc import *
from figs_cover import star
def f12_acumulo():
    fig,ax=plt.subplots(figsize=(8.2,3.6)); t=np.linspace(0,10,100); rng=np.random.default_rng(3)
    esp=np.sin(t*0.6)*1.0; ax.plot(t,esp,c=GRN,lw=3,label="especialista (dados de treino)"); ax.fill_between(t,esp-0.45,esp+0.45,color=GRN,alpha=0.15,label="região coberta pelos dados")
    desvio=np.cumsum(rng.normal(0,0.04,100))*0.9+np.linspace(0,1.5,100)**2.2*0.9; ap=esp+desvio
    ax.plot(t,ap,c=RED,lw=2.5,label="aprendiz (clone)"); k=int(np.argmax(np.abs(desvio)>0.5)); ax.scatter([t[k]],[ap[k]],s=110,c=RED,zorder=5); ax.annotate("aqui o clone sai da região\nconhecida: erra mais e se\nafasta ainda mais",xy=(t[k],ap[k]),xytext=(t[k]-4.2,ap[k]+1.0),arrowprops=dict(arrowstyle="->",color=RED),fontsize=9,color=RED,fontweight="bold"); ax.set_ylim(-1.6,3.2)
    ax.set_xlabel("tempo"); ax.set_ylabel("estado"); ax.set_title("Acúmulo de erros na clonagem de comportamento"); ax.legend(frameon=False,fontsize=8.5,loc="lower left"); save(fig,"f12_acumulo")
def f12_deslocamento():
    fig,ax=plt.subplots(figsize=(7.4,3.6)); a=np.linspace(-3,3,300); verdadeiro=np.exp(-0.5*(a-0.5)**2)*1.2+0.2*np.sin(a)
    ax.plot(a,verdadeiro,c=GRN,lw=2.5,label="valor verdadeiro Q(s, a)"); dentro=(a>-1)&(a<1.6)
    est=verdadeiro.copy(); est[~dentro]=verdadeiro[~dentro]+0.5+0.5*np.abs(a[~dentro])*0.8*(a[~dentro]>1.6)+0.8*(a[~dentro]<-1)*np.abs(a[~dentro]+1)
    ax.plot(a,est,c=RED,lw=2.5,ls="--",label="Q aprendida (extrapola e superestima)"); ax.axvspan(-1,1.6,color=IND,alpha=0.1); ax.text(0.3,-0.22,"ações presentes\nnos dados",ha="center",fontsize=9,color=IND,fontweight="bold")
    ax.scatter([2.7],[est[np.argmin(np.abs(a-2.7))]],s=140,c=RED,marker="X",zorder=5); ax.annotate("a política escolheria\nesta ação, nunca vista",xy=(2.7,est[np.argmin(np.abs(a-2.7))]),xytext=(0.6,2.0),arrowprops=dict(arrowstyle="->",color=RED),fontsize=9,color=RED,fontweight="bold")
    ax.set_xlabel("ação a"); ax.set_ylabel("valor"); ax.set_ylim(-0.4,3.2); ax.set_title("AR offline: extrapolação fora dos dados"); ax.legend(frameon=False,fontsize=8.5,loc="upper left"); save(fig,"f12_deslocamento")
def f12_estimadores():
    from sklearn.linear_model import LogisticRegression
    rng=np.random.default_rng(0); n=1500; X=rng.normal(0,1,(n,2))
    def mu_v(X,a): z=np.where(a==1,0.5*np.sin(2*X[:,0])+0.6*X[:,1]**2-0.4,-0.3*X[:,0]+0.2); return 1/(1+np.exp(-z))
    e=1/(1+np.exp(-(0.8*X[:,0]+0.3*X[:,1]))); e=np.clip(e,0.1,0.9)
    pi=((mu_v(X,np.ones(n,int))>mu_v(X,np.zeros(n,int)))).astype(int); verdade=mu_v(X,pi).mean(); R=200; est={k:[] for k in ["DM","IPS","SNIPS","DR"]}
    for r in range(R):
        a=(rng.random(n)<e).astype(int); y=(rng.random(n)<mu_v(X,a)).astype(int); F=np.column_stack([X,X**2,X[:,:1]*X[:,1:]])
        mu=np.column_stack([LogisticRegression(max_iter=500).fit(F[a==k],y[a==k]).predict_proba(F)[:,1] for k in (0,1)]); idx=np.arange(n); w=(a==pi)/np.where(a==1,e,1-e)
        est["DM"].append(mu[idx,pi].mean()); est["IPS"].append((w*y).mean()); est["SNIPS"].append((w*y).sum()/w.sum()); est["DR"].append((mu[idx,pi]+w*(y-mu[idx,a])).mean())
    fig,ax=plt.subplots(figsize=(8,3.8)); cols=[AMB,RED,CYA,GRN]
    parts=ax.violinplot([est[k] for k in est],showextrema=False,widths=0.8)
    for pc,c in zip(parts["bodies"],cols): pc.set_facecolor(c); pc.set_alpha(0.5)
    for i,(k,c) in enumerate(zip(est,cols)): ax.scatter(np.full(R,i+1)+rng.normal(0,0.05,R),est[k],s=5,c=c,alpha=0.5); ax.scatter([i+1],[np.mean(est[k])],s=80,c=INK,marker="D",zorder=5)
    ax.axhline(verdade,c=INK,ls="--",lw=1.5); ax.text(0.55,verdade,"valor verdadeiro",va="bottom",ha="left",fontsize=9,fontweight="bold"); ax.set_xticks(range(1,5)); ax.set_xticklabels(list(est)); ax.set_ylabel("estimativa de V(π)")
    ax.set_title("Estimadores off-policy em 200 repetições (losango = média)"); save(fig,"f12_estimadores")
if __name__=="__main__":
    import sys
    fs={f.__name__[4:]:f for f in [f12_acumulo,f12_deslocamento,f12_estimadores]}
    for k in (sys.argv[1:] or fs): fs[k]()
