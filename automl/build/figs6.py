from fc import *
import json, os, math
from hb_lib import *
C6="/home/user/claude/automl/build/f6_cache.json"

def f6_peneira():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"A peneira de talentos: treinar pouco, cortar, treinar mais",ha="center",fontsize=11,fontweight="bold")
    rounds=[(27,"Rodada 1","1 treino",IND),(9,"Rodada 2","3 treinos",CYA),(3,"Rodada 3","9 treinos",GRN),(1,"Final","27 treinos",AMB)]
    for i,(n,t,r,c) in enumerate(rounds):
        x=3+i*24.5; ax.text(x+10,86,t,ha="center",fontsize=9.5,fontweight="bold",color=c)
        cols=min(n,9); rows=int(np.ceil(n/9))
        for k in range(n):
            cx=x+1.5+(k%9)*2.2; cy=76-(k//9)*5.5
            ax.scatter([cx],[cy],s=34,c=c,edgecolors="white",linewidths=0.8,zorder=3)
        ax.text(x+10,58 if n>9 else 66,f"{n} candidato"+("s" if n>1 else ""),ha="center",fontsize=8.8,color=INK)
        rbox(ax,x,30,20,14,r,fc=c,ec=c,fs=9,tc="white",bold=True,r=1.5)
        if i<3: arrow(ax,(x+21,70),(x+24,70),c=GRY,lw=1.8,ms=11); ax.text(x+22.7,64,"corta\n2/3",ha="center",fontsize=7.3,color=RED,va="top")
    ax.text(50,16,"Candidatos fracos saem cedo e gastam pouco. Os fortes recebem cada vez mais treino.",ha="center",fontsize=9,color=INK)
    ax.text(50,8,"Cada rodada multiplica por 3 o treino e divide por 3 o número de candidatos: o custo de cada rodada é parecido.",ha="center",fontsize=8.3,color=GRY,style="italic")
    save(fig,"f6_peneira")

def f6_fidelidade():
    fig,ax=canvas(8.6,3.8)
    ax.text(50,97,"Fidelidade: versões baratas de uma avaliação cara",ha="center",fontsize=11,fontweight="bold")
    items=[("Épocas","treinar poucas\nvoltas e continuar\nse for bom",IND),("Dados","treinar com 10%,\n30%, 100% das\nlinhas",CYA),("Atributos","usar um subconjunto\nde colunas no\nteste rápido",GRN),("Resolução","imagens pequenas\nantes das\ngrandes",AMB),("Validação","menos partes da\nvalidação cruzada\nno início",PNK)]
    for i,(t,d,c) in enumerate(items):
        x=2+i*19.6; rbox(ax,x,56,18,30,"",fc="white",ec=c,lw=2,r=1.5); ax.text(x+9,80,t,ha="center",fontsize=9.5,fontweight="bold",color=c); ax.text(x+9,71,d,ha="center",va="center",fontsize=7.6,color=INK,linespacing=1.2)
        for k in range(5): ax.add_patch(Rectangle((x+2+k*3,58),2.4,1+k*0.9,fc=c,alpha=0.35+k*0.15,ec="none"))
    ax.text(50,40,"Barato demais perde precisão. Caro demais perde velocidade.",ha="center",fontsize=9.5,fontweight="bold",color=INK)
    ax.text(50,28,"A hipótese central: o ranking das configurações em baixa fidelidade se parece com o de alta fidelidade.",ha="center",fontsize=9,color=INK)
    ax.text(50,18,"Quando essa hipótese falha (o melhor candidato final parecia ruim no começo), o corte precoce descarta um campeão.",ha="center",fontsize=8.5,color=RED,style="italic")
    save(fig,"f6_fidelidade")

def f6_curvas():
    D=dados(0); rng=np.random.default_rng(4); cs=[Candidato(rng,D) for _ in range(40)]
    for c in cs: c.treinar_ate(81)
    fig,axs=plt.subplots(1,2,figsize=(8.6,3.6),sharey=True)
    fin=np.array([c.nota for c in cs]); top=set(np.argsort(-fin)[:3])
    e=np.arange(1,82)
    for ax,tit in zip(axs,("Todas as 40 configurações","O que o corte em 3 épocas enxerga")):
        for i,c in enumerate(cs):
            cur=np.array(c.curva); col=GRN if i in top else GRY
            if ax is axs[1]: ax.plot(e[:3],cur[:3],c=col,lw=1.6 if i in top else 0.8,alpha=1 if i in top else 0.6)
            else: ax.plot(e,cur,c=col,lw=2 if i in top else 0.8,alpha=1 if i in top else 0.5)
        ax.set_title(tit,fontsize=9.5,fontweight="bold"); ax.set_xlabel("épocas de treino",fontsize=8.5); ax.set_ylim(0.7,1.0)
        if ax is axs[1]: ax.set_xticks([1,2,3])
    axs[0].set_ylabel("acurácia na validação",fontsize=8.5); axs[0].plot([],[],c=GRN,lw=2,label="as 3 melhores no fim"); axs[0].plot([],[],c=GRY,label="as demais"); axs[0].legend(frameon=False,fontsize=8,loc="lower right")
    top3=np.array([[c.curva[2] for c in cs]]).ravel(); rank=int((top3>=np.sort(top3)[-13]).sum())
    fig.suptitle("Curvas de aprendizado reais: as melhores no fim já se destacam no começo?",fontsize=10,fontweight="bold",y=1.02); fig.text(0.5,-0.02,"(configurações que divergem ficam abaixo de 0,7 e saem do gráfico)",ha="center",fontsize=7.5,color=GRY,style="italic")
    fig.tight_layout(); save(fig,"f6_curvas")

def f6_brackets():
    R,eta=81,3; smax=4; B=(smax+1)*R
    fig,ax=canvas(8.6,4.3); ax.text(50,97,"Hyperband (R = 81, η = 3): cinco \"torneios\" com apostas diferentes",ha="center",fontsize=11,fontweight="bold")
    ax.text(3,88,"Torneio",fontsize=8.5,fontweight="bold",color=GRY)
    cols=[IND,CYA,GRN,AMB,RED]
    for s in range(smax,-1,-1):
        n=int(math.ceil(B/R*eta**s/(s+1))); r=R*eta**(-s); y=80-(smax-s)*14
        ax.text(3,y+2.5,f"s = {s}",fontsize=9,fontweight="bold",color=cols[smax-s],va="center")
        for i in range(s+1):
            ni=int(n*eta**(-i)); ri=r*eta**i
            rbox(ax,13+i*17,y-2,16,9,f"{ni} cfg × {ri:g} ép.",fc=cols[smax-s],ec="white",fs=7.6,tc="white",bold=True,r=1)
            if i<s: arrow(ax,(29.2+i*17,y+2.5),(30+i*17,y+2.5),c=INK,lw=1,ms=7)
    ax.text(50,5,"s grande: muitas configurações com pouco treino (aposta ampla). s = 0: poucas configurações com treino completo (aposta conservadora).",ha="center",fontsize=7.8,color=GRY,style="italic")
    save(fig,"f6_brackets")

def f6_bohb():
    fig,ax=canvas(8.6,3.4)
    ax.text(50,96,"BOHB = o melhor de dois mundos",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,3,50,28,30,"Hyperband\n\neconomiza\navaliações com\ncortes precoces",fc=IND,ec=IND,fs=8.8,tc="white",bold=True,r=2)
    rbox(ax,69,50,28,30,"Otimização\nbayesiana (TPE)\n\ngera candidatos\ninteligentes",fc=GRN,ec=GRN,fs=8.8,tc="white",bold=True,r=2)
    ax.text(50,66,"+",ha="center",fontsize=26,fontweight="bold",color=INK)
    rbox(ax,22,12,56,24,"BOHB: o Hyperband organiza os torneios, mas os candidatos\nnão são sorteados ao acaso: vêm de um modelo TPE\ntreinado com os resultados de cada fidelidade",fc="#FEF3C7",ec=AMB,fs=8.5,tc=INK,lw=2,r=2)
    arrow(ax,(17,49),(32,37),c=GRY,lw=1.6,ms=10); arrow(ax,(83,49),(68,37),c=GRY,lw=1.6,ms=10)
    save(fig,"f6_bohb")

def dados_comparacao():
    if os.path.exists(C6): return json.load(open(C6))
    sh_,ra_,hb_,rb_=[],[],[],[]
    for s in range(12):
        D=dados(0)
        sh,gs=sucessivo([Candidato(np.random.default_rng(2000+s*1000+i),D) for i in range(81)],1,3,81); ra,_=aleatoria(np.random.default_rng(3000+s),D,gs)
        c,g,n=hyperband(np.random.default_rng(s),D); rb,_=aleatoria(np.random.default_rng(1000+s),D,g)
        sh_.append(sh.nota); ra_.append(ra); hb_.append(c.nota); rb_.append(rb)
    d={"sh":sh_,"ra":ra_,"hb":hb_,"rb":rb_,"gasto_sh":gs,"gasto_hb":g}; json.dump(d,open(C6,"w")); return d

def f6_comparacao():
    d=dados_comparacao(); fig,ax=plt.subplots(figsize=(8.0,3.6))
    labs=[f"Aleatória\n({d['gasto_sh']} épocas)",f"Successive\nHalving\n({d['gasto_sh']} épocas)",f"Aleatória\n({d['gasto_hb']} épocas)",f"Hyperband\n({d['gasto_hb']} épocas)"]
    vals=[d["ra"],d["sh"],d["rb"],d["hb"]]; cols=[GRY,IND,GRY,IND]
    for i,(v,c) in enumerate(zip(vals,cols)):
        v=np.array(v); ax.bar(i,v.mean(),color=c,alpha=0.85,width=0.6); ax.errorbar(i,v.mean(),yerr=v.std(),c=INK,capsize=4); ax.scatter([i]*len(v)+np.random.default_rng(i).uniform(-0.15,0.15,len(v)),v,s=10,c="white",edgecolors=INK,linewidths=0.5,zorder=3)
        ax.text(i,0.918,f"{v.mean():.3f}",ha="center",fontsize=9,fontweight="bold",color="white" if False else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs,fontsize=8.3); ax.set_ylim(0.91,0.985); ax.set_ylabel("melhor acurácia (validação)")
    ax.set_title("Mesmo orçamento, resultados parecidos: o ganho é modesto neste problema",fontweight="bold",fontsize=10)
    save(fig,"f6_comparacao")

if __name__=="__main__":
    for fn_ in (f6_peneira,f6_fidelidade,f6_curvas,f6_brackets,f6_bohb,f6_comparacao): fn_()
