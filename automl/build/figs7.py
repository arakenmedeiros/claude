from fc import *
import json, os
from ga_lib import *
C7="/home/user/claude/automl/build/f7_cache.json"

def dados_noisy():
    X,y=carregar(); rng=np.random.default_rng(0); idx=rng.choice(len(y),300,replace=False)
    return np.hstack([X[idx],rng.normal(size=(300,200))]),y[idx]

def experimento():
    if os.path.exists(C7): return json.load(open(C7))
    X,y=dados_noisy(); out={"ga":[],"al":[],"pop_media":[],"todos0":[]}
    for s in range(5):
        h,todos,cache=evoluir(X,y,s,pop=14,geracoes=5); n=h[-1][0]; al=aleatoria(X,y,s,n)
        curva=[]; j=0
        for k in range(1,n+1):
            while j+1<len(h) and h[j+1][0]<=k: j+=1
            curva.append(h[j][1])
        out["ga"].append(curva); out["al"].append(al); out["pop_media"].append([m for _,_,m in h])
        if s==0: out["todos0"]=[[g.get("pca",0),float(n_)] for g,n_ in todos]
    m=min(len(c) for c in out["ga"]); out["ga"]=[c[:m] for c in out["ga"]]; out["al"]=[c[:m] for c in out["al"]]
    json.dump(out,open(C7,"w")); return out

def f7_natureza():
    rng=np.random.default_rng(2); xs=np.linspace(0,10,200); ys=np.linspace(0,10,200); XX,YY=np.meshgrid(xs,ys)
    Z=np.exp(-(((XX-7)/1.6)**2+((YY-6.5)/1.6)**2))+0.55*np.exp(-(((XX-2.5)/1.8)**2+((YY-3)/1.8)**2))
    fig,axs=plt.subplots(1,4,figsize=(8.6,2.6),sharey=True)
    pop=rng.uniform(0,10,(30,2))
    f=lambda p: np.exp(-(((p[:,0]-7)/1.6)**2+((p[:,1]-6.5)/1.6)**2))+0.55*np.exp(-(((p[:,0]-2.5)/1.8)**2+((p[:,1]-3)/1.8)**2))
    for g,ax in enumerate(axs):
        ax.contourf(XX,YY,Z,levels=10,cmap="YlGn",alpha=0.7); ax.scatter(pop[:,0],pop[:,1],s=22,c=[IND,CYA,AMB,RED][g],edgecolors="white",linewidths=0.6)
        ax.set_title(f"Geração {[0,2,5,10][g]}",fontsize=9.5,fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        for _ in range([2,3,5][g] if g<3 else 0):
            for _ in range(1):
                for _ in range(1): pass
        steps=[2,3,5][g] if g<3 else 0
        for _ in range(steps):
            fit=f(pop); new=[]
            for _ in range(len(pop)):
                a,b=rng.choice(len(pop),3,replace=False)[:2]; i=a if fit[a]>fit[b] else b; new.append(pop[i]+rng.normal(0,0.35,2))
            pop=np.clip(np.array(new),0,10)
    fig.suptitle("A população evolui e se concentra nas regiões de maior aptidão (simulação real de um algoritmo genético)",fontsize=9.3,fontweight="bold",y=1.03)
    fig.tight_layout(); save(fig,"f7_natureza")

def f7_ciclo():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"O ciclo de um algoritmo genético",ha="center",fontsize=11,fontweight="bold")
    B=[(37,72,"1. População inicial\n(pipelines sorteados)",IND),(70,52,"2. Avaliar\n(aptidão = nota de\nvalidação cruzada)",CYA),(70,18,"3. Selecionar\n(os melhores têm\nmais chance)",GRN),(37,2,"4. Cruzar\n(misturar dois pais)",AMB),(4,18,"5. Mutar\n(pequenas mudanças\naleatórias)",RED),(4,52,"6. Nova geração\n(repete até acabar\no orçamento)",VIO)]
    for x,y,t,c in B: rbox(ax,x,y,26,20,t,fc=c,ec=c,fs=8.5,tc="white",bold=True,r=2)
    arrow(ax,(55,76),(70,66),lw=2,ms=12); arrow(ax,(83,51.5),(83,38.5),lw=2,ms=12); arrow(ax,(70,24),(60,16),lw=2,ms=12)
    arrow(ax,(37,10),(26,18),lw=2,ms=12); arrow(ax,(17,38.5),(17,51.5),lw=2,ms=12); arrow(ax,(26,66),(37,74),lw=2,ms=12)
    ax.text(50,45,"gerações",ha="center",fontsize=10,color=GRY,style="italic",fontweight="bold")
    save(fig,"f7_ciclo")

def f7_operadores():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"Cruzamento e mutação sobre o \"genoma\" de um pipeline",ha="center",fontsize=11,fontweight="bold")
    genes=["escala","PCA","modelo","C","gamma"]
    A=["padrão","20","SVM","10","0,01"]; B=["min-máx","0","floresta","—","—"]
    def linha(x,y,vals,c,lab):
        ax.text(x-1,y+3.5,lab,ha="right",fontsize=8.5,fontweight="bold",va="center",color=c)
        for i,v in enumerate(vals): rbox(ax,x+i*11.5,y,10.5,7,v,fc=c,ec="white",fs=8,tc="white",bold=True,r=0.8)
    for i,g in enumerate(genes): ax.text(21.5+i*11.5,88,g,ha="center",fontsize=8,color=GRY)
    linha(16,78,A,IND,"Pai A"); linha(16,67,B,CYA,"Pai B")
    ax.text(50,59,"cruzamento: cada gene vem de um dos pais",ha="center",fontsize=8.5,color=INK,style="italic")
    arrow(ax,(50,66),(50,52),c=GRY,lw=1.8,ms=11)
    filho=[("padrão",IND),("0",CYA),("SVM",IND),("10",IND),("0,01",IND)]
    ax.text(15,47.5,"Filho",ha="right",fontsize=8.5,fontweight="bold",va="center",color=GRN)
    for i,(v,c) in enumerate(filho): rbox(ax,16+i*11.5,44,10.5,7,v,fc=c,ec="white",fs=8,tc="white",bold=True,r=0.8)
    arrow(ax,(50,43),(50,29),c=GRY,lw=1.8,ms=11); ax.text(50,36,"mutação: um gene muda por acaso",ha="center",fontsize=8.5,color=INK,style="italic")
    ax.text(15,21.5,"Filho\nmutado",ha="right",fontsize=8.5,fontweight="bold",va="center",color=RED)
    mut=[("padrão",IND),("0",CYA),("SVM",IND),("12",RED),("0,01",IND)]
    for i,(v,c) in enumerate(mut): rbox(ax,16+i*11.5,18,10.5,7,v,fc=c,ec="white",fs=8,tc="white",bold=True,r=0.8)
    ax.text(50,6,"O cruzamento combina boas ideias; a mutação mantém a diversidade e evita ficar preso em um só tipo de solução.",ha="center",fontsize=8.3,color=GRY,style="italic")
    save(fig,"f7_operadores")

def f7_arvore():
    fig,ax=canvas(8.6,4.3)
    ax.text(50,97,"Programação genética: pipelines como árvores",ha="center",fontsize=11,fontweight="bold")
    def no(x,y,t,c,w=15): rbox(ax,x-w/2,y-4.4,w,8.8,t,fc=c,ec="white",fs=7.4,tc="white",bold=True,r=1)
    def lig(a,b): ax.plot([a[0],b[0]],[a[1]-4.4,b[1]+4.4],c=GRY,lw=1.3,zorder=0)
    # tree A
    A={"raiz":(22,80),"n1":(12,62),"n2":(32,62),"n3":(6,44),"n4":(18,44)}
    no(*A["raiz"],"Modelo\n(SVM)",IND); no(*A["n1"],"PCA",CYA,10); no(*A["n2"],"Escala\n(padrão)",CYA,12); no(*A["n3"],"Dados",GRY,9); no(*A["n4"],"Seleção",GRN,11)
    for a,b in [("raiz","n1"),("raiz","n2"),("n1","n3"),("n1","n4")]: lig(A[a],A[b])
    ax.text(22,90,"Pai A",ha="center",fontsize=9,fontweight="bold",color=IND)
    # tree B
    Bt={"raiz":(70,80),"n1":(60,62),"n2":(80,62),"n3":(76,44)}
    no(*Bt["raiz"],"Modelo\n(floresta)",AMB); no(*Bt["n1"],"Escala\n(min-máx)",CYA,12); no(*Bt["n2"],"Seleção",GRN,11); no(*Bt["n3"],"Dados",GRY,9)
    for a,b in [("raiz","n1"),("raiz","n2"),("n2","n3")]: lig(Bt[a],Bt[b])
    ax.text(70,90,"Pai B",ha="center",fontsize=9,fontweight="bold",color=AMB)
    ax.text(46,70,"troca de\nsubárvores\n(cruzamento)",ha="center",fontsize=8,color=RED,fontweight="bold")
    ax.annotate("",xy=(54,66),xytext=(39,66),arrowprops=dict(arrowstyle="<->",color=RED,lw=2))
    # child
    C={"raiz":(46,32),"n1":(36,16),"n2":(56,16)}
    ax.text(46,41,"Filho",ha="center",fontsize=9,fontweight="bold",color=GRN)
    no(*C["raiz"],"Modelo\n(SVM)",IND); no(*C["n1"],"PCA",CYA,10); no(*C["n2"],"Seleção",GRN,11)
    for a,b in [("raiz","n1"),("raiz","n2")]: lig(C[a],C[b])
    ax.text(50,3,"Cada nó é uma etapa do pipeline. Trocar subárvores troca pedaços inteiros do tratamento dos dados.",ha="center",fontsize=8.3,color=GRY,style="italic")
    save(fig,"f7_arvore")

def f7_pareto():
    d=experimento(); P=np.array(d["todos0"]); feats=np.where(P[:,0]>0,P[:,0],264); acc=P[:,1]
    dom=lambda i: any((feats[j]<=feats[i] and acc[j]>=acc[i]) and (feats[j]<feats[i] or acc[j]>acc[i]) for j in range(len(P)))
    nd=np.array([not dom(i) for i in range(len(P))])
    fig,ax=plt.subplots(figsize=(8.0,3.8))
    ax.scatter(feats[~nd]+np.random.default_rng(0).uniform(-2,2,(~nd).sum()),acc[~nd],c=GRY,s=26,alpha=0.7,label="pipelines avaliados")
    pf=sorted(set(zip(feats[nd],acc[nd]))); ax.step([p[0] for p in pf],[p[1] for p in pf],where="post",c=RED,lw=2,label="fronteira de Pareto")
    ax.scatter([p[0] for p in pf],[p[1] for p in pf],s=70,c=RED,edgecolors="white",zorder=4)
    ax.set_xlabel("número de atributos usados pelo modelo (menos = mais simples)"); ax.set_ylabel("acurácia (validação cruzada)"); ax.legend(frameon=False,loc="lower right")
    ax.set_title("Dois objetivos ao mesmo tempo: acertar muito e usar poucos atributos",fontweight="bold",fontsize=10)
    save(fig,"f7_pareto")

def f7_comparacao():
    d=experimento(); G=np.array(d["ga"],dtype=float); A=np.array(d["al"]); k=np.arange(1,G.shape[1]+1); G[:,:13]=np.nan   # o GA só tem nota depois de avaliar a 1ª geração (14 pipelines)
    fig,axs=plt.subplots(1,2,figsize=(8.6,3.5),gridspec_kw={"width_ratios":[1.6,1]})
    ax=axs[0]
    for M,c,l in [(A,GRY,"busca aleatória"),(G,IND,"algoritmo genético")]:
        ax.plot(k,np.nanmean(M,0),c=c,lw=2.2,label=l); ax.fill_between(k,np.nanmin(M,0),np.nanmax(M,0),color=c,alpha=0.15)
    ax.set_ylim(0.8,0.96); ax.set_xlabel("pipelines avaliados"); ax.set_ylabel("melhor acurácia até agora"); ax.legend(frameon=False,loc="lower right"); ax.set_title("Convergência (média e faixa de 5 sementes)",fontsize=9.5,fontweight="bold")
    ax=axs[1]; PM=np.array(d["pop_media"]); ax.plot(range(PM.shape[1]),PM.mean(0),"o-",c=GRN,lw=2); ax.set_xlabel("geração"); ax.set_title("Acurácia média da população",fontsize=9.5,fontweight="bold")
    fig.tight_layout(); save(fig,"f7_comparacao")

def f7_tpot():
    fig,ax=canvas(8.6,3.8)
    ax.text(50,97,"Como o TPOT trabalha",ha="center",fontsize=11,fontweight="bold")
    B=[(2,"Dados\nde treino",GRY),(21,"População de\npipelines\n(árvores)",IND),(40,"Avaliação por\nvalidação\ncruzada",CYA),(59,"Seleção\nNSGA-II\n(acerto e\nsimplicidade)",GRN),(78,"Cruzamento\ne mutação",AMB)]
    for x,t,c in B: rbox(ax,x,48,18,26,t,fc=c,ec=c,fs=8.3,tc="white",bold=True,r=2)
    for x in (20.2,39.2,58.2,77.2): arrow(ax,(x,61),(x+0.9,61),lw=1.6,ms=9)
    ax.annotate("",xy=(30,46),xytext=(87,46),arrowprops=dict(arrowstyle="-|>",color=GRY,lw=1.6,connectionstyle="arc3,rad=-0.25")); ax.text(58,31,"nova geração",ha="center",fontsize=8.5,color=GRY,style="italic")
    rbox(ax,12,5,76,14,"Saída: o melhor pipeline, exportado como código Python do scikit-learn",fc="#ECFDF5",ec=GRN,fs=8.6,tc=INK,lw=1.8,r=1.5)
    save(fig,"f7_tpot")

if __name__=="__main__":
    for fn_ in (f7_natureza,f7_ciclo,f7_operadores,f7_arvore,f7_tpot,f7_pareto,f7_comparacao): fn_()
