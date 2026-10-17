from fc import *
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def f2_dados():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"Dados em tabela: o formato que o AutoML mais usa",ha="center",fontsize=11,fontweight="bold")
    cols=["idade","renda (mil)","tempo de casa","comprou?"]; rows=[[34,5.2,3,"sim"],[51,8.9,12,"não"],[27,3.1,1,"não"],[45,7.4,8,"sim"],[39,6.0,5,"sim"]]
    x0=14; cw=17; y0=76
    for j,c in enumerate(cols):
        rbox(ax,x0+j*cw,y0,cw-0.6,8,c,fc=IND if j<3 else AMB,ec="white",fs=8.6,tc="white",bold=True,r=0.6)
    for i,r in enumerate(rows):
        for j,v in enumerate(r):
            rbox(ax,x0+j*cw,y0-9.5*(i+1),cw-0.6,8,str(v),fc="#EEF2FF" if j<3 else "#FEF3C7",ec="white",fs=8.6,r=0.6)
    ax.text(7,y0-9.5*3+4,"Exemplos\n(linhas)",ha="center",va="center",fontsize=8.5,color=INK,fontweight="bold"); arrow(ax,(11.5,y0-9.5*3+4),(13.5,y0-9.5*3+4),c=GRY,ms=9)
    ax.annotate("",xy=(x0,y0+11),xytext=(x0+3*cw-0.6,y0+11),arrowprops=dict(arrowstyle="-",color=IND,lw=2))
    ax.text(x0+1.5*cw,y0+13.5,"X = atributos (colunas de entrada)",ha="center",fontsize=9,color=IND,fontweight="bold")
    ax.text(x0+3.5*cw-0.3,y0+13.5,"y = alvo",ha="center",fontsize=9,color="#B45309",fontweight="bold")
    ax.text(50,14,"Prever uma categoria (sim/não) é classificação.  Prever um número (renda) é regressão.",ha="center",fontsize=9,color=INK)
    ax.text(50,6,"Cada linha é um exemplo; cada coluna, um atributo; a última coluna, o que queremos prever.",ha="center",fontsize=8.5,color=GRY,style="italic")
    save(fig,"f2_dados")

def f2_modelos():
    X,y=make_moons(n_samples=220,noise=0.28,random_state=1)
    ms=[("Árvore (prof. 4)",DecisionTreeClassifier(max_depth=4,random_state=0),IND),("k-vizinhos (k=7)",KNeighborsClassifier(7),CYA),("Regressão logística",LogisticRegression(),GRN),("Floresta aleatória",RandomForestClassifier(100,random_state=0),AMB)]
    fig,axs=plt.subplots(1,4,figsize=(8.6,2.5))
    xx,yy=np.meshgrid(np.linspace(-1.8,2.8,260),np.linspace(-1.4,1.9,260))
    for ax,(t,m,c) in zip(axs,ms):
        m.fit(X,y); Z=m.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx,yy,Z,cmap=ListedColormap(["#C7D2FE","#FDE68A"]),alpha=0.9)
        ax.scatter(X[:,0],X[:,1],c=np.where(y==0,IND,AMB),s=9,edgecolors="white",linewidths=0.3)
        ax.set_title(t,color=c,fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(True); s.set_color(c); s.set_linewidth(1.5)
    fig.suptitle("Retratos de quatro algoritmos: cada um \"desenha\" uma fronteira diferente",fontsize=10,fontweight="bold",y=1.03)
    save(fig,"f2_modelos")

def f2_overfit():
    rng=np.random.default_rng(4); x=np.sort(rng.uniform(0,1,18)); f=lambda t: np.sin(2*np.pi*t)
    y=f(x)+rng.normal(0,0.25,18); xt=np.linspace(0,1,300)
    fig,axs=plt.subplots(1,3,figsize=(8.6,2.7))
    for ax,(g,t,c) in zip(axs,[(1,"Subajuste (grau 1)",RED),(3,"Bom ajuste (grau 3)",GRN),(15,"Sobreajuste (grau 15)",AMB)]):
        m=make_pipeline(PolynomialFeatures(g),StandardScaler(),LogisticRegression) if False else None
        from sklearn.linear_model import Ridge
        m=make_pipeline(PolynomialFeatures(g),Ridge(alpha=1e-8)).fit(x[:,None],y)
        ax.plot(xt,f(xt),c=GRY,lw=1.5,ls="--",label="verdade"); ax.plot(xt,np.clip(m.predict(xt[:,None]),-2.2,2.2),c=c,lw=2.2,label="modelo")
        ax.scatter(x,y,c=INK,s=16,zorder=3); ax.set_ylim(-2.2,2.2); ax.set_title(t,color=c,fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    axs[0].legend(fontsize=7,loc="lower left",frameon=False)
    fig.suptitle("Aprender é achar o padrão, e não decorar o ruído",fontsize=10,fontweight="bold",y=1.02)
    save(fig,"f2_overfit")

def f2_viesvar():
    X,y=load_breast_cancer(return_X_y=True); X=StandardScaler().fit_transform(X)
    depths=np.arange(1,16)
    tr,va=validation_curve(DecisionTreeClassifier(random_state=0),X,y,param_name="max_depth",param_range=depths,cv=5)
    fig,ax=plt.subplots(figsize=(8.0,3.6))
    ax.plot(depths,1-tr.mean(1),"o-",c=IND,label="erro no treino"); ax.plot(depths,1-va.mean(1),"s-",c=AMB,label="erro na validação")
    b=int(np.argmin(1-va.mean(1))); ax.scatter([depths[b]],[1-va.mean(1)[b]],s=260,marker="*",c=GRN,edgecolors="white",zorder=5)
    ax.annotate("melhor complexidade",(depths[b],1-va.mean(1)[b]),(depths[b]+3.2,0.118),arrowprops=dict(arrowstyle="->",color=GRN),color=GRN,fontsize=9,fontweight="bold")
    ax.axvspan(0.5,depths[b]-0.5,color=RED,alpha=0.06); ax.axvspan(depths[b]+0.5,15.5,color=AMB,alpha=0.07)
    ax.text(0.8,0.118,"subajuste",fontsize=8.5,color=RED,fontweight="bold"); ax.text(9.5,0.112,"sobreajuste (complexo demais)",fontsize=8.5,color="#B45309",fontweight="bold")
    ax.set_xlabel("complexidade do modelo (profundidade da árvore)"); ax.set_ylabel("erro"); ax.set_xlim(0.5,15.5); ax.set_ylim(0,0.135); ax.legend(frameon=False,loc="center right")
    ax.set_title("Erro no treino sempre cai; na validação, volta a subir",fontweight="bold")
    save(fig,"f2_viesvar")

def f2_cv():
    fig,ax=canvas(8.6,3.7)
    ax.text(50,97,"Validação cruzada de 5 partes: cada parte é teste uma vez",ha="center",fontsize=11,fontweight="bold")
    for r in range(5):
        y=80-r*14
        ax.text(5,y+4,f"Rodada {r+1}",fontsize=8.5,fontweight="bold",va="center",color=INK)
        for k in range(5):
            isv=(k==r)
            rbox(ax,22+k*12.6,y,11.9,8,"validação" if isv else "treino",fc=AMB if isv else "#C7D2FE",ec="white",fs=8,tc="white" if isv else INK,bold=isv,r=0.8)
        ax.text(97,y+4,f"nota {r+1}",fontsize=8.5,ha="right",va="center",color=GRY)
    ax.text(50,6,"Nota final = média das 5 notas. Todo exemplo é validado exatamente uma vez.",ha="center",fontsize=9,color=INK,style="italic")
    save(fig,"f2_cv")

def f2_cofre():
    fig,ax=canvas(8.6,3.4)
    ax.text(50,95,"Três conjuntos, três papéis (a analogia do estudante)",ha="center",fontsize=11,fontweight="bold")
    parts=[(2,56,"TREINO","o aluno estuda\ncom estes exercícios",IND),(58,22,"VALIDAÇÃO","simulados para\nescolher como estudar",CYA),(82,16,"TESTE","a prova final,\nno cofre",RED)]
    for x,w,t,d,c in parts:
        rbox(ax,x,48,w-1.5,30,t,fc=c,ec=c,fs=10,tc="white",bold=True,r=1.5); ax.text(x+(w-1.5)/2,38,d,ha="center",va="center",fontsize=8,color=INK,linespacing=1.15)
    ax.text(30,86,"aprende os parâmetros",ha="center",fontsize=8.5,color=IND,fontweight="bold"); ax.text(69,86,"escolhe os hiperparâmetros",ha="center",fontsize=8.5,color="#0E7490",fontweight="bold"); ax.text(90,86,"mede o final",ha="center",fontsize=8.5,color=RED,fontweight="bold")
    ax.text(50,16,"Quem vê a prova antes de estudar não está medindo aprendizado.\nO teste só pode ser aberto UMA vez, no final.",ha="center",fontsize=9,color=INK,linespacing=1.3)
    save(fig,"f2_cofre")

def f2_confusao():
    fig,ax=canvas(8.6,3.8)
    ax.text(50,97,"Matriz de confusão: de onde saem as métricas",ha="center",fontsize=11,fontweight="bold")
    vals=[[("VP",40,GRN),("FN",10,RED)],[("FP",5,RED),("VN",45,GRN)]]
    ax.text(26.5,86,"Previu: positivo",ha="center",fontsize=8.5,fontweight="bold",color=INK); ax.text(46.5,86,"Previu: negativo",ha="center",fontsize=8.5,fontweight="bold",color=INK)
    ax.text(2,69,"Real:\npositivo",ha="left",va="center",fontsize=8.5,fontweight="bold"); ax.text(2,47,"Real:\nnegativo",ha="left",va="center",fontsize=8.5,fontweight="bold")
    for i in range(2):
        for j in range(2):
            n,v,c=vals[i][j]; rbox(ax,17+j*20,58-i*22,19,20,"",fc=c,ec="white",alpha=0.85,r=1.5)
            ax.text(26.5+j*20,72-i*22-2,n,ha="center",fontsize=11,fontweight="bold",color="white"); ax.text(26.5+j*20,64-i*22-1,str(v),ha="center",fontsize=14,fontweight="bold",color="white")
    ax.text(79,78,"Acurácia  = (VP+VN) / total = 85/100 = 0,85",fontsize=9,color=INK,ha="center")
    ax.text(79,66,"Precisão  = VP / (VP+FP) = 40/45 ≈ 0,89",fontsize=9,color=INK,ha="center")
    ax.text(79,54,"Revocação = VP / (VP+FN) = 40/50 = 0,80",fontsize=9,color=INK,ha="center")
    ax.text(79,42,"F1 = média harmônica ≈ 0,84",fontsize=9,color=INK,ha="center")
    ax.text(79,24,"Precisão: dos que previ positivos, quantos eram?\nRevocação: dos positivos reais, quantos achei?",fontsize=8.3,color=GRY,ha="center",style="italic",linespacing=1.3)
    ax.text(36,14,"VP verdadeiro positivo · VN verdadeiro negativo\nFP falso positivo · FN falso negativo",ha="center",fontsize=7.8,color=GRY)
    save(fig,"f2_confusao")

if __name__=="__main__":
    for f in (f2_dados,f2_modelos,f2_overfit,f2_viesvar,f2_cv,f2_cofre,f2_confusao): f()
