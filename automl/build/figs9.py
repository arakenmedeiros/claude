from fc import *
from meta_lib import tudo
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from matplotlib.colors import ListedColormap

def f9_medico():
    fig,ax=canvas(8.6,3.8)
    ax.text(50,97,"Aprender a aprender: a experiência de quem já resolveu casos parecidos",ha="center",fontsize=10.5,fontweight="bold")
    rbox(ax,2,52,28,34,"Médico iniciante\n\ntesta todos os\nexames em cada\npaciente novo",fc=GRY,ec=GRY,fs=8.6,tc="white",bold=True,r=2)
    rbox(ax,36,52,28,34,"Médico experiente\n\nreconhece o caso:\n\"parece com os que\njá vi\" e começa\npelo exame certo",fc=IND,ec=IND,fs=8.6,tc="white",bold=True,r=2)
    rbox(ax,70,52,28,34,"Meta-aprendizado\n\nusa o histórico de\nconjuntos anteriores\npara decidir por\nonde começar",fc=GRN,ec=GRN,fs=8.6,tc="white",bold=True,r=2)
    arrow(ax,(31,69),(35,69),lw=2,ms=11); arrow(ax,(65,69),(69,69),lw=2,ms=11)
    ax.text(50,36,"Em AutoML, cada \"caso\" é um conjunto de dados já visto, com os resultados de diferentes modelos e ajustes.",ha="center",fontsize=9,color=INK)
    ax.text(50,24,"O meta-aprendizado responde: dado um problema novo, que algoritmos e ajustes valem a pena testar primeiro?",ha="center",fontsize=9,color=INK)
    ax.text(50,10,"Quanto mais conjuntos anteriores, melhor a \"experiência\" acumulada.",ha="center",fontsize=8.5,color=GRY,style="italic")
    save(fig,"f9_medico")

def f9_rice():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"O quadro de Rice (1976) para a seleção de algoritmos",ha="center",fontsize=11,fontweight="bold")
    B=[(2,"Espaço dos\nproblemas\n(conjuntos de dados)",IND),(27,"Espaço das\ncaracterísticas\n(meta-atributos)",CYA),(52,"Mapeamento de\nseleção\n(o meta-modelo)",GRN),(77,"Espaço dos\nalgoritmos",AMB)]
    for x,t,c in B: rbox(ax,x,52,21,30,t,fc=c,ec=c,fs=8.6,tc="white",bold=True,r=2)
    for x in (23.4,48.4,73.4): arrow(ax,(x,67),(x+3.2,67),lw=2,ms=11)
    ax.text(25,89,"extrair",ha="center",fontsize=8,color=GRY,style="italic"); ax.text(50,89,"aprender",ha="center",fontsize=8,color=GRY,style="italic"); ax.text(75,89,"recomendar",ha="center",fontsize=8,color=GRY,style="italic")
    rbox(ax,20,16,62,20,"Medida de desempenho\nQuão bem o algoritmo escolhido resolve o problema?",fc="#FEF3C7",ec=AMB,fs=8.8,tc=INK,lw=2,r=2)
    arrow(ax,(87,51),(76,37),c=GRY,lw=1.5,ms=10); arrow(ax,(63,51),(55,37),c=GRY,lw=1.5,ms=10)
    ax.text(50,6,"Rice formulou o problema em 1976; o meta-aprendizado moderno é a sua versão guiada por dados (Brazdil et al., 2022).",ha="center",fontsize=8.3,color=GRY,style="italic")
    save(fig,"f9_rice")

def f9_metaatributos():
    fig,ax=canvas(8.6,4.2)
    ax.text(50,97,"Meta-atributos: o \"retrato\" de um conjunto de dados",ha="center",fontsize=11,fontweight="bold")
    G=[("Simples e estatísticos","nº de exemplos, nº de atributos,\nnº de classes, assimetria média",IND),("Teoria da informação","entropia das classes,\ninformação mútua",CYA),("Complexidade do problema","sobreposição entre classes,\nsuavidade da fronteira",GRN),("Landmarkers","nota de modelos simples\n(árvore rasa, k-NN) no problema",AMB)]
    for i,(t,d,c) in enumerate(G):
        x=2+(i%2)*49; y=50-(i//2)*0 if False else (52 if i<2 else 14)
        rbox(ax,x,y,47,33,"",fc="white",ec=c,lw=2.2,r=2); ax.text(x+23.5,y+27,t,ha="center",fontsize=9.5,fontweight="bold",color=c); ax.text(x+23.5,y+13,d,ha="center",va="center",fontsize=8.5,color=INK,linespacing=1.3)
    ax.text(50,6,"Quatro famílias de meta-atributos (Lorena et al., 2019; Rivolli et al., 2022). Calculados em segundos, antes de treinar qualquer modelo.",ha="center",fontsize=8,color=GRY,style="italic")
    save(fig,"f9_metaatributos")

def f9_metaresultado():
    d=tudo()["meta"]; fig,axs=plt.subplots(1,2,figsize=(8.6,3.3))
    a=axs[0]; v=[d["acc_maj"],d["acc_meta"]]; a.bar(["sempre o mais\nfrequente (SVM)","meta-modelo\n(floresta)"],v,color=[GRY,IND],width=0.55); a.set_ylim(0,0.8); a.set_title("Acertos ao indicar o melhor algoritmo",fontsize=9,fontweight="bold")
    for i,x in enumerate(v): a.text(i,x+0.02,f"{x:.2f}",ha="center",fontweight="bold")
    b=axs[1]; r=[-d["regret_maj"],-d["regret_meta"]]; b.bar(["sempre o mais\nfrequente (SVM)","meta-modelo\n(floresta)"],r,color=[GRY,IND],width=0.55); b.set_title("Perda média frente ao melhor (menor = melhor)",fontsize=9,fontweight="bold")
    for i,x in enumerate(r): b.text(i,x+0.0003,f"{x:.4f}",ha="center",fontweight="bold")
    b.set_ylim(0,max(r)*1.25)
    fig.suptitle("Meta-aprendizado em 150 conjuntos simulados (validação deixando um conjunto de fora)",fontsize=9.6,fontweight="bold",y=1.02); fig.tight_layout(); save(fig,"f9_metaresultado")

def f9_warmstart():
    d=tudo()["portfolio"]["alvos"]; fig,axs=plt.subplots(1,4,figsize=(8.6,3.2),sharey=False)
    for ax,(nome,r) in zip(axs,d.items()):
        ks=[1,3,6]; ax.plot([1,2,3],r["portfolio"],"o-",c=IND,lw=2.2,label="portfólio (warm start)")
        ax.plot(ks,[r["aleatoria"][str(k)] for k in ks],"s--",c=GRY,lw=2,label="sorteio aleatório"); ax.axhline(r["melhor"],c=GRN,ls=":",lw=1.5)
        ax.set_title(nome,fontsize=9.5,fontweight="bold"); ax.set_xticks([1,3,6]); ax.set_xlabel("tentativas",fontsize=8)
    axs[0].set_ylabel("melhor acurácia até agora",fontsize=8.5); axs[0].legend(fontsize=6.5,frameon=False,loc="lower right")
    fig.suptitle("Warm start: começar por configurações que funcionaram em problemas parecidos (linha verde = melhor da grade)",fontsize=9.3,fontweight="bold",y=1.03); fig.tight_layout(); save(fig,"f9_warmstart")

def f9_diversidade():
    X,y=make_moons(300,noise=0.32,random_state=3); xx,yy=np.meshgrid(np.linspace(-2,3,240),np.linspace(-1.6,2.1,240))
    ms=[("Árvore",DecisionTreeClassifier(max_depth=4,random_state=0),IND),("k-vizinhos",KNeighborsClassifier(9),CYA),("SVM",SVC(gamma=0.5,probability=True,random_state=0),GRN)]
    ens=VotingClassifier([(n,m) for n,m,_ in ms],voting="soft")
    Xtr,ytr=X[:200],y[:200]; Xte,yte=X[200:],y[200:]
    fig,axs=plt.subplots(1,4,figsize=(8.6,2.7))
    for ax,(n,m,c) in zip(axs,ms+[("Votação",ens,AMB)]):
        m.fit(Xtr,ytr); Z=m.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape); ax.contourf(xx,yy,Z,cmap=ListedColormap(["#C7D2FE","#FDE68A"]),alpha=0.9)
        ax.scatter(Xte[:,0],Xte[:,1],c=np.where(yte==0,IND,AMB),s=9,edgecolors="white",linewidths=0.3)
        ax.set_title(f"{n}\nteste: {m.score(Xte,yte):.2f}",fontsize=8.8,fontweight="bold",color=c); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Modelos diferentes erram em lugares diferentes; a votação combina o melhor de cada um",fontsize=9.6,fontweight="bold",y=1.03); fig.tight_layout(); save(fig,"f9_diversidade")

def f9_stacking():
    fig,ax=canvas(8.6,3.9)
    ax.text(50,97,"Stacking: um modelo que aprende a combinar os outros",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,42,13,22,"Dados",fc=GRY,ec=GRY,fs=9,tc="white",bold=True,r=2)
    for i,(t,c) in enumerate([("Modelo A\n(SVM)",IND),("Modelo B\n(floresta)",CYA),("Modelo C\n(k-NN)",GRN)]):
        y=72-i*26; rbox(ax,22,y-6,19,15,t,fc=c,ec=c,fs=8.4,tc="white",bold=True,r=1.5); arrow(ax,(15.5,53),(21.5,y+1),lw=1.4,ms=9,c=GRY); arrow(ax,(41.5,y+1),(49,53),lw=1.4,ms=9,c=GRY)
    rbox(ax,50,40,20,26,"Previsões dos\nmodelos A, B e C\n(feitas fora\ndo treino)",fc="#EEF2FF",ec=IND,fs=8.2,tc=INK,lw=1.8,r=2)
    arrow(ax,(70.5,53),(77,53),lw=2,ms=11)
    rbox(ax,78,40,20,26,"Meta-modelo\n(ex.: regressão\nlogística)\ndefine os pesos",fc=AMB,ec=AMB,fs=8.2,tc="white",bold=True,r=2)
    ax.text(50,10,"As previsões usadas para treinar o meta-modelo vêm de validação cruzada: assim ele não se deixa enganar por modelos que apenas decoraram o treino.",ha="center",fontsize=7.9,color=GRY,style="italic")
    save(fig,"f9_stacking")

def f9_caruana():
    d=tudo()["ensemble"]; fig,axs=plt.subplots(1,2,figsize=(8.6,3.5),gridspec_kw={"width_ratios":[1.1,1]})
    a=axs[0]; tipos=d["tipos"]; ind=np.array(d["teste_ind"]); cores={"svm":IND,"knn":CYA,"lr":GRN,"rf":AMB,"hgb":RED}
    order=np.argsort(-ind); a.bar(range(len(ind)),ind[order],color=[cores[tipos[j]] for j in order],width=0.85)
    a.axhline(d["ens_teste"],c=INK,lw=2,ls="--"); a.text(len(ind)-1,d["ens_teste"]+0.006,f"ensemble de Caruana: {d['ens_teste']:.3f}",ha="right",fontsize=8,fontweight="bold")
    a.set_ylim(0.5,0.8); a.set_xlabel("os 40 modelos da biblioteca (ordenados pela nota)",fontsize=8.5); a.set_ylabel("acurácia no teste"); a.set_title("Biblioteca de modelos e o ensemble",fontsize=9.5,fontweight="bold")
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=c,label=k) for k,c in cores.items()],fontsize=7.5,frameon=True,facecolor="white",edgecolor="none",loc="center right")
    b=axs[1]; nomes=["melhor modelo\npela validação","média de\ntodos os 40","ensemble\nde Caruana"]; v=[d["melhor_teste"],d["todos_teste"],d["ens_teste"]]
    b.bar(nomes,v,color=[GRY,CYA,IND],width=0.55); b.set_ylim(0.5,0.8); b.set_title("Acurácia no teste final",fontsize=9.5,fontweight="bold")
    for i,x in enumerate(v): b.text(i,x+0.006,f"{x:.3f}",ha="center",fontweight="bold",fontsize=9)
    fig.tight_layout(); save(fig,"f9_caruana")

def f9_autosklearn():
    fig,ax=canvas(8.6,4.0)
    ax.text(50,97,"A arquitetura do Auto-sklearn (Feurer et al., 2015)",ha="center",fontsize=11,fontweight="bold")
    rbox(ax,2,42,18,28,"Novo\nconjunto\nde dados",fc=GRY,ec=GRY,fs=9,tc="white",bold=True,r=2)
    rbox(ax,26,56,22,26,"1. Meta-aprendizado\nconfigurações que\nfuncionaram em\nconjuntos parecidos",fc=IND,ec=IND,fs=8.2,tc="white",bold=True,r=2)
    rbox(ax,26,22,22,26,"Espaço de busca\n15 classificadores,\n14 pré-processadores\nde atributos,\n4 de dados",fc="#EEF2FF",ec=IND,fs=8,tc=INK,lw=1.8,r=2)
    rbox(ax,54,38,22,32,"2. Otimização\nbayesiana (SMAC)\navalia candidatos\ncom validação\ncruzada",fc=CYA,ec=CYA,fs=8.4,tc="white",bold=True,r=2)
    rbox(ax,81,38,17,32,"3. Ensemble\nconstruído com\ntodos os modelos\navaliados",fc=GRN,ec=GRN,fs=8.2,tc="white",bold=True,r=2)
    arrow(ax,(20.5,58),(25.5,66),lw=1.8,ms=10); arrow(ax,(48.5,66),(53.5,58),lw=1.8,ms=10); arrow(ax,(48.5,36),(53.5,44),lw=1.8,ms=10); arrow(ax,(76.5,54),(80.5,54),lw=1.8,ms=10)
    rbox(ax,26,3,72,12,"Saída: um ensemble de modelos, e não apenas o melhor único",fc="#ECFDF5",ec=GRN,fs=8.8,tc=INK,lw=1.8,r=1.5)
    save(fig,"f9_autosklearn")

if __name__=="__main__":
    for fn_ in (f9_medico,f9_rice,f9_metaatributos,f9_metaresultado,f9_warmstart,f9_diversidade,f9_stacking,f9_caruana,f9_autosklearn): fn_()
