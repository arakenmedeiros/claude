from fc import *
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------ Capítulo 3
def f3_qualidade():
    fig,axs=plt.subplots(2,4,figsize=(8.6,4.4),subplot_kw=dict(projection="polar"))
    dims=[("Completude",0.88,GRN),("Validade",0.97,GRN),("Unicidade",0.74,AMB),("Consistência",0.91,GRN),("Atualidade",0.45,RED),("Exatidão",0.80,AMB),("Representatividade",0.62,AMB)]
    for ax,(nm,v,c) in zip(axs.flat,dims):
        ax.set_theta_zero_location("W"); ax.set_theta_direction(-1); ax.set_thetamin(0); ax.set_thetamax(180)
        th=np.linspace(0,np.pi,100); ax.plot(th,np.ones(100),c=MID,lw=9,solid_capstyle="round"); ax.plot(th[:int(v*100)],np.ones(int(v*100)),c=c,lw=9,solid_capstyle="round")
        ax.set_ylim(0,1.3); ax.set_axis_off(); ax.text(np.pi/2,0.1,f"{v:.0%}",ha="center",va="center",fontsize=12,fontweight="bold",color=c); ax.text(np.pi/2,-0.45,nm,ha="center",va="center",fontsize=8.2,fontweight="bold")
    axs.flat[-1].set_axis_off(); axs.flat[-1].text(0.5,0.5,"verde: ok\namarelo: atenção\nvermelho: falha",transform=axs.flat[-1].transAxes,ha="center",va="center",fontsize=8,color=GRY)
    save(fig,"f3_qualidade")

def f3_validacao():
    fig,ax=canvas(8.6,3.8,(0,100),(0,44))
    rbox(ax,1,16,14,12,"Fonte de\ndados",fc="white",ec=GRY,lw=2,fs=8.5)
    arrow(ax,(15,22),(22,22),INK); rbox(ax,22,12,20,20,"PORTÃO\nde validação\n\nesquema\nfaixas\nausentes",fc="#ECFDF5",ec=GRN,lw=2.4,fs=8.2,bold=True)
    arrow(ax,(42,26),(58,34),GRN,lw=2); arrow(ax,(42,18),(58,10),RED,lw=2)
    ax.text(48,33,"aprovado",color=GRN,fontweight="bold",fontsize=8.5); ax.text(48,8,"reprovado",color=RED,fontweight="bold",fontsize=8.5)
    rbox(ax,58,28,17,12,"Treino /\nprevisão",fc="white",ec=GRN,lw=2,fs=8.5); rbox(ax,58,4,17,12,"Quarentena",fc="#FEF2F2",ec=RED,lw=2,fs=8.5)
    arrow(ax,(75,34),(82,34),GRN); rbox(ax,82,28,17,12,"Modelo",fc="white",ec=IND,lw=2,fs=8.5)
    arrow(ax,(75,10),(82,10),RED); rbox(ax,82,4,17,12,"Alerta +\ninvestigação",fc="white",ec=RED,lw=2,fs=8.5)
    save(fig,"f3_validacao")

def f3_rotulos():
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    X,y=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    niv=[0,0.1,0.2,0.3,0.4,0.5,0.6]; acc=[]
    for r in niv:
        rng=np.random.default_rng(1); yr=ytr.copy(); idx=rng.choice(len(ytr),int(r*len(ytr)),replace=False); yr[idx]=rng.integers(0,10,len(idx))
        acc.append(RandomForestClassifier(100,random_state=0).fit(Xtr,yr).score(Xte,yte))
    fig,ax=plt.subplots(figsize=(6.6,3.6)); ax.plot(np.array(niv)*100,np.array(acc)*100,"-o",c=IND,lw=2.5,ms=7)
    for x,a in zip(niv,acc): ax.text(x*100,a*100+1.6,f"{a*100:.0f}",ha="center",fontsize=8)
    ax.set_xlabel("ruído aleatório nos rótulos de treino (%)"); ax.set_ylabel("acurácia no teste limpo (%)"); ax.set_ylim(60,102); ax.grid(alpha=0.25)
    ax.set_title("A floresta tolera ruído moderado, mas não infinito")
    save(fig,"f3_rotulos")

def f3_versionamento():
    fig,ax=canvas(8.6,4.2,(0,100),(0,50))
    rbox(ax,2,24,26,20,"",fc="#FFF7ED",ec=AMB,lw=2); ax.text(15,40.5,"GIT",ha="center",fontweight="bold",color=AMB,fontsize=11)
    for k,t in enumerate(["código v1 · v2 · v3","dados.csv.dvc (ponteiro)","config.json"]): ax.text(3.5,35.5-k*4.2,"• "+t,fontsize=7)
    rbox(ax,38,28,26,16,"",fc="#ECFEFF",ec=CYA,lw=2); ax.text(51,40.5,"DVC",ha="center",fontweight="bold",color=CYA,fontsize=11)
    ax.text(51,34,"calcula o hash\nenvia / traz os dados",ha="center",fontsize=8,va="center")
    rbox(ax,74,28,24,16,"",fc="#F5F3FF",ec=VIO,lw=2); ax.text(86,40.5,"ARMAZÉM REMOTO",ha="center",fontweight="bold",color=VIO,fontsize=9)
    for k,t in enumerate(["dados v1 (hash a3f…)","dados v2 (hash 9c1…)"]): rbox(ax,76,32-k*6,20,4.6,t,fc="white",ec=VIO,fs=7.4,r=1)
    arrow(ax,(28,36),(38,36),INK); arrow(ax,(64,36),(74,36),INK); arrow(ax,(74,31),(64,31),INK,ls="--")
    ax.text(33,39,"ponteiro",fontsize=7.5,ha="center",color=GRY); ax.text(69,39,"dados",fontsize=7.5,ha="center",color=GRY)
    rbox(ax,2,3,96,15,"",fc="#F9FAFB",ec=MID,lw=1.5)
    ax.text(50,14,"Mudar de versão no Git  →  o DVC traz de volta os dados correspondentes",ha="center",fontsize=9.5,fontweight="bold")
    ax.text(50,7.5,"git checkout v2  +  dvc checkout  =  código, configuração e dados da mesma época",ha="center",fontsize=8.3,color=GRY)
    save(fig,"f3_versionamento")

# ------------------------------------------------------------------ Capítulo 4
def f4_linha():
    fig,ax=canvas(8.6,3.6,(0,100),(0,40))
    etapas=[("dados\nbrutos",GRY),("imputar\nausentes",CYA),("codificar\ncategorias",CYA),("padronizar",GRN),("selecionar\natributos",GRN),("modelo",IND),("previsão",AMB)]
    for k,(t,c) in enumerate(etapas):
        x=2+k*13.8; rbox(ax,x,14,11,13,t,fc="white",ec=c,lw=2.2,fs=7.8,r=2)
        if k: arrow(ax,(x-2.8,20.5),(x,20.5),INK,lw=1.6,ms=11)
    ax.plot([13,13+13.8*5+11-11],[11,11],c=IND,lw=0)
    ax.annotate("",xy=(95,10),xytext=(4,10),arrowprops=dict(arrowstyle="<->",color=GRY,lw=1.2)); ax.text(50,6.2,"um único objeto Pipeline: fit() treina tudo; predict() aplica tudo",ha="center",fontsize=8.6,color=GRY,style="italic")
    ax.text(50,33,"o mesmo molde para o treino e para a produção",ha="center",fontsize=10,fontweight="bold",color=IND)
    save(fig,"f4_linha")

def f4_vazamento():
    from sklearn.feature_selection import SelectKBest,f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold,cross_val_score
    from sklearn.pipeline import make_pipeline
    fora,dentro=[],[]
    for s in range(15):
        rng=np.random.default_rng(s); X=rng.normal(size=(100,2000)); y=rng.integers(0,2,100); cv=StratifiedKFold(5,shuffle=True,random_state=s)
        Xs=SelectKBest(f_classif,k=20).fit_transform(X,y); fora.append(cross_val_score(LogisticRegression(max_iter=1000),Xs,y,cv=cv).mean())
        dentro.append(cross_val_score(make_pipeline(SelectKBest(f_classif,k=20),LogisticRegression(max_iter=1000)),X,y,cv=cv).mean())
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    ax.boxplot([np.array(fora)*100,np.array(dentro)*100],widths=0.5,patch_artist=True,tick_labels=["seleção FORA\ndo pipeline (vaza)","seleção DENTRO\ndo pipeline (correto)"],
               boxprops=dict(facecolor="#E0E7FF",edgecolor=IND),medianprops=dict(color=RED,lw=2))
    ax.axhline(50,c=GRY,ls="--",lw=1); ax.text(2.45,50.8,"acaso = 50%",fontsize=8,color=GRY,ha="right")
    ax.set_ylabel("acurácia em validação cruzada (%)"); ax.set_title("Dados aleatórios: não há nada a aprender")
    save(fig,"f4_vazamento")

def f4_coltrans():
    fig,ax=canvas(8.6,4.4,(0,100),(0,52))
    rbox(ax,2,18,15,16,"DataFrame\ncru\n\nidade, renda,\nplano, cidade",fc="white",ec=GRY,fs=7.8,lw=2)
    rbox(ax,28,34,28,12,"numéricas\nmediana → padronizar",fc="#ECFDF5",ec=GRN,fs=8,lw=2); rbox(ax,28,18,28,12,"categóricas\nmoda → one-hot",fc="#ECFEFF",ec=CYA,fs=8,lw=2); rbox(ax,28,2,28,12,"restantes\n(passthrough / descartar)",fc="#F9FAFB",ec=GRY,fs=8,lw=2)
    for y in (40,24,8): arrow(ax,(17,26),(28,y),INK,lw=1.4,ms=10,rad=0.0)
    rbox(ax,66,18,12,16,"concatenar",fc="#FFF7ED",ec=AMB,fs=8,lw=2)
    for y in (40,24,8): arrow(ax,(56,y),(66,26),INK,lw=1.4,ms=10)
    arrow(ax,(78,26),(84,26),INK,lw=1.6); rbox(ax,84,18,14,16,"Modelo\n(floresta,\nregressão…)",fc="#EEF2FF",ec=IND,fs=8,lw=2)
    ax.text(42,49.5,"ColumnTransformer",ha="center",fontsize=10,fontweight="bold",color=INK)
    save(fig,"f4_coltrans")

# ------------------------------------------------------------------ Capítulo 5
def f5_caderno():
    fig,ax=canvas(8.6,4.2,(0,100),(0,50))
    rbox(ax,2,6,38,38,"",fc="#FEF2F2",ec=RED,lw=2); ax.text(21,40,"SEM rastreamento",ha="center",fontweight="bold",color=RED,fontsize=10)
    nomes=["modelo_final","final_v2","final_OK","final2_NOVO","teste1","modelo(3)"]
    for k,nm in enumerate(nomes):
        x=6+(k%2)*16; y=30-(k//2)*8.5; ax.add_patch(Rectangle((x,y),14,6,fc="white",ec=GRY,lw=1,angle=[-4,3,2,-3,4,-2][k])); ax.text(x+7,y+3,nm,fontsize=6.6,ha="center",va="center")
    ax.text(21,11,"? ? ?",ha="center",fontsize=14,color=RED,fontweight="bold")
    ax.text(21,3,"“qual era o melhor?”",ha="center",fontsize=8.5,style="italic",color=RED)
    rbox(ax,50,6,48,38,"",fc="#ECFDF5",ec=GRN,lw=2); ax.text(74,40,"COM rastreamento",ha="center",fontweight="bold",color=GRN,fontsize=10)
    cab=["execução","n_árv.","F1","dados","commit"]
    for j,t in enumerate(cab): ax.text(53.5+j*9.3,35,t,fontsize=6.8,fontweight="bold",color=IND)
    for i in range(5):
        for j,v in enumerate([f"rf_{i+1}",[5,50,100,200,300][i],[.96,.95,.97,.96,.97][i],"a3f9","9d1c"]): ax.text(53.5+j*9.3,30-i*4.6,str(v),fontsize=6.8)
    rbox(ax,50.8,18.1,46,4.4,"",fc="#FDE68A",ec=AMB,lw=1,alpha=0.5)
    ax.text(74,6.8,"tudo registrado e reproduzível",ha="center",fontsize=8.5,style="italic",color=GRN)
    save(fig,"f5_caderno")

def f5_comparacao():
    import os,tempfile,warnings
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold,cross_val_score
    X,y=load_breast_cancer(return_X_y=True); cv=StratifiedKFold(5,shuffle=True,random_state=0)
    ns=[5,20,50,100,200,300]; m=[];s=[]
    for n in ns:
        sc=cross_val_score(RandomForestClassifier(n,random_state=0),X,y,cv=cv,scoring="f1"); m.append(sc.mean()); s.append(sc.std())
    fig,ax=plt.subplots(figsize=(6.6,3.6)); ax.errorbar(range(len(ns)),np.array(m)*100,yerr=np.array(s)*100,fmt="-o",c=IND,lw=2,capsize=4,ms=7)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"rf_{n}" for n in ns]); ax.set_ylabel("F1 em validação cruzada (%)"); ax.set_xlabel("execução registrada (número de árvores)")
    ax.set_title("Comparar execuções: a média e a dispersão"); ax.grid(alpha=0.25)
    ax.text(2.5,93.6,"diferenças menores que a barra podem ser só ruído",fontsize=8,color=GRY,ha="center")
    save(fig,"f5_comparacao")

def f5_registro():
    fig,ax=canvas(8.6,4.2,(0,100),(0,50))
    ax.text(50,47,"modelo registrado: detector_cancer",ha="center",fontsize=10.5,fontweight="bold",color=INK)
    vers=[("v1","F1 0,91",GRY),("v2","F1 0,94",CYA),("v3","F1 0,95",GRN),("v4","F1 0,96",AMB)]
    for k,(v,m,c) in enumerate(vers):
        x=6+k*23; rbox(ax,x,18,19,18,"",fc="white",ec=c,lw=2.4); ax.text(x+9.5,31,v,ha="center",fontsize=15,fontweight="bold",color=c); ax.text(x+9.5,23,m,ha="center",fontsize=8.5)
    rbox(ax,31,2,20,7,"champion",fc=GRN,tc="white",bold=True,fs=9.5,r=3); 
    ax.add_patch(FancyArrowPatch((41,9),(6+2*23+9.5,17.8),arrowstyle="-|>",mutation_scale=14,color=GRN,lw=2.4,connectionstyle="arc3,rad=-0.15"))
    rbox(ax,70,2,20,7,"challenger",fc=AMB,tc="white",bold=True,fs=9.5,r=3); ax.add_patch(FancyArrowPatch((82,9),(6+3*23+9.5,17.8),arrowstyle="-|>",mutation_scale=14,color=AMB,lw=2.4,connectionstyle="arc3,rad=0.1"))
    ax.text(2,5.5,"mover o apelido =\ntrocar ou reverter\nsem mexer no código",fontsize=7.6,style="italic",color=GRY,va="center")
    save(fig,"f5_registro")

# ------------------------------------------------------------------ Capítulo 6
def _dados_raros():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    X,y=make_classification(6000,n_features=12,n_informative=5,weights=[0.95],flip_y=0.01,class_sep=0.8,random_state=0)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
    p=RandomForestClassifier(200,random_state=0).fit(Xtr,ytr).predict_proba(Xte)[:,1]; return yte,p

def f6_curvas():
    from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
    yte,p=_dados_raros(); fpr,tpr,_=roc_curve(yte,p); pr,rc,_=precision_recall_curve(yte,p)
    fig,ax=plt.subplots(1,2,figsize=(8.4,3.6))
    ax[0].plot(fpr,tpr,c=IND,lw=2.5); ax[0].plot([0,1],[0,1],"--",c=GRY); ax[0].fill_between(fpr,tpr,alpha=0.12,color=IND); ax[0].set_title(f"ROC (AUC = {roc_auc_score(yte,p):.2f})"); ax[0].set_xlabel("taxa de falsos positivos"); ax[0].set_ylabel("taxa de verdadeiros positivos")
    ax[1].plot(rc,pr,c=RED,lw=2.5); ax[1].axhline(yte.mean(),ls="--",c=GRY); ax[1].text(0.5,yte.mean()+0.03,f"acaso = {yte.mean():.2f}",fontsize=8,color=GRY); ax[1].fill_between(rc,pr,alpha=0.12,color=RED)
    ax[1].set_title(f"Precisão-revocação (AUC-PR = {average_precision_score(yte,p):.2f})"); ax[1].set_xlabel("revocação"); ax[1].set_ylabel("precisão"); ax[1].set_ylim(0,1.02)
    fig.tight_layout(); save(fig,"f6_curvas")

def f6_calibracao():
    from sklearn.calibration import CalibratedClassifierCV,calibration_curve
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    X,y=make_classification(8000,n_features=15,n_informative=6,flip_y=0.05,class_sep=0.7,random_state=1)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.4,random_state=1,stratify=y)
    fig,ax=plt.subplots(figsize=(5.2,4.4)); ax.plot([0,1],[0,1],"--",c=GRY,label="calibração perfeita")
    for nm,m,c in [("floresta bruta",RandomForestClassifier(200,random_state=0),RED),("floresta calibrada",CalibratedClassifierCV(RandomForestClassifier(200,random_state=0),method="isotonic",cv=5),GRN)]:
        p=m.fit(Xtr,ytr).predict_proba(Xte)[:,1]; f,mp=calibration_curve(yte,p,n_bins=8,strategy="quantile"); ax.plot(mp,f,"-o",c=c,lw=2.4,label=nm)
    ax.set_xlabel("probabilidade prevista"); ax.set_ylabel("frequência observada de positivos"); ax.legend(frameon=False); ax.set_title("Diagrama de confiabilidade"); ax.grid(alpha=0.25)
    save(fig,"f6_calibracao")

def f6_cv():
    fig,ax=canvas(8,3.6,(0,100),(0,36))
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((14+j*14,28-i*5.4),13,4.2,fc=AMB if i==j else "#C7D2FE",ec="white",lw=1.5))
        ax.text(10,30-i*5.4,f"rodada {i+1}",ha="right",va="center",fontsize=8)
        ax.text(86,30-i*5.4,f"F1₍{i+1}₎",va="center",fontsize=8.5,color=INK,fontweight="bold")
    ax.add_patch(Rectangle((14,2),13,3,fc=AMB)); ax.text(29,3.5,"teste",va="center",fontsize=8.5); ax.add_patch(Rectangle((44,2),13,3,fc="#C7D2FE")); ax.text(59,3.5,"treino",va="center",fontsize=8.5)
    ax.text(84,3.5,"média ± desvio",fontsize=8.5,fontweight="bold",color=IND)
    save(fig,"f6_cv")

def f6_aninhada():
    fig,ax=canvas(8.6,4.2,(0,100),(0,48))
    rbox(ax,2,4,96,40,"",fc="#F9FAFB",ec=MID,lw=1.5); ax.text(50,41,"LAÇO EXTERNO: estima o desempenho do procedimento",ha="center",fontweight="bold",color=AMB,fontsize=9.5)
    for j in range(4): ax.add_patch(Rectangle((8+j*21,32),19,4.5,fc=AMB if j==0 else "#C7D2FE",ec="white"))
    ax.text(8,29.2,"dobra externa 1 = teste (laranja); as demais = treino externo (azul)",fontsize=7.8,color=AMB)
    
    rbox(ax,26,7,70,19,"",fc="#EEF2FF",ec=IND,lw=2); ax.text(61,23.2,"LAÇO INTERNO: escolhe hiperparâmetros (busca + validação cruzada)",ha="center",fontsize=8.5,fontweight="bold",color=IND)
    for k,(a,b) in enumerate([("C=1","F1 0,86"),("C=10","F1 0,90"),("C=100","F1 0,88")]): rbox(ax,30+k*21,9,18,9,f"{a}\n{b}",fc="white",ec=IND if k!=1 else GRN,lw=2 if k==1 else 1.2,fs=8)
    arrow(ax,(8+9.5,31.5),(14,26.5),AMB,lw=1.4,ms=9,ls="--"); ax.text(4,18,"treino\nexterno",fontsize=7.8,color=IND,ha="left",va="center")
    save(fig,"f6_aninhada")

def f6_busca():
    rng=np.random.default_rng(0)
    fig,ax=plt.subplots(1,2,figsize=(8,3.5),sharey=True)
    gx,gy=np.meshgrid([0.17,0.5,0.83],[0.17,0.5,0.83]); ax[0].scatter(gx,gy,s=70,c=IND)
    ax[1].scatter(rng.uniform(0.03,0.97,9),rng.uniform(0.03,0.97,9),s=70,c=GRN)
    xs=np.linspace(0,1,200)
    for a,t in zip(ax,["Grade: 9 tentativas, 3 valores distintos do eixo importante","Aleatória: 9 tentativas, 9 valores distintos"]):
        a.set_xlim(0,1); a.set_ylim(0,1); a.set_title(t,fontsize=8.6); a.set_xlabel("hiperparâmetro IMPORTANTE"); a.set_xticks([]); a.set_yticks([])
        a.fill_between(xs,0,0.04+0.0*xs,color=AMB,alpha=0.0)
        a.plot(xs,0.08+0.18*np.exp(-((xs-0.42)/0.12)**2),c=AMB,lw=2.2); 
    ax[0].set_ylabel("hiperparâmetro pouco importante")
    fig.tight_layout(); save(fig,"f6_busca")
