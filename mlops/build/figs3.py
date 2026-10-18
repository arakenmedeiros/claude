from fc import *
from matplotlib.colors import LinearSegmentedColormap

def f7_padroes():
    fig,ax=canvas(8.6,4.4,(0,100),(0,52))
    cards=[("LOTE","minutos a horas","lista diária de risco",CYA,2),("TEMPO REAL","milissegundos","aprovar compra no cartão",IND,26),("FLUXO","segundos","fraude em sequência de eventos",AMB,50),("EMBARCADO","imediata, sem rede","voz no celular",GRN,74)]
    for t,l,e,c,x in cards:
        rbox(ax,x,6,23,40,"",fc="white",ec=c,lw=2.4,r=2); rbox(ax,x,38,23,8,t,fc=c,tc="white",bold=True,fs=9.5,r=2)
        ax.text(x+11.5,31,"latência",ha="center",fontsize=7.5,color=GRY); ax.text(x+11.5,26.5,l,ha="center",fontsize=8.3,fontweight="bold")
        ax.text(x+11.5,18,"exemplo",ha="center",fontsize=7.5,color=GRY); ax.text(x+11.5,12,e.replace(" em "," em\n").replace(" no "," no\n").replace("fraude ","fraude\n").replace("lista ","lista\n"),ha="center",fontsize=7.4,va="center")
    ax.text(50,49.5,"quanto antes a resposta, mais complexa a operação",ha="center",fontsize=8.5,style="italic",color=GRY)
    save(fig,"f7_padroes")

def f7_api():
    fig,ax=canvas(8.8,3.6,(0,100),(0,40))
    etapas=[("Cliente\n(app, sistema)",GRY),("POST /previsao\nJSON",CYA),("Contrato\n(Pydantic)",GRN),("Pipeline\n+ modelo",IND),("Resposta\n+ versão",AMB)]
    for k,(t,c) in enumerate(etapas):
        x=2+k*20; rbox(ax,x,14,16,14,t,fc="white",ec=c,lw=2.2,fs=8)
        if k: arrow(ax,(x-4,21),(x,21),INK,lw=1.6,ms=11)
    arrow(ax,(50,14),(50,6),RED,lw=1.6,ms=11); rbox(ax,34,0.5,32,5.5,"422: dados inválidos, o modelo nem é chamado",fc="#FEF2F2",ec=RED,fs=7.6,r=1)
    ax.text(50,35,"modelo carregado uma única vez, na partida do serviço",ha="center",fontsize=8.5,style="italic",color=GRY)
    save(fig,"f7_api")

def f7_conteiner():
    fig,ax=canvas(7.6,4.4,(0,100),(0,56))
    camadas=[("Sistema base (Linux slim)",GRY,6),("Python 3.13",CYA,16),("Dependências (requirements.txt)",GRN,26),("Código (src/)",AMB,36),("Modelo (modelo.joblib)",IND,46)]
    for t,c,y in camadas: rbox(ax,8,y-4,60,8.4,t,fc=c,tc="white",bold=True,fs=8.8,r=1.2,alpha=0.92)
    ax.annotate("",xy=(78,48),xytext=(78,4),arrowprops=dict(arrowstyle="<->",color=INK,lw=1.4))
    ax.text(80,40,"muda\nmuito",fontsize=8,color=RED,va="center"); ax.text(80,12,"muda\npouco",fontsize=8,color=GRN,va="center")
    ax.text(38,53,"IMAGEM = camadas empilhadas (cache reaproveita as de baixo)",ha="center",fontsize=8.6,fontweight="bold")
    save(fig,"f7_conteiner")

def f8_piramide():
    fig,ax=canvas(8,4.6,(0,100),(0,56))
    niv=[("Sistema: ponta a ponta, API",VIO,"poucos · lentos",8),("Modelo: desempenho, fatias",IND,"alguns",8),("Dados: esquema e faixas",GRN,"vários",8),("Código: funções e contrato",CYA,"muitos · rápidos",8)]
    for k,(t,c,s,_) in enumerate(niv):
        y=40-k*12; w=34+k*16; x=50-w/2
        ax.add_patch(Polygon([[x+3,y+10.5],[x+w-3,y+10.5],[x+w+4.5,y-0.5],[x-4.5,y-0.5]],fc=c,ec="white",lw=2,alpha=0.92)) if False else rbox(ax,x,y,w,10.4,t,fc=c,tc="white",fs=7.4,bold=True,r=1.2)
        ax.text(99,y+5,s,ha="right",fontsize=7.8,color=GRY,style="italic",va="center")
    ax.text(50,54,"pirâmide de testes para ML",ha="center",fontsize=10,fontweight="bold")
    save(fig,"f8_piramide")

def f8_cicd():
    fig,ax=canvas(8.8,3.8,(0,100),(0,42))
    et=[("commit",GRY),("estilo\ntipos",CYA),("testes\nde código",CYA),("testes\nde dados",GRN),("treino\nrápido",GRN),("portão de\nqualidade",AMB),("imagem\nversionada",IND),("homologação\n→ produção",RED)]
    for k,(t,c) in enumerate(et):
        x=1+k*12.3; rbox(ax,x,14,10.2,13,t,fc="white",ec=c,lw=2.2,fs=6.9)
        if k: arrow(ax,(x-2.1,20.5),(x,20.5),INK,lw=1.4,ms=9)
    ax.annotate("",xy=(60,8),xytext=(5,8),arrowprops=dict(arrowstyle="-",color=GRY)); 
    ax.text(30,33,"CI: a cada mudança",ha="center",fontsize=9,fontweight="bold",color=CYA); ax.text(80,33,"CD: entrega / implantação",ha="center",fontsize=9,fontweight="bold",color=RED)
    ax.text(50,5,"qualquer etapa que falhar interrompe o fluxo (código de saída ≠ 0)",ha="center",fontsize=8.3,style="italic",color=GRY)
    save(fig,"f8_cicd")

def _mundo(n=18):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    def mes(m,k=1500):
        r=np.random.default_rng(1000+m); X=r.normal(size=(k,4)); X[:,0]+=0.12*m; a=np.deg2rad(5*m); w=np.array([np.cos(a),np.sin(a),0.5,0.0])
        return X,(r.random(k)<1/(1+np.exp(-(X@w)))).astype(int)
    def psi(ref,at,b=10):
        c=np.quantile(ref,np.linspace(0,1,b+1)); c[0],c[-1]=-np.inf,np.inf
        p=np.clip(np.histogram(ref,c)[0]/len(ref),1e-6,None); q=np.clip(np.histogram(at,c)[0]/len(at),1e-6,None); return float(np.sum((q-p)*np.log(q/p)))
    D=[mes(m) for m in range(n)]; mod=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(*D[0])
    return [psi(D[0][0][:,0],D[m][0][:,0]) for m in range(1,n)],[(mod.predict(D[m][0])==D[m][1]).mean() for m in range(1,n)]

def f9_envelhece():
    fig,ax=plt.subplots(figsize=(6.8,3.6)); t=np.arange(0,19); acc=0.92-0.45*(1/(1+np.exp(-(t-11)/2.2)))
    ax.plot(t,acc*100,c=RED,lw=3); ax.fill_between(t,acc*100,50,color=RED,alpha=0.08); ax.axhline(80,ls="--",c=GRY); ax.text(0.3,81,"piso aceitável",fontsize=8,color=GRY)
    ax.annotate("o mundo mudou,\nninguém avisou",xy=(12,acc[12]*100),xytext=(4,60),arrowprops=dict(arrowstyle="->",color=INK),fontsize=8.8)
    ax.set_xlabel("meses desde o treino"); ax.set_ylabel("qualidade das previsões (%)"); ax.set_ylim(45,100); ax.set_title("Um modelo estático envelhece sozinho"); ax.grid(alpha=0.25)
    save(fig,"f9_envelhece")

def f9_camadas():
    fig,ax=canvas(8.4,4.4,(0,100),(0,52))
    cam=[("NEGÓCIO","receita, reclamações, cancelamentos evitados",VIO,"lento · decisivo"),("DESEMPENHO","F1, AUC, por fatias (exige rótulos)",RED,""),("PREVISÕES","distribuição das saídas, confiança",AMB,""),("DADOS DE ENTRADA","ausentes, faixas, PSI, categorias novas",GRN,""),("SERVIÇO","latência p95, erros, CPU, memória",CYA,"rápido · indireto")]
    for k,(t,d,c,s) in enumerate(cam):
        y=40-k*9; w=96-k*8; x=2+k*4
        rbox(ax,x,y,w,8,"",fc=c,alpha=0.17,ec=c,lw=2); ax.text(x+2,y+4,t,fontsize=8.6,fontweight="bold",color=c,va="center"); ax.text(x+w-2,y+4,d,fontsize=7.6,ha="right",va="center")
    ax.text(50,51,"camadas de monitoramento",ha="center",fontsize=10,fontweight="bold")
    save(fig,"f9_camadas")

def f9_tipos():
    rng=np.random.default_rng(0)
    fig,ax=plt.subplots(1,3,figsize=(8.8,3.2))
    X=rng.normal(size=(150,2)); y=(X[:,0]+X[:,1]>0)
    def base(a,X,y,t,line=None):
        a.scatter(X[:,0],X[:,1],c=np.where(y,IND,AMB),s=12,edgecolors="white",lw=0.3); a.set_title(t,fontsize=8.8); a.set_xticks([]); a.set_yticks([]); a.set_xlim(-3.5,5); a.set_ylim(-3.5,4)
        xs=np.linspace(-3.5,5,10)
        if line: a.plot(xs,line(xs),c=INK,ls="--",lw=1.6)
    Xc=X+np.array([1.8,0.2]); base(ax[0],Xc,(Xc[:,0]+Xc[:,1]-2>0),"Covariáveis: P(X) muda",lambda x:-x+2)
    X2=rng.normal(size=(150,2)); y2=rng.random(150)<0.75; X2[y2]+=np.array([0.8,0.8]); X2[~y2]-=np.array([0.8,0.8]); base(ax[1],X2,y2,"Rótulos: P(y) muda (75% azul)",lambda x:-x)
    y3=(X[:,0]-X[:,1]>0); base(ax[2],X,y3,"Conceito: a fronteira gira",lambda x:x)
    fig.tight_layout(); save(fig,"f9_tipos")

def f9_psi():
    rng=np.random.default_rng(0); ref=rng.normal(50,10,5000)
    cen=[("sem mudança",rng.normal(50,10,2000),GRN),("média +3",rng.normal(53,10,2000),AMB),("média +10",rng.normal(60,10,2000),RED),("variância ×2",rng.normal(50,20,2000),VIO)]
    def psi(r,a,b=10):
        c=np.quantile(r,np.linspace(0,1,b+1)); c[0],c[-1]=-np.inf,np.inf
        p=np.clip(np.histogram(r,c)[0]/len(r),1e-6,None); q=np.clip(np.histogram(a,c)[0]/len(a),1e-6,None); return float(np.sum((q-p)*np.log(q/p)))
    fig,ax=plt.subplots(1,4,figsize=(9,2.7),sharey=True)
    for a,(n,d,c) in zip(ax,cen):
        a.hist(ref,bins=40,density=True,color=GRY,alpha=0.55); a.hist(d,bins=40,density=True,color=c,alpha=0.6); a.set_title(f"{n}\nPSI = {psi(ref,d):.2f}",fontsize=8.5); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); save(fig,"f9_psi")

def f9_deriva():
    ps,acc=_mundo(); m=np.arange(1,18)
    fig,ax=plt.subplots(figsize=(7,3.8)); ax.bar(m,ps,color=CYA,alpha=0.55,label="PSI do atributo 0 (sem rótulos)"); ax.axhline(0.25,ls="--",c=CYA); ax.set_ylabel("PSI",color=CYA); ax.set_xlabel("mês de produção")
    a2=ax.twinx(); a2.plot(m,np.array(acc)*100,c=RED,lw=3,marker="o",ms=5,label="acurácia do modelo estático"); a2.set_ylabel("acurácia (%)",color=RED); a2.set_ylim(50,75); a2.spines["right"].set_visible(True)
    ax.annotate("alerta antecipado",xy=(4,ps[3]),xytext=(5.5,2.7),arrowprops=dict(arrowstyle="->"),fontsize=8.5); a2.annotate("queda confirmada\ntarde",xy=(13,acc[12]*100),xytext=(8,53),arrowprops=dict(arrowstyle="->"),fontsize=8.5)
    ax.set_title("O PSI dispara muito antes de a acurácia cair"); save(fig,"f9_deriva")

def f9_ab():
    fig,ax=canvas(8.4,3.6,(0,100),(0,40))
    rbox(ax,2,15,16,12,"Tráfego\nde usuários",fc="white",ec=GRY,lw=2,fs=8.5); rbox(ax,26,15,14,12,"Sorteio\naleatório",fc="#FFF7ED",ec=AMB,lw=2,fs=8.5)
    arrow(ax,(18,21),(26,21),INK); arrow(ax,(40,24),(52,31),INK); arrow(ax,(40,18),(52,10),INK)
    rbox(ax,52,26,20,10,"A: modelo atual (50%)",fc="white",ec=IND,lw=2,fs=8); rbox(ax,52,4,20,10,"B: modelo novo (50%)",fc="white",ec=GRN,lw=2,fs=8)
    arrow(ax,(72,31),(80,23),INK); arrow(ax,(72,9),(80,19),INK); rbox(ax,80,15,18,12,"Comparar\nmétricas de\nnegócio",fc="#ECFDF5",ec=GRN,lw=2,fs=8)
    ax.text(50,38,"tamanho da amostra definido ANTES do teste",ha="center",fontsize=8.5,style="italic",color=GRY)
    save(fig,"f9_ab")

def f10_governanca():
    fig,ax=canvas(8.4,4.4,(0,100),(0,52))
    pil=[("Documen-\ntação",IND),("Explica-\nbilidade",CYA),("Justiça",GRN),("Privacidade",AMB),("Segurança",RED),("Auditoria",VIO)]
    ax.add_patch(Polygon([[4,38],[50,50],[96,38]],fc="#E0E7FF",ec=IND,lw=2)); ax.text(50,42,"GOVERNANÇA DE ML",ha="center",fontsize=10.5,fontweight="bold",color=IND)
    for k,(t,c) in enumerate(pil):
        x=7+k*15; ax.add_patch(Rectangle((x,12),10,25,fc=c,alpha=0.85,ec="white")); ax.text(x+5,30,t,ha="center",fontsize=6.8,color="white",fontweight="bold",va="center")
    rbox(ax,4,3,92,8,"Rastreabilidade: dados, código, modelo, decisões",fc="#F3F4F6",ec=GRY,fs=8.6,bold=True)
    save(fig,"f10_governanca")

def f10_explicacao():
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    d=load_breast_cancer(as_frame=True); X,y=d.data,d.target; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
    m=RandomForestClassifier(200,random_state=0).fit(Xtr,ytr); r=permutation_importance(m,Xte,yte,n_repeats=15,random_state=0,scoring="f1")
    o=np.argsort(r.importances_mean)[::-1][:6][::-1]
    fig,ax=plt.subplots(1,2,figsize=(8.8,3.5),gridspec_kw=dict(width_ratios=[1.1,1]))
    ax[0].barh([X.columns[i] for i in o],r.importances_mean[o],xerr=r.importances_std[o],color=IND,alpha=0.85); ax[0].set_title("Global: queda do F1 ao embaralhar",fontsize=9); ax[0].tick_params(labelsize=7.5)
    nm=["dívidas","renda","tempo de emprego","idade"]; v=[-0.28,0.18,0.08,0.01]; ax[1].barh(nm[::-1],v[::-1],color=[GRN if x>0 else RED for x in v[::-1]]); ax[1].axvline(0,c=INK,lw=1)
    ax[1].set_title("Local: por que ESTE solicitante\nfoi reprovado (ilustrativo)",fontsize=9); ax[1].set_xlabel("contribuição para a probabilidade")
    fig.tight_layout(); save(fig,"f10_explicacao")

def f10_featurestore():
    fig,ax=canvas(8.6,4.2,(0,100),(0,50))
    rbox(ax,2,18,18,14,"Fontes\nde dados",fc="white",ec=GRY,lw=2,fs=8.5); arrow(ax,(20,25),(28,25),INK)
    rbox(ax,28,16,20,18,"Cálculo único\ndos atributos",fc="#ECFEFF",ec=CYA,lw=2.2,fs=8.5,bold=True)
    arrow(ax,(48,29),(58,38),INK); arrow(ax,(48,21),(58,12),INK)
    rbox(ax,58,32,20,12,"Armazém OFFLINE\n(histórico)",fc="white",ec=IND,lw=2,fs=8); rbox(ax,58,4,20,12,"Armazém ONLINE\n(valor atual)",fc="white",ec=GRN,lw=2,fs=8)
    arrow(ax,(78,38),(86,38),INK); rbox(ax,86,32,12,12,"Treino\ne lote",fc="#EEF2FF",ec=IND,fs=8); arrow(ax,(78,10),(86,10),INK); rbox(ax,86,4,12,12,"Serviço\ntempo real",fc="#ECFDF5",ec=GRN,fs=8)
    ax.text(50,47,"mesmos valores no treino e na produção",ha="center",fontsize=9,style="italic",color=GRY)
    save(fig,"f10_featurestore")

def f10_dag():
    fig,ax=canvas(8.6,3.6,(0,100),(0,40))
    nos={"ingestão":(4,15,GRY),"validação":(24,15,GRN),"treino A":(46,26,IND),"treino B":(46,6,IND),"avaliação":(66,15,AMB),"registro":(85,15,VIO)}
    for n,(x,y,c) in nos.items(): rbox(ax,x,y,13,9,n,fc="white",ec=c,lw=2.2,fs=8.5)
    for a,b in [("ingestão","validação"),("validação","treino A"),("validação","treino B"),("treino A","avaliação"),("treino B","avaliação"),("avaliação","registro")]:
        xa,ya,_=nos[a]; xb,yb,_=nos[b]; arrow(ax,(xa+13,ya+4.5),(xb,yb+4.5),INK,lw=1.4,ms=10)
    ax.text(52,36,"treino A e B são independentes: podem rodar em paralelo",ha="center",fontsize=8.3,style="italic",color=GRY)
    save(fig,"f10_dag")

def f10_llmops():
    fig,ax=canvas(8.8,4.2,(0,100),(0,50))
    et=[("Prompt\nversionado",IND,4,32),("Recuperação\nde contexto (RAG)",CYA,26,32),("Modelo\nde linguagem",VIO,50,32),("Guardrails\nentrada/saída",AMB,74,32),("Resposta\nao usuário",GRN,74,8),("Avaliação\ncontínua",RED,50,8),("Monitoramento\nde custo e qualidade",GRY,22,8)]
    for t,c,x,y in et: rbox(ax,x,y,20,11,t,fc="white",ec=c,lw=2.2,fs=7.8)
    for p,q in [((24,37.5),(26,37.5)),((46,37.5),(50,37.5)),((70,37.5),(74,37.5)),((84,32),(84,19)),((74,13.5),(70,13.5)),((50,13.5),(42,13.5))]: arrow(ax,p,q,INK,lw=1.5,ms=10)
    ax.add_patch(FancyArrowPatch((22,13.5),(14,32),arrowstyle="-|>",mutation_scale=11,color=RED,lw=1.6,ls="--",connectionstyle="arc3,rad=-0.3"))
    ax.text(50,47,"o ciclo continua: o que se mede volta ao prompt",ha="center",fontsize=8.5,style="italic",color=GRY)
    save(fig,"f10_llmops")

def f11_etapas():
    fig,ax=canvas(8.8,4.6,(0,100),(0,56))
    et=[("1 Dados","validar · hash","Cap. 3",CYA),("2 Candidatos","base · busca","Cap. 4, 6",CYA),("3 Registro","MLflow · portão","Cap. 5",GRN),("4 Serviço","API · contrato","Cap. 7",GRN),("5 Testes","pytest · CI","Cap. 8",AMB),("6 Produção","simulada","Cap. 9",AMB),("7 Monitorar","PSI · domínio","Cap. 9",RED),("8 Retreino","CT · portão","Cap. 5,8,9",RED),("9 Governança","cartão · relatório","Cap. 10",VIO)]
    for k,(t,s,c,col) in enumerate(et):
        x=2+(k%5)*19.5; y=30 if k<5 else 6
        rbox(ax,x,y,17,19,"",fc="white",ec=col,lw=2.4); ax.text(x+8.5,y+14,t,ha="center",fontsize=8.6,fontweight="bold",color=col); ax.text(x+8.5,y+8.5,s,ha="center",fontsize=7.3); ax.text(x+8.5,y+3.5,c,ha="center",fontsize=7,color=GRY,style="italic")
        if k%5 and True: arrow(ax,(x-2.5,y+9.5),(x,y+9.5),INK,lw=1.3,ms=9)
    arrow(ax,(90,30),(12,25),GRY,lw=1.3,ms=10,rad=0.0)
    ax.text(50,52,"do dado à produção, e de volta ao dado",ha="center",fontsize=10,fontweight="bold")
    save(fig,"f11_etapas")

def f11_producao():
    fig,ax=plt.subplots(figsize=(7.4,3.4)); s=np.arange(1,9)
    nivel=np.minimum(s,4); ax.step(s,nivel,where="mid",c=RED,lw=3,label="degradação da imagem (foco + tinta)")
    ax.set_ylim(0,5.5); ax.set_xlabel("semana de produção"); ax.set_ylabel("nível de degradação"); ax.set_xticks(s)
    for w in s[:-1]: ax.annotate("",xy=(w+1,0.35),xytext=(w,0.35),arrowprops=dict(arrowstyle="->",color=GRN,lw=1.4))
    ax.text(4.5,0.65,"rótulos da semana s chegam na semana s+1",ha="center",fontsize=8,color=GRN)
    ax.axvspan(0.5,4.5,color=AMB,alpha=0.12); ax.text(2.5,5.0,"degradação crescente",ha="center",fontsize=8.5,color=AMB); ax.text(6.5,5.0,"novo regime (estável)",ha="center",fontsize=8.5,color=INK)
    ax.set_title("A linha do tempo da produção simulada"); ax.grid(alpha=0.2); save(fig,"f11_producao")
